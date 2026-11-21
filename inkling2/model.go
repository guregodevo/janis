package inkling2

import (
	"fmt"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

const prefix = "model.language_model."

// Model is the Inkling 2 text stack.
type Model struct {
	Cfg       Config
	Embed     *qwen.QuantEmbedding
	EmbedNorm *qwen.RMSNorm
	Layers    []*Layer
	// MessageQueryT are the looped layers' queries into the pass before:
	// one [hidden, 4] per core layer.
	MessageQueryT []engine.Tensor
	FinalQueryT   engine.Tensor
	Norm          *qwen.RMSNorm
	Head          *qwen.QuantLinear

	c *ctx
}

// LoadModel loads the text stack. The backend must have engine.LoopOps.
func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	ops, ok := b.(engine.LoopOps)
	if !ok {
		return nil, fmt.Errorf("%s needs operations this backend does not have; it runs on MLX (Apple Silicon)", cfg.ModelType)
	}
	dt, _, _, err := st.Get(prefix + "embed_norm.weight")
	if err != nil {
		return nil, err
	}
	c := &ctx{Backend: b, ops: ops, ones: map[int]engine.Tensor{}}
	switch dt {
	case "F32":
		c.dtype = engine.F32
	case "F16":
		c.dtype = engine.F16
	case "BF16":
		c.dtype = engine.BF16
	default:
		return nil, fmt.Errorf("embed_norm.weight is %s, not a float", dt)
	}
	c.one = b.FromFloats([]float32{1}, 1)
	for _, n := range []int{cfg.Hidden, cfg.HeadDim, cfg.SWAHeadDim} {
		if c.ones[n] == nil {
			ones := make([]float32, n)
			for i := range ones {
				ones[i] = 1
			}
			c.ones[n] = b.FromFloats(ones, n)
		}
	}

	m := &Model{Cfg: cfg, c: c}
	if m.Embed, err = qwen.LoadQuantEmbedding(b, st, prefix+"embed_tokens", cfg.GroupSize, cfg.Bits); err != nil {
		return nil, err
	}
	ld := loader{c: c, st: st, cfg: cfg}
	m.EmbedNorm = ld.norm(prefix + "embed_norm")
	m.Norm = ld.norm(prefix + "norm")
	m.FinalQueryT = ld.queries(prefix + "final_input.query")
	m.Head = ld.linear("lm_head")
	if core := cfg.CoreEnd - cfg.CoreStart; core > 0 && cfg.Passes > 1 {
		all := ld.f32(prefix + "message_input.query") // [core, 4, hidden]
		for i := 0; i < core && all != nil; i++ {
			q := c.Reshape(c.Slice(all, 0, i, i+1), 4, cfg.Hidden)
			m.MessageQueryT = append(m.MessageQueryT, c.Transpose(q, 1, 0))
		}
	}
	if ld.err != nil {
		return nil, ld.err
	}
	m.Layers = make([]*Layer, cfg.Layers)
	for i := range m.Layers {
		if m.Layers[i], err = loadLayer(c, st, i, cfg); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// hidden runs the decoder on L tokens at positions pos.. and returns the
// state the head reads, [1, L, hidden] in the model dtype. caches has one
// VisitCache per layer visit, in the order of the visits. trace, when set,
// sees the output of every visit.
func (m *Model) hidden(ids engine.Tensor, L, pos int, caches []qwen.LayerCache, trace func(visit int, h engine.Tensor)) engine.Tensor {
	c, cfg := m.c, m.Cfg
	masks := &maskMaker{c: c, cfg: cfg}
	h := m.EmbedNorm.Forward(c, c.Reshape(m.Embed.Forward(c, ids), 1, L, cfg.Hidden))
	if trace != nil {
		trace(-1, h)
	}

	visit := 0
	var blocks []engine.Tensor
	run := func(from, to int, bank []engine.Tensor, last engine.Tensor) {
		for i := from; i < to; i++ {
			var messages engine.Tensor
			if last != nil {
				messages = c.depthAttention(bank, last, m.MessageQueryT[i-from])
			}
			h, blocks = m.Layers[i].Forward(c, h, blocks, caches[visit].(*VisitCache), pos, messages, masks)
			if trace != nil {
				trace(visit, h)
			}
			visit++
		}
	}

	run(0, cfg.CoreStart, nil, nil)
	// Every pass starts from the prelude's output. From the second on, each
	// core layer also reads a message: its own mix of the blocks the pass
	// before closed inside the core and of that pass's final state.
	preH, preBlocks := h, blocks
	for pass := 0; pass < cfg.Passes; pass++ {
		var bank []engine.Tensor
		var last engine.Tensor
		if pass > 0 {
			if from := len(preBlocks) + 1; from < len(blocks) {
				bank = blocks[from:]
			}
			last = c.rms(c.f32(h), defaultEps)
			h, blocks = preH, preBlocks
		}
		run(cfg.CoreStart, cfg.CoreEnd, bank, last)
	}
	run(cfg.CoreEnd, cfg.Layers, nil, nil)

	out := c.pick(c.depthAttention(blocks, c.rms(c.f32(h), defaultEps), m.FinalQueryT), 0)
	return c.model(m.Norm.Forward(c, out))
}

// logits is the head: the softly capped scores of the real vocabulary.
func (m *Model) logits(h engine.Tensor) engine.Tensor {
	c := m.c
	out := c.f32(m.Head.Forward(c, h))
	if cap := m.Cfg.HeadCap; cap > 0 {
		out = c.ScalarMul(c.Tanh(c.ScalarMul(out, 1/cap)), cap)
	}
	return c.Slice(out, 2, 0, m.Cfg.UnpaddedVocab)
}

// forwardCachedT is the session's ForwardFunc: the logits [1, 1, vocab] of
// the last of seqLen tokens.
func (m *Model) forwardCachedT(_ engine.Backend, ids engine.Tensor, seqLen, offset int, caches []qwen.LayerCache) engine.Tensor {
	h := m.hidden(ids, seqLen, offset, caches, nil)
	return m.logits(m.c.Slice(h, 1, seqLen-1, seqLen))
}

func (m *Model) newCaches() []qwen.LayerCache {
	caches := make([]qwen.LayerCache, m.Cfg.Visits())
	for i := range caches {
		caches[i] = &VisitCache{}
	}
	return caches
}

// NewSession returns a session on the shared generate loop, with one cache
// per layer visit.
func (m *Model) NewSession() *qwen.Session {
	return &qwen.Session{
		NLayers:  m.Cfg.Visits(),
		Vocab:    m.Cfg.UnpaddedVocab,
		Forward:  m.forwardCachedT,
		NewCache: func(int) qwen.LayerCache { return &VisitCache{} },
	}
}
