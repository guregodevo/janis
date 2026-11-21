package inkling2

import (
	"container/list"
	"fmt"
	"sort"
	"time"

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
	Head          projection

	// draft is the draft head, nil without one: with it a session's caches
	// end with a DraftCache and its decoding is speculative.
	draft *Draft

	// images are the rows that stand in for the image tokens of the prompt
	// being read, and imageAt the positions of those tokens, in order.
	images  engine.Tensor
	imageAt []int

	c *ctx
}

// SetImages gives the rows [n, hidden] to read in place of the embeddings
// of the tokens at positions (ascending, n of them) of the prompts read
// from now on; nil rows clears them. The rows live until the next call.
func (m *Model) SetImages(rows engine.Tensor, positions []int) {
	sw, _ := m.c.Backend.(engine.Sweeper)
	if sw != nil && m.images != nil {
		sw.Unpin(m.images)
	}
	m.images, m.imageAt = rows, positions
	if sw != nil && rows != nil {
		sw.Pin(rows)
	}
}

// splice replaces, in the embeddings h of the tokens at pos.., the rows of
// the image tokens.
func (m *Model) splice(h engine.Tensor, pos, L int) engine.Tensor {
	c := m.c
	first := sort.SearchInts(m.imageAt, pos)
	if m.images == nil || first == len(m.imageAt) || m.imageAt[first] >= pos+L {
		return h
	}
	var out engine.Tensor
	add := func(piece engine.Tensor) {
		if out == nil {
			out = piece
		} else {
			out = c.Concat(out, piece, 1)
		}
	}
	at := 0 // tokens of h already placed
	for k := first; k < len(m.imageAt) && m.imageAt[k] < pos+L; {
		run := k
		for run+1 < len(m.imageAt) && m.imageAt[run+1] == m.imageAt[run]+1 && m.imageAt[run+1] < pos+L {
			run++
		}
		i := m.imageAt[k] - pos
		if i > at {
			add(c.Slice(h, 1, at, i))
		}
		n := run - k + 1
		add(c.model(c.Reshape(c.Slice(m.images, 0, k, k+n), 1, n, m.Cfg.Hidden)))
		at, k = i+n, run+1
	}
	if at < L {
		add(c.Slice(h, 1, at, L))
	}
	return out
}

// LoadModel loads the text stack. The backend must have engine.LoopOps.
func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	ops, ok := b.(engine.LoopOps)
	if !ok {
		return nil, fmt.Errorf("%s needs operations this backend does not have; it runs on MLX (Apple Silicon)", cfg.ModelType)
	}
	if _, _, err := st.Info("language_model.model.embed_norm.weight"); err == nil {
		cfg.AltLayout = true
	}
	names := loader{st: st, cfg: cfg}
	dt, _, err := st.Info(names.name(prefix + "embed_norm.weight"))
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
	if cfg.StreamExperts {
		c.experts = &expertCache{budget: cfg.ExpertCacheBytes, lru: list.New()}
	}
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
	ld := loader{c: c, st: st, cfg: cfg}
	if m.Embed, err = qwen.LoadQuantEmbedding(b, st, ld.name(prefix+"embed_tokens"), cfg.GroupSize, ld.bits(prefix+"embed_tokens")); err != nil {
		return nil, err
	}
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
	h = m.splice(h, pos, L)
	if trace != nil {
		trace(-1, h)
	}

	// The draft head reads the model's state entering some visits, and the
	// final one.
	var dc *DraftCache
	var taps []engine.Tensor
	if m.draft != nil && len(caches) == cfg.Visits()+1 {
		dc, _ = caches[cfg.Visits()].(*DraftCache)
	}
	tap := func(visit int, h engine.Tensor) {
		if dc == nil {
			return
		}
		for _, t := range m.draft.Cfg.Targets {
			if t == visit {
				taps = append(taps, h)
			}
		}
	}

	visit := 0
	var blocks []engine.Tensor
	run := func(from, to int, bank []engine.Tensor, last engine.Tensor) {
		for i := from; i < to; i++ {
			var messages engine.Tensor
			if last != nil {
				messages = c.depthAttention(bank, last, m.MessageQueryT[i-from])
			}
			tap(visit, h)
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
	final := c.model(m.Norm.Forward(c, out))
	tap(visit, final)
	if dc != nil && len(taps) == len(m.draft.Cfg.Targets) {
		states := taps[0]
		for _, t := range taps[1:] {
			states = c.Concat(states, t, 2)
		}
		dc.record(c, states)
	}
	return final
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

// forwardAllT is forwardCachedT for every position: [1, seqLen, vocab].
func (m *Model) forwardAllT(_ engine.Backend, ids engine.Tensor, seqLen, offset int, caches []qwen.LayerCache) engine.Tensor {
	return m.logits(m.hidden(ids, seqLen, offset, caches, nil))
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
	// With the experts on disk a chunk of prompt reads the union of its
	// tokens' experts for every visit; one token at a time reads 8.
	chunk := 0
	if m.Cfg.StreamExperts {
		chunk = 1
	}
	s := &qwen.Session{
		PrefillChunk: chunk,
		NLayers:      m.Cfg.Visits(),
		Vocab:        m.Cfg.UnpaddedVocab,
		Forward:      m.forwardCachedT,
		NewCache:     func(int) qwen.LayerCache { return &VisitCache{} },
	}
	if m.draft != nil {
		visits := m.Cfg.Visits()
		s.NLayers = visits + 1
		s.NewCache = func(i int) qwen.LayerCache {
			if i == visits {
				return &DraftCache{}
			}
			return &VisitCache{}
		}
		s.Speculator, s.ForwardAll = m.draft, m.forwardAllT
	}
	return s
}

// ExpertIO is what reading experts from disk has cost so far: bytes read
// and the time spent waiting for them. Zero when the experts are resident.
func (m *Model) ExpertIO() (bytes int64, dur time.Duration) {
	var nanos int64
	for _, l := range m.Layers {
		for _, e := range l.MLP.stacks() {
			bytes += e.readBytes
			nanos += e.readNanos
		}
	}
	return bytes, time.Duration(nanos)
}

// ExpertCache is how the cache of streamed experts has served: the experts
// found on the device, the ones read from disk, and the bytes it holds.
func (m *Model) ExpertCache() (hits, misses, bytes int64) {
	if m.c.experts == nil {
		return 0, 0, 0
	}
	return m.c.experts.hits, m.c.experts.misses, m.c.experts.bytes
}
