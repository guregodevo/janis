package qwen35

import (
	"fmt"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// prefix is the multimodal checkpoint's text-stack root; the vision tower
// ("vision_tower.*") is deliberately never loaded.
const prefix = "language_model."

// Block is one decoder layer: pre-norm + (DeltaNet | gated attention) +
// residual, then pre-norm + dense MLP + residual.
type Block struct {
	InputNorm *qwen.RMSNorm
	Linear    *GatedDeltaNet // set on linear-attention layers
	Attn      *Attention     // set on full-attention layers
	PostNorm  *qwen.RMSNorm
	FFN       *qwen.MLP
}

// LoadBlock loads decoder layer i. Layer kind follows the reference rule:
// every full_attention_interval-th layer is softmax attention, the rest are
// Gated DeltaNet.
func LoadBlock(b engine.Backend, st *safetensors.File, i int, cfg qwen.Config) (*Block, error) {
	lp := fmt.Sprintf("%smodel.layers.%d", prefix, i)
	inNorm, err := qwen.LoadRMSNorm(b, st, lp+".input_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	postNorm, err := qwen.LoadRMSNorm(b, st, lp+".post_attention_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	ffn, err := qwen.LoadMLP(b, st, lp+".mlp", cfg.GroupSize, cfg.Bits)
	if err != nil {
		return nil, err
	}
	bl := &Block{InputNorm: inNorm, PostNorm: postNorm, FFN: ffn}
	if isLinear(i, cfg) {
		if bl.Linear, err = LoadGatedDeltaNet(b, st, lp+".linear_attn", cfg); err != nil {
			return nil, err
		}
	} else if bl.Attn, err = LoadAttention(b, st, lp+".self_attn", cfg); err != nil {
		return nil, err
	}
	return bl, nil
}

func isLinear(layer int, cfg qwen.Config) bool {
	return (layer+1)%cfg.FullAttnInterval != 0
}

// ForwardCached runs the block with its layer cache and position offset.
func (bl *Block) ForwardCached(b engine.Backend, x engine.Tensor, cache qwen.LayerCache, offset int) engine.Tensor {
	var r engine.Tensor
	h := bl.InputNorm.Forward(b, x)
	if bl.Linear != nil {
		var lc *LinearCache
		if cache != nil {
			lc = cache.(*LinearCache)
		}
		r = bl.Linear.Forward(b, h, lc)
	} else {
		var kv *qwen.KVCache
		if cache != nil {
			kv = cache.(*qwen.KVCache)
		}
		r = bl.Attn.Forward(b, h, kv, offset)
	}
	h = b.Add(x, r)
	return b.Add(h, bl.FFN.Forward(b, bl.PostNorm.Forward(b, h)))
}

// Model is the Qwen3.5 text stack: embedding, hybrid decoder blocks, final
// norm, and an untied quantized lm_head.
type Model struct {
	Embed  *qwen.QuantEmbedding
	Blocks []*Block
	Norm   *qwen.RMSNorm
	LMHead *qwen.QuantLinear
	Cfg    qwen.Config
}

// LoadModel loads the full text stack from a multimodal Qwen3.5 checkpoint.
func LoadModel(b engine.Backend, st *safetensors.File, cfg qwen.Config) (*Model, error) {
	embGS, embBits := cfg.QuantFor(prefix + "model.embed_tokens")
	emb, err := qwen.LoadQuantEmbedding(b, st, prefix+"model.embed_tokens", embGS, embBits)
	if err != nil {
		return nil, err
	}
	blocks := make([]*Block, cfg.Layers)
	for i := range blocks {
		if blocks[i], err = LoadBlock(b, st, i, cfg); err != nil {
			return nil, err
		}
	}
	norm, err := qwen.LoadRMSNorm(b, st, prefix+"model.norm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	m := &Model{Embed: emb, Blocks: blocks, Norm: norm, Cfg: cfg}
	if !cfg.TieWordEmbeddings {
		gs, bits := cfg.QuantFor(prefix + "lm_head")
		if m.LMHead, err = qwen.LoadQuantLinear(b, st, prefix+"lm_head", gs, bits, false); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (m *Model) head(b engine.Backend, h engine.Tensor) engine.Tensor {
	if m.LMHead != nil {
		return m.LMHead.Forward(b, h)
	}
	return m.Embed.AsLinear(b, h)
}

// forwardCachedT is the ForwardFunc for the shared session loop: logits
// [1, 1, vocab] for the last position of a device index tensor.
func (m *Model) forwardCachedT(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []qwen.LayerCache) engine.Tensor {
	h := b.Reshape(m.Embed.Forward(b, idsT), 1, seqLen, m.Cfg.Hidden)
	for i, blk := range m.Blocks {
		var c qwen.LayerCache
		if caches != nil {
			c = caches[i]
		}
		h = blk.ForwardCached(b, h, c, offset)
	}
	h = m.Norm.Forward(b, h)
	h = b.Slice(h, 1, seqLen-1, seqLen)
	return m.head(b, h)
}

// Logits runs a full stateless forward pass, returning last-position logits
// [1, 1, vocab] (test/oracle entry point).
func (m *Model) Logits(b engine.Backend, ids []int32) engine.Tensor {
	caches := m.newCaches()
	return m.forwardCachedT(b, b.FromInt32(ids, len(ids)), len(ids), 0, caches)
}

func (m *Model) newCaches() []qwen.LayerCache {
	caches := make([]qwen.LayerCache, m.Cfg.Layers)
	for i := range caches {
		caches[i] = m.newCache(i)
	}
	return caches
}

func (m *Model) newCache(layer int) qwen.LayerCache {
	if isLinear(layer, m.Cfg) {
		return &LinearCache{}
	}
	return &qwen.KVCache{}
}

// NewSession returns a prefix-reusing session on the shared generate loop.
// The hybrid cache mix means a diverged prefix resets rather than truncates
// (see LinearCache); prefix extension — the multi-turn case — reuses fully.
func (m *Model) NewSession() *qwen.Session {
	return &qwen.Session{
		NLayers:  m.Cfg.Layers,
		Vocab:    m.Cfg.Vocab,
		Forward:  m.forwardCachedT,
		NewCache: m.newCache,
	}
}
