package qwen35

import (
	"math"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
)

// Attention is Qwen3.5's gated softmax attention (every
// full_attention_interval-th layer): q_proj emits [query, gate] per head, RoPE
// rotates only the first partial_rotary_factor of head_dim, and the SDPA
// output is sigmoid-gated before o_proj.
type Attention struct {
	QProj, KProj, VProj, OProj *qwen.QuantLinear
	QNorm, KNorm               *qwen.RMSNorm
	NHeads, NKVHeads, HeadDim  int
	RopeDims                   int
	RopeBase, Scale            float32
}

// LoadAttention loads the projections under prefix (".../self_attn").
func LoadAttention(b engine.Backend, st *safetensors.File, prefix string, cfg qwen.Config) (*Attention, error) {
	load := func(name string) (*qwen.QuantLinear, error) {
		gs, bits := cfg.QuantFor(prefix + "." + name)
		return qwen.LoadQuantLinear(b, st, prefix+"."+name, gs, bits, false)
	}
	a := &Attention{
		NHeads: cfg.NHeads, NKVHeads: cfg.NKVHeads, HeadDim: cfg.HeadDim,
		RopeDims: int(float32(cfg.HeadDim) * cfg.PartialRotary),
		RopeBase: cfg.RopeBase,
		Scale:    float32(1.0 / math.Sqrt(float64(cfg.HeadDim))),
	}
	var err error
	if a.QProj, err = load("q_proj"); err != nil {
		return nil, err
	}
	if a.KProj, err = load("k_proj"); err != nil {
		return nil, err
	}
	if a.VProj, err = load("v_proj"); err != nil {
		return nil, err
	}
	if a.OProj, err = load("o_proj"); err != nil {
		return nil, err
	}
	if a.QNorm, err = qwen.LoadRMSNorm(b, st, prefix+".q_norm", cfg.RMSEps); err != nil {
		return nil, err
	}
	if a.KNorm, err = qwen.LoadRMSNorm(b, st, prefix+".k_norm", cfg.RMSEps); err != nil {
		return nil, err
	}
	return a, nil
}

// Forward runs gated attention on x [1, L, hidden] with the layer's KV cache
// at the given RoPE offset -> [1, L, hidden].
func (a *Attention) Forward(b engine.Backend, x engine.Tensor, cache *qwen.KVCache, offset int) engine.Tensor {
	L := x.Shape()[1]

	// q_proj packs [query, gate] per head: [1, L, H, 2*hd] -> two [1, L, H, hd].
	qg := b.Reshape(a.QProj.Forward(b, x), 1, L, a.NHeads, 2*a.HeadDim)
	q := b.Slice(qg, 3, 0, a.HeadDim)
	gate := b.Slice(qg, 3, a.HeadDim, 2*a.HeadDim)

	k := b.Reshape(a.KProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim)
	q = a.QNorm.Forward(b, q)
	k = a.KNorm.Forward(b, k)
	q = b.Transpose(q, 0, 2, 1, 3)
	k = b.Transpose(k, 0, 2, 1, 3)
	v := b.Transpose(b.Reshape(a.VProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim), 0, 2, 1, 3)

	// Partial RoPE: only the first RopeDims of head_dim rotate. (The text-only
	// path of Qwen3.5's mrope: all three position streams are the token index,
	// which reduces to plain RoPE over the partial dims — mlx-lm does the same.)
	q = b.RoPE(q, a.RopeDims, false, a.RopeBase, 1.0, offset)
	k = b.RoPE(k, a.RopeDims, false, a.RopeBase, 1.0, offset)

	if cache != nil {
		k, v = cache.Update(b, k, v)
	}

	out := b.SDPA(q, k, v, a.Scale, L > 1) // [1, H, L, hd]
	out = b.Reshape(b.Transpose(out, 0, 2, 1, 3), 1, L, a.NHeads*a.HeadDim)
	out = b.Mul(out, b.Sigmoid(b.Reshape(gate, 1, L, a.NHeads*a.HeadDim)))
	return a.OProj.Forward(b, out)
}
