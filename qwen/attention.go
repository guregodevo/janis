package qwen

import (
	"math"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// Attention is a Qwen2 grouped-query attention block: quantized q/k/v/o
// projections (q/k/v carry a bias, o does not), RoPE on q/k, and fused SDPA.
type Attention struct {
	QProj, KProj, VProj, OProj *QuantLinear
	NHeads, NKVHeads, HeadDim  int
	RopeBase, Scale            float32
	// QNorm/KNorm are the per-head Q/K RMSNorms in Qwen3 (nil for Qwen2).
	QNorm, KNorm *RMSNorm
	// RopeFreqs holds precomputed llama3-scaled frequencies (nil = plain RoPE).
	RopeFreqs engine.Tensor
}

// llama3Freqs computes the rope-scaled per-dim frequencies (mlx-lm Llama3RoPE).
func llama3Freqs(headDim int, base, factor, lowF, highF float32, origCtx int) []float32 {
	half := headDim / 2
	out := make([]float32, half)
	lowWavelen := float64(origCtx) / float64(lowF)
	highWavelen := float64(origCtx) / float64(highF)
	for i := 0; i < half; i++ {
		freq := math.Pow(float64(base), float64(2*i)/float64(headDim))
		wavelen := 2 * math.Pi * freq
		f := freq
		if wavelen > lowWavelen {
			f = freq * float64(factor)
		}
		if wavelen > highWavelen && wavelen < lowWavelen {
			sf := (float64(origCtx)/wavelen - float64(lowF)) / (float64(highF) - float64(lowF))
			f = f / ((1-sf)/float64(factor) + sf)
		}
		out[i] = float32(f)
	}
	return out
}

// LoadAttention loads the four projections under prefix (e.g.
// "model.layers.0.self_attn"), with q/k/v bias controlled by the config.
func LoadAttention(b engine.Backend, st *safetensors.File, prefix string, cfg Config) (*Attention, error) {
	load := func(name string, bias bool) (*QuantLinear, error) {
		gs, bits := cfg.QuantFor(prefix + "." + name)
		return LoadQuantLinear(b, st, prefix+"."+name, gs, bits, bias)
	}
	q, err := load("q_proj", cfg.AttentionBias)
	if err != nil {
		return nil, err
	}
	k, err := load("k_proj", cfg.AttentionBias)
	if err != nil {
		return nil, err
	}
	v, err := load("v_proj", cfg.AttentionBias)
	if err != nil {
		return nil, err
	}
	o, err := load("o_proj", false)
	if err != nil {
		return nil, err
	}
	a := &Attention{
		QProj: q, KProj: k, VProj: v, OProj: o,
		NHeads: cfg.NHeads, NKVHeads: cfg.NKVHeads, HeadDim: cfg.HeadDim,
		RopeBase: cfg.RopeBase, Scale: float32(1.0 / math.Sqrt(float64(cfg.HeadDim))),
	}
	// llama3 rope scaling: precompute the scaled frequencies once per layer.
	if cfg.RopeScaling == "llama3" {
		freqs := llama3Freqs(cfg.HeadDim, cfg.RopeBase, cfg.RopeFactor, cfg.RopeLowFreq, cfg.RopeHighFreq, cfg.RopeOrigMaxPos)
		a.RopeFreqs = b.FromFloats(freqs, len(freqs))
	}
	// Qwen3 adds a per-head RMSNorm on Q and K (over head_dim).
	if cfg.QKNorm {
		if a.QNorm, err = LoadRMSNorm(b, st, prefix+".q_norm", cfg.RMSEps); err != nil {
			return nil, err
		}
		if a.KNorm, err = LoadRMSNorm(b, st, prefix+".k_norm", cfg.RMSEps); err != nil {
			return nil, err
		}
	}
	return a, nil
}

// Forward runs attention on x [B, L, hidden] -> [B, L, hidden] (no cache).
func (a *Attention) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return a.ForwardCached(b, x, nil, 0)
}

// ForwardCached runs attention with an optional KV cache and RoPE position
// offset. When cache is non-nil, k/v are appended and attention runs over the
// full history. The causal mask is used only while processing multiple tokens
// (prefill); a single decode token attends to all cached positions unmasked.
func (a *Attention) ForwardCached(b engine.Backend, x engine.Tensor, cache *KVCache, offset int) engine.Tensor {
	B, L := x.Shape()[0], x.Shape()[1]

	q := b.Reshape(a.QProj.Forward(b, x), B, L, a.NHeads, a.HeadDim)
	k := b.Reshape(a.KProj.Forward(b, x), B, L, a.NKVHeads, a.HeadDim)
	// Qwen3: normalize each head's Q/K (over head_dim) before transpose+RoPE.
	if a.QNorm != nil {
		q = a.QNorm.Forward(b, q)
		k = a.KNorm.Forward(b, k)
	}
	q = b.Transpose(q, 0, 2, 1, 3)
	k = b.Transpose(k, 0, 2, 1, 3)
	v := b.Transpose(b.Reshape(a.VProj.Forward(b, x), B, L, a.NKVHeads, a.HeadDim), 0, 2, 1, 3)

	if a.RopeFreqs != nil {
		q = b.RoPEFreqs(q, a.HeadDim, false, 1.0, offset, a.RopeFreqs)
		k = b.RoPEFreqs(k, a.HeadDim, false, 1.0, offset, a.RopeFreqs)
	} else {
		q = b.RoPE(q, a.HeadDim, false, a.RopeBase, 1.0, offset)
		k = b.RoPE(k, a.HeadDim, false, a.RopeBase, 1.0, offset)
	}

	if cache != nil {
		k, v = cache.Update(b, k, v)
	}

	out := b.SDPA(q, k, v, a.Scale, L > 1) // [B, NHeads, L, head_dim]

	out = b.Reshape(b.Transpose(out, 0, 2, 1, 3), B, L, a.NHeads*a.HeadDim)
	return a.OProj.Forward(b, out)
}
