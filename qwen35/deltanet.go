// Package qwen35 implements Qwen3.5's hybrid text stack (model_type
// "qwen3_5"): a multimodal checkpoint whose language tower interleaves Gated
// DeltaNet linear-attention layers with gated softmax attention every
// full_attention_interval-th layer. Only the text stack is loaded — the
// vision tower's weights are never touched. The forward pass mirrors
// mlx-lm's models/qwen3_5.py (the reference this port is verified against).
package qwen35

import (
	"math"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// LinearCache is a Gated DeltaNet layer's cross-step state: the causal-conv
// tail (last K-1 input rows) and the recurrent delta-rule state. Unlike a KV
// cache it holds no per-token history, so it can only ever be reset — never
// cut back to a mid-history prefix.
type LinearCache struct {
	Conv  engine.Tensor // [1, K-1, convDim], activation dtype
	State engine.Tensor // [1, Hv, Dv, Dk], float32
	steps int
}

func (c *LinearCache) Len() int { return c.steps }

func (c *LinearCache) CanTruncate(n int) bool { return n == 0 || n >= c.steps }

func (c *LinearCache) Truncate(_ engine.Backend, n int) {
	if n >= c.steps {
		return
	}
	c.Conv, c.State, c.steps = nil, nil, 0
}

func (c *LinearCache) Tensors() []engine.Tensor {
	ts := make([]engine.Tensor, 0, 2)
	if c.Conv != nil {
		ts = append(ts, c.Conv)
	}
	if c.State != nil {
		ts = append(ts, c.State)
	}
	return ts
}

// GatedDeltaNet is one linear-attention layer: a depthwise causal conv over
// the packed q/k/v projection, the gated delta-rule recurrence over a
// [heads, Dv, Dk] state, and a z-gated RMSNorm before the output projection.
type GatedDeltaNet struct {
	InQKV, InZ, InA, InB *qwen.QuantLinear
	OutProj              *qwen.QuantLinear
	ConvW                engine.Tensor // [convDim, K, 1]
	DtBias               engine.Tensor // [Hv]
	NegA                 engine.Tensor // -exp(A_log), [Hv], f32 (precomputed)
	NormW                engine.Tensor // gated-RMSNorm weight [Dv]
	QKOnes               engine.Tensor // ones [Dk] — weightless q/k RMSNorm

	KHeads, VHeads, KDim, VDim, ConvK int
	Eps                               float32
}

// LoadGatedDeltaNet loads the layer under prefix (".../linear_attn").
func LoadGatedDeltaNet(b engine.Backend, st *safetensors.File, prefix string, cfg qwen.Config) (*GatedDeltaNet, error) {
	load := func(name string) (*qwen.QuantLinear, error) {
		gs, bits := cfg.QuantFor(prefix + "." + name)
		return qwen.LoadQuantLinear(b, st, prefix+"."+name, gs, bits, false)
	}
	d := &GatedDeltaNet{
		KHeads: cfg.LinearKHeads, VHeads: cfg.LinearVHeads,
		KDim: cfg.LinearKDim, VDim: cfg.LinearVDim, ConvK: cfg.ConvKernel,
		Eps: cfg.RMSEps,
	}
	var err error
	if d.InQKV, err = load("in_proj_qkv"); err != nil {
		return nil, err
	}
	if d.InZ, err = load("in_proj_z"); err != nil {
		return nil, err
	}
	if d.InA, err = load("in_proj_a"); err != nil {
		return nil, err
	}
	if d.InB, err = load("in_proj_b"); err != nil {
		return nil, err
	}
	if d.OutProj, err = load("out_proj"); err != nil {
		return nil, err
	}
	if d.ConvW, err = qwen.LoadTensor(b, st, prefix+".conv1d.weight"); err != nil {
		return nil, err
	}
	if d.DtBias, err = qwen.LoadTensor(b, st, prefix+".dt_bias"); err != nil {
		return nil, err
	}
	aLog, err := qwen.LoadTensor(b, st, prefix+".A_log")
	if err != nil {
		return nil, err
	}
	// The decay base -exp(A_log) is a constant of the weights; fold it once.
	d.NegA = b.ScalarMul(b.Exp(b.Cast(aLog, engine.F32)), -1)
	if d.NormW, err = qwen.LoadTensor(b, st, prefix+".norm.weight"); err != nil {
		return nil, err
	}
	ones := make([]float32, d.KDim)
	for i := range ones {
		ones[i] = 1
	}
	d.QKOnes = b.FromFloats(ones, d.KDim)
	return d, nil
}

// Forward runs the layer on x [1, S, hidden] with the layer's cache (nil for
// a stateless single pass), returning [1, S, hidden].
func (d *GatedDeltaNet) Forward(b engine.Backend, x engine.Tensor, cache *LinearCache) engine.Tensor {
	S := x.Shape()[1]
	convDim := 2*d.KHeads*d.KDim + d.VHeads*d.VDim
	keyDim := d.KHeads * d.KDim

	qkv := d.InQKV.Forward(b, x) // [1, S, convDim]
	z := b.Reshape(d.InZ.Forward(b, x), 1, S, d.VHeads, d.VDim)
	aProj := d.InA.Forward(b, x) // [1, S, Hv]
	bProj := d.InB.Forward(b, x) // [1, S, Hv]

	// Causal depthwise conv over [conv tail ++ qkv]; keep the new tail. The
	// zero pad is f32 and Concat promotes — cheaper than guessing the
	// activation dtype, and the recurrence casts to f32 anyway.
	var convIn engine.Tensor
	if cache != nil && cache.Conv != nil {
		convIn = b.Concat(cache.Conv, qkv, 1)
	} else {
		zeros := b.FromFloats(make([]float32, (d.ConvK-1)*convDim), 1, d.ConvK-1, convDim)
		convIn = b.Concat(zeros, qkv, 1)
	}
	if cache != nil {
		total := convIn.Shape()[1]
		cache.Conv = b.Slice(convIn, 1, total-(d.ConvK-1), total)
	}
	convOut := b.SiLU(d.Conv(b, convIn)) // [1, S, convDim]

	q := b.Reshape(b.Slice(convOut, 2, 0, keyDim), 1, S, d.KHeads, d.KDim)
	k := b.Reshape(b.Slice(convOut, 2, keyDim, 2*keyDim), 1, S, d.KHeads, d.KDim)
	v := b.Reshape(b.Slice(convOut, 2, 2*keyDim, convDim), 1, S, d.VHeads, d.VDim)

	// Weightless per-head RMSNorm, then the reference's scale split: q carries
	// inv_scale^2, k carries inv_scale (inv_scale = Dk^-1/2).
	invScale := float32(1.0) / sqrtf(float32(d.KDim))
	q = b.ScalarMul(b.RMSNorm(q, d.QKOnes, 1e-6), invScale*invScale)
	k = b.ScalarMul(b.RMSNorm(k, d.QKOnes, 1e-6), invScale)

	// Gates, computed in f32 like the reference: beta = sigmoid(b),
	// g = exp(-exp(A_log) * softplus(a + dt_bias)).
	beta := b.Sigmoid(b.Cast(bProj, engine.F32))                                      // [1, S, Hv]
	g := b.Exp(b.Mul(b.Softplus(b.Cast(b.Add(aProj, d.DtBias), engine.F32)), d.NegA)) // [1, S, Hv]

	var state engine.Tensor
	if cache != nil && cache.State != nil {
		state = cache.State
	} else {
		state = b.FromFloats(make([]float32, d.VHeads*d.VDim*d.KDim), 1, d.VHeads, d.VDim, d.KDim)
	}

	// The whole recurrence is one fused op (a metal kernel on MLX, a plain
	// loop on CPU); k-head → v-head mapping happens inside it.
	ys, state := b.GatedDeltaScan(
		b.Cast(q, engine.F32), b.Cast(k, engine.F32), b.Cast(v, engine.F32),
		g, beta, state) // ys [1, S, Hv, Dv]
	if cache != nil {
		cache.State = state
		cache.steps += S
	}

	// Gated RMSNorm (norm over Dv, then silu(z) gate in f32), then out_proj.
	yn := b.RMSNorm(ys, d.NormW, d.Eps)
	out := b.Mul(b.SiLU(b.Cast(z, engine.F32)), yn) // [1, S, Hv, Dv]
	return d.OutProj.Forward(b, b.Reshape(out, 1, S, d.VHeads*d.VDim))
}

// Conv applies the depthwise causal conv with the checkpoint's [C, K, 1]
// weight over convIn [1, L, C] -> [1, L-K+1, C].
func (d *GatedDeltaNet) Conv(b engine.Backend, convIn engine.Tensor) engine.Tensor {
	return b.Conv1dDepthwise(convIn, d.ConvW)
}

func sqrtf(v float32) float32 { return float32(math.Sqrt(float64(v))) }
