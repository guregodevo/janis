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
// cache it holds no per-token history, so it can be cut back only to a
// mark: a copy of the state kept at a length the session asked for (where
// two prompts parted: the end of a shared system prompt, the end of a
// turn). The session keeps the set small (Forget).
type LinearCache struct {
	Conv  engine.Tensor // [1, K-1, convDim], activation dtype
	State engine.Tensor // [1, Hv, Dv, Dk], float32
	steps int

	marks map[int]linearMark // by length

	// lent is set while the conversation decodes in a batch: its tail and
	// state are row row of the batch's stacked tensors, and Conv and State
	// are nil. The next batched step with the same conversations in the
	// same order reads the stack as it is; anything else takes the row back
	// first (own).
	lent *lentRow
}

type lentRow struct {
	b           engine.Backend
	conv, state engine.Tensor // [B, K-1, convDim], [B, Hv, Dv, Dk]
	row         int
}

// own takes the conversation's row back from the batch it decoded in.
func (c *LinearCache) own() {
	if l := c.lent; l != nil {
		c.Conv, c.State = l.b.Slice(l.conv, 0, l.row, l.row+1), l.b.Slice(l.state, 0, l.row, l.row+1)
		c.lent = nil
	}
}

// linearMark is the state at one earlier length. The forward pass makes new
// tensors each step and never writes into these, so they stay as they were.
type linearMark struct {
	conv, state engine.Tensor
}

func (c *LinearCache) Len() int { return c.steps }

func (c *LinearCache) CanTruncate(n int) bool {
	if n == 0 || n >= c.steps {
		return true
	}
	_, ok := c.marks[n]
	return ok
}

// Truncate returns to the mark at n, or resets when n is 0. The marks past
// n go: the prefix they were made on changes from n.
func (c *LinearCache) Truncate(_ engine.Backend, n int) {
	if n >= c.steps {
		return
	}
	c.own()
	if m, ok := c.marks[n]; ok {
		c.Conv, c.State, c.steps = m.conv, m.state, n
	} else {
		c.Conv, c.State, c.steps = nil, nil, 0
	}
	for at := range c.marks {
		if at > n {
			delete(c.marks, at)
		}
	}
}

func (c *LinearCache) Mark() {
	if c.steps == 0 {
		return
	}
	c.own()
	if c.marks == nil {
		c.marks = map[int]linearMark{}
	}
	c.marks[c.steps] = linearMark{conv: c.Conv, state: c.State}
}

func (c *LinearCache) Forget(n int) { delete(c.marks, n) }

// Carry is the conv tail and the state a decode step replaces.
func (c *LinearCache) Carry() []engine.Tensor {
	c.own()
	if c.Conv == nil || c.State == nil {
		return []engine.Tensor{}
	}
	return []engine.Tensor{c.Conv, c.State}
}

func (c *LinearCache) SetCarry(ts []engine.Tensor) {
	if len(ts) == 2 {
		c.Conv, c.State, c.lent = ts[0], ts[1], nil
	}
}

// Settle: a recorded step's runs advanced the state in place.
func (c *LinearCache) Settle(_ engine.StepReplayer, n int) { c.steps = n }

func (c *LinearCache) Tensors() []engine.Tensor {
	ts := make([]engine.Tensor, 0, 2+2*len(c.marks))
	if c.lent != nil {
		ts = append(ts, c.lent.conv, c.lent.state)
	}
	if c.Conv != nil {
		ts = append(ts, c.Conv)
	}
	if c.State != nil {
		ts = append(ts, c.State)
	}
	for _, m := range c.marks {
		ts = append(ts, m.conv, m.state)
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
	qkv, z, aProj, bProj := d.project(b, x, S)
	ys := d.scan(b, qkv, aProj, bProj, []*LinearCache{cache}, S)
	return d.output(b, x, ys, z, S)
}

// ForwardBatch runs B conversations' next tokens, x [1, B, hidden], one row
// each: the projections once over the B rows, and the conv and the
// recurrence as one op each over the B conversations' stacked states.
func (d *GatedDeltaNet) ForwardBatch(b engine.Backend, x engine.Tensor, caches []*LinearCache) engine.Tensor {
	B := x.Shape()[1]
	qkv, z, aProj, bProj := d.project(b, x, B)
	ys := d.scan(b, qkv, aProj, bProj, caches, 1)
	return d.output(b, x, ys, z, B)
}

// project is the layer's input projections for S rows.
func (d *GatedDeltaNet) project(b engine.Backend, x engine.Tensor, S int) (qkv, z, aProj, bProj engine.Tensor) {
	qkv = d.InQKV.Forward(b, x) // [1, S, convDim]
	z = b.Reshape(d.InZ.Forward(b, x), 1, S, d.VHeads, d.VDim)
	aProj = d.InA.Forward(b, x) // [1, S, Hv]
	bProj = d.InB.Forward(b, x) // [1, S, Hv]
	return qkv, z, aProj, bProj
}

// scan is the conv and the recurrence for len(caches) conversations of S new
// rows each, the rows [1, B*S, ·] in conversation order, each conversation
// on its cache's tail and state -> ys [1, B*S, Hv, Dv]. The B conversations'
// tails and states are stacked into one conv and one scan; one conversation
// builds exactly the graph it always did.
func (d *GatedDeltaNet) scan(b engine.Backend, qkv, aProj, bProj engine.Tensor, caches []*LinearCache, S int) engine.Tensor {
	B := len(caches)
	convDim := 2*d.KHeads*d.KDim + d.VHeads*d.VDim
	keyDim := d.KHeads * d.KDim
	if B > 1 {
		qkv = b.Reshape(qkv, B, S, convDim)
		aProj = b.Reshape(aProj, B, S, d.VHeads)
		bProj = b.Reshape(bProj, B, S, d.VHeads)
	}
	// The B conversations decoded the last step together, in this order:
	// their stacked tails and states are this step's input as they are.
	var lent *lentRow
	if B > 1 {
		lent = caches[0].lentStack(caches)
	}
	for _, c := range caches {
		if lent == nil && c != nil {
			c.own()
		}
	}
	stack := func(part func(c *LinearCache) engine.Tensor, zeros func() engine.Tensor) engine.Tensor {
		var out engine.Tensor
		for _, c := range caches {
			t := part(c)
			if t == nil {
				t = zeros()
			}
			if out == nil {
				out = t
			} else {
				out = b.Concat(out, t, 0)
			}
		}
		return out
	}
	unstack := func(t engine.Tensor, set func(c *LinearCache, t engine.Tensor)) {
		for j, c := range caches {
			if c == nil {
				continue
			}
			if B == 1 {
				set(c, t)
			} else {
				set(c, b.Slice(t, 0, j, j+1))
			}
		}
	}

	// Causal depthwise conv over [conv tail ++ qkv]; keep the new tail. The
	// zero pad is f32 and Concat promotes — cheaper than guessing the
	// activation dtype, and the recurrence casts to f32 anyway.
	var tails engine.Tensor
	if lent != nil {
		tails = lent.conv
	} else {
		tails = stack(func(c *LinearCache) engine.Tensor {
			if c == nil {
				return nil
			}
			return c.Conv
		}, func() engine.Tensor {
			return b.FromFloats(make([]float32, (d.ConvK-1)*convDim), 1, d.ConvK-1, convDim)
		})
	}
	convIn := b.Concat(tails, qkv, 1)
	total := convIn.Shape()[1]
	newTails := b.Slice(convIn, 1, total-(d.ConvK-1), total)
	convOut := b.SiLU(d.Conv(b, convIn)) // [B, S, convDim]

	q := b.Reshape(b.Slice(convOut, 2, 0, keyDim), B, S, d.KHeads, d.KDim)
	k := b.Reshape(b.Slice(convOut, 2, keyDim, 2*keyDim), B, S, d.KHeads, d.KDim)
	v := b.Reshape(b.Slice(convOut, 2, 2*keyDim, convDim), B, S, d.VHeads, d.VDim)

	// Weightless per-head RMSNorm, then the reference's scale split: q carries
	// inv_scale^2, k carries inv_scale (inv_scale = Dk^-1/2).
	invScale := float32(1.0) / sqrtf(float32(d.KDim))
	q = b.ScalarMul(b.RMSNorm(q, d.QKOnes, 1e-6), invScale*invScale)
	k = b.ScalarMul(b.RMSNorm(k, d.QKOnes, 1e-6), invScale)

	// Gates, computed in f32 like the reference: beta = sigmoid(b),
	// g = exp(-exp(A_log) * softplus(a + dt_bias)).
	beta := b.Sigmoid(b.Cast(bProj, engine.F32))                                      // [B, S, Hv]
	g := b.Exp(b.Mul(b.Softplus(b.Cast(b.Add(aProj, d.DtBias), engine.F32)), d.NegA)) // [B, S, Hv]

	var state engine.Tensor
	if lent != nil {
		state = lent.state
	} else {
		state = stack(func(c *LinearCache) engine.Tensor {
			if c == nil {
				return nil
			}
			return c.State
		}, func() engine.Tensor {
			return b.FromFloats(make([]float32, d.VHeads*d.VDim*d.KDim), 1, d.VHeads, d.VDim, d.KDim)
		})
	}

	// The whole recurrence is one fused op (a metal kernel on MLX, a plain
	// loop on CPU); k-head → v-head mapping happens inside it.
	ys, state := b.GatedDeltaScan(
		b.Cast(q, engine.F32), b.Cast(k, engine.F32), b.Cast(v, engine.F32),
		g, beta, state) // ys [B, S, Hv, Dv]
	if B > 1 {
		// Each conversation's row stays in the stack until something else
		// needs it.
		for j, c := range caches {
			c.Conv, c.State, c.steps = nil, nil, c.steps+S
			c.lent = &lentRow{b: b, conv: newTails, state: state, row: j}
		}
	} else {
		unstack(newTails, func(c *LinearCache, t engine.Tensor) { c.Conv = t })
		unstack(state, func(c *LinearCache, t engine.Tensor) { c.State, c.steps = t, c.steps+S })
	}
	if B > 1 {
		ys = b.Reshape(ys, 1, B*S, d.VHeads, d.VDim)
	}
	return ys
}

// lentStack is the stack caches decoded in together last step, rows in
// caches' order; nil when they did not.
func (c *LinearCache) lentStack(caches []*LinearCache) *lentRow {
	if c == nil || c.lent == nil || c.lent.conv.Shape()[0] != len(caches) {
		return nil
	}
	for j, o := range caches {
		if o == nil || o.lent == nil || o.lent.row != j || o.lent.conv != c.lent.conv || o.lent.state != c.lent.state {
			return nil
		}
	}
	return c.lent
}

// output is the gated RMSNorm (norm over Dv, then silu(z) gate in f32) and
// out_proj, for S rows: ys [1, S, Hv, Dv] -> [1, S, hidden].
func (d *GatedDeltaNet) output(b engine.Backend, x, ys, z engine.Tensor, S int) engine.Tensor {
	yn := b.RMSNorm(ys, d.NormW, d.Eps)
	out := b.Mul(b.SiLU(b.Cast(z, engine.F32)), yn) // [1, S, Hv, Dv]
	// Back to the activation dtype, as the reference does (precise_swiglu's
	// astype): left in f32 it made out_proj's output f32, and the residual
	// add promoted the whole stream to f32 from the first layer on.
	if rr, ok := b.(engine.RawReader); ok {
		out = b.Cast(out, rr.DTypeOf(x))
	}
	return d.OutProj.Forward(b, b.Reshape(out, 1, S, d.VHeads*d.VDim))
}

// Conv applies the depthwise causal conv with the checkpoint's [C, K, 1]
// weight over convIn [1, L, C] -> [1, L-K+1, C].
func (d *GatedDeltaNet) Conv(b engine.Backend, convIn engine.Tensor) engine.Tensor {
	return b.Conv1dDepthwise(convIn, d.ConvW)
}

func sqrtf(v float32) float32 { return float32(math.Sqrt(float64(v))) }
