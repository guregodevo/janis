package inkling2

import (
	"math"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// Attention is one layer's self-attention. Queries and keys are L2-clipped
// instead of rotated; position enters as a bias each query computes for
// itself over the distances up to RelExtent (r_proj, rel_logits_proj); a
// sink logit per head joins the softmax and takes its share of the weight
// away; the output is gated per channel before o_proj.
//
// A sliding layer sees the last Window tokens and filters its keys and
// values with a static short convolution. A full layer sees everything,
// computes the filter's weights from its input, and scales its queries
// with the log of the position.
type Attention struct {
	Q, K, V, R, O, Gate, Sink *qwen.QuantLinear
	QScale, RelScale          engine.Tensor // float32 [heads, head dim], [heads, d_rel]
	RelProj                   engine.Tensor // [d_rel, RelExtent]

	// Sliding layers.
	KConv, VConv engine.Tensor // [channels, kernel, 1]
	// Full layers: the static part of the filter, float32
	// [kv heads, head dim, kernel], and the projection of the dynamic part.
	KStatic, VStatic engine.Tensor
	KDyn, VDyn       *qwen.QuantLinear

	Heads, KVHeads, HeadDim int
	Window                  int // 0 on a full layer
	RelExtent, Kernel       int
}

func loadAttention(c *ctx, st *safetensors.File, prefix string, cfg Config, sliding bool) (*Attention, error) {
	a := &Attention{
		Heads: cfg.Heads, KVHeads: cfg.KVHeads, HeadDim: cfg.HeadDim,
		RelExtent: cfg.RelExtent, Kernel: cfg.ConvKernel,
	}
	if sliding {
		a.Heads, a.KVHeads, a.HeadDim = cfg.SWAHeads, cfg.SWAKVHeads, cfg.SWAHeadDim
		a.Window, a.RelExtent = cfg.Window, cfg.Window
	}
	ld := loader{c: c, st: st, cfg: cfg}
	a.Q = ld.linear(prefix + ".q_proj")
	a.K = ld.linear(prefix + ".k_proj")
	a.V = ld.linear(prefix + ".v_proj")
	a.R = ld.linear(prefix + ".r_proj")
	a.O = ld.linear(prefix + ".o_proj")
	a.Gate = ld.linear(prefix + ".gate_proj")
	a.Sink = ld.linear(prefix + ".sink_proj")
	a.QScale = ld.f32(prefix + ".q_l2_clip.scale")
	a.RelScale = ld.f32(prefix + ".rel_l2_clip.scale")
	a.RelProj = ld.tensor(prefix + ".rel_logits_proj.proj")
	if sliding {
		a.KConv = ld.tensor(prefix + ".k_sconv.conv1d.weight")
		a.VConv = ld.tensor(prefix + ".v_sconv.conv1d.weight")
	} else {
		static := func(name string) engine.Tensor {
			w := ld.f32(name) // [kernel, channels]
			if w == nil {
				return nil
			}
			return c.Reshape(c.Transpose(w, 1, 0), a.KVHeads, a.HeadDim, a.Kernel)
		}
		a.KStatic = static(prefix + ".k_sconv.static_weight")
		a.VStatic = static(prefix + ".v_sconv.static_weight")
		a.KDyn = ld.linear(prefix + ".k_sconv.weight_proj")
		a.VDyn = ld.linear(prefix + ".v_sconv.weight_proj")
	}
	return a, ld.err
}

// dynamicConv is the full layers' short convolution: the filter of each
// position is a static weight plus one its own input chooses, per head.
// x is [1, L, C], in the layer input the projection x was made from.
func (a *Attention) dynamicConv(c *ctx, x, in, static engine.Tensor, dyn *qwen.QuantLinear, tail engine.Tensor) (engine.Tensor, engine.Tensor) {
	L, K, H, D := x.Shape()[1], a.Kernel, a.KVHeads, a.HeadDim
	weights := c.Reshape(c.f32(dyn.Forward(c, in)), 1, L, H, 1, K)
	weights = c.Add(weights, static) // [1, L, H, D, K]
	window := c.convWindow(x, tail, K)
	var out engine.Tensor
	for lag := 0; lag < K; lag++ {
		at := K - 1 - lag
		rows := c.Reshape(c.Slice(window, 1, at, at+L), 1, L, H, D)
		term := c.Mul(rows, c.Reshape(c.Slice(weights, 4, lag, lag+1), 1, L, H, D))
		if out == nil {
			out = term
		} else {
			out = c.Add(out, term)
		}
	}
	out = c.Add(c.Reshape(out, 1, L, H*D), c.f32(x))
	out = c.ScalarMul(out, float32(1/math.Sqrt(float64(2*K+1))))
	return c.model(out), c.tailOf(window, K)
}

// Forward runs attention for L new tokens at positions pos.. on the query,
// key and value inputs [1, L, hidden] (the same tensor on a sliding layer
// outside the loop), reading and extending the visit's cache.
func (a *Attention) Forward(c *ctx, qIn, kIn, vIn engine.Tensor, cache *VisitCache, pos int, masks *maskMaker, eps float32) engine.Tensor {
	L, H, G, D := qIn.Shape()[1], a.Heads, a.KVHeads, a.HeadDim
	var keys, values engine.Tensor
	if a.Window > 0 {
		keys, cache.Conv[0] = c.shortConv(a.K.Forward(c, kIn), a.KConv, cache.Conv[0])
		values, cache.Conv[1] = c.shortConv(a.V.Forward(c, vIn), a.VConv, cache.Conv[1])
	} else {
		keys, cache.Conv[0] = a.dynamicConv(c, a.K.Forward(c, kIn), kIn, a.KStatic, a.KDyn, cache.Conv[0])
		values, cache.Conv[1] = a.dynamicConv(c, a.V.Forward(c, vIn), vIn, a.VStatic, a.VDyn, cache.Conv[1])
	}
	gate := a.Gate.Forward(c, qIn)
	sinks := a.Sink.Forward(c, qIn) // [1, L, H]

	q := c.model(c.l2clip(c.Reshape(a.Q.Forward(c, qIn), 1, L, H, D), a.QScale, masks.cfg.QKClip))
	k := c.model(c.l2clip(c.Reshape(keys, 1, L, G, D), nil, masks.cfg.QKClip))
	v := c.model(c.rms(c.f32(c.Reshape(values, 1, L, G, D)), eps))
	rel := c.model(c.l2clip(c.Reshape(a.R.Forward(c, qIn), 1, L, H, masks.cfg.DRel), a.RelScale, masks.cfg.RelClip))
	if a.Window == 0 {
		tau := masks.tau(pos, L)
		q = c.model(c.Mul(c.f32(q), tau))
		rel = c.model(c.Mul(c.f32(rel), tau))
	}
	bias := c.MatMul(rel, a.RelProj) // [1, L, H, RelExtent]

	q = c.Transpose(q, 0, 2, 1, 3)
	k, v = cache.update(c, c.Transpose(k, 0, 2, 1, 3), c.Transpose(v, 0, 2, 1, 3), max(a.Window-1, 0))
	S := k.Shape()[2]

	// The bias of each query for each key is the entry at their distance;
	// past RelExtent it is zero, and a key the query may not see is -inf.
	m := masks.get(a.Window, a.RelExtent, pos, L, S)
	bias = c.f32(c.ops.TakeAlong(c.Transpose(bias, 0, 2, 1, 3), m.index, 3)) // [1, H, L, S]
	bias = c.Add(c.Mul(bias, m.near), m.hidden)

	per := H / G
	scores := c.MatMul(
		c.Reshape(c.f32(q), 1, G, per, L, D),
		c.Transpose(c.ExpandDims(c.f32(k), 2), 0, 1, 2, 4, 3),
	) // [1, G, per, L, S]
	scale := float32(1 / math.Sqrt(float64(D)))
	scores = c.Add(c.ScalarMul(scores, scale), c.Reshape(bias, 1, G, per, L, S))
	sink := c.Reshape(c.Transpose(c.f32(sinks), 0, 2, 1), 1, G, per, L, 1)
	weights := c.Slice(c.Softmax(c.Concat(scores, sink, 4), 4), 4, 0, S)
	out := c.MatMul(c.model(weights), c.ExpandDims(v, 2)) // [1, G, per, L, D]
	out = c.Reshape(c.Transpose(c.Reshape(out, 1, H, L, D), 0, 2, 1, 3), 1, L, H*D)
	out = c.Mul(c.f32(out), c.Sigmoid(c.f32(gate)))
	return a.O.Forward(c, c.model(out))
}

// mask is what the attention of one kind of layer adds to its scores in a
// forward pass: where to read each query's bias, and what to make of it.
type mask struct {
	index  engine.Tensor // int32 [1, 1, L, S]: distance, clipped into the bias
	near   engine.Tensor // float32 [L, S]: 1 where the distance has a bias, else 0
	hidden engine.Tensor // float32 [L, S]: 0 where the query sees the key, else -inf
}

// maskMaker builds the masks of a forward pass once per kind of layer:
// every sliding layer of a pass has the same, and every full one.
type maskMaker struct {
	c     *ctx
	cfg   Config
	masks map[[2]int]mask
	taus  engine.Tensor
}

func (mm *maskMaker) get(window, extent, pos, L, S int) mask {
	key := [2]int{window, S}
	if m, ok := mm.masks[key]; ok {
		return m
	}
	index := make([]int32, L*S)
	near := make([]float32, L*S)
	hidden := make([]float32, L*S)
	ninf := float32(math.Inf(-1))
	first := pos + L - S // position of the oldest key
	for i := 0; i < L; i++ {
		for j := 0; j < S; j++ {
			d := (pos + i) - (first + j)
			at := i*S + j
			if d < 0 || (window > 0 && d >= window) {
				hidden[at] = ninf
			}
			if d >= 0 && d < extent {
				near[at] = 1
				index[at] = int32(d)
			}
		}
	}
	m := mask{
		index:  mm.c.FromInt32(index, 1, 1, L, S),
		near:   mm.c.FromFloats(near, L, S),
		hidden: mm.c.FromFloats(hidden, L, S),
	}
	if mm.masks == nil {
		mm.masks = map[[2]int]mask{}
	}
	mm.masks[key] = m
	return m
}

// tau is the full layers' query scale, 1 + alpha*ln(max((p+1)/floor, 1)) at
// position p: [1, L, 1, 1] float32.
func (mm *maskMaker) tau(pos, L int) engine.Tensor {
	if mm.taus != nil {
		return mm.taus
	}
	t := make([]float32, L)
	for i := range t {
		r := float64(pos+i+1) / float64(mm.cfg.NFloor)
		t[i] = float32(1 + float64(mm.cfg.Alpha)*math.Log(math.Max(r, 1)))
	}
	mm.taus = mm.c.FromFloats(t, 1, L, 1, 1)
	return mm.taus
}
