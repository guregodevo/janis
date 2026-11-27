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
	Q, K, V, R, O, Gate, Sink projection
	QScale, RelScale          engine.Tensor // float32 [heads, head dim], [heads, d_rel]
	RelProj                   engine.Tensor // [d_rel, RelExtent]
	// KVConv is KConv and VConv as one filter over the keys' channels then
	// the values', and kv the K and V projections as one weight for when
	// their inputs are the same tensor: one convolution a visit for the
	// two, on a backend where the node is the cost (engine.RelAttention).
	KVConv engine.Tensor
	kv     *joined
	// qIn holds the projections that read the query input (Q, R, Gate and
	// Sink) as one weight, their rows one after the other, on a backend
	// where a matmul's cost is the node more than its rows
	// (engine.RelAttention). Nil when they are kept apart.
	qIn *joined
	// RelProjFar is RelProj with one more column of zeros, the bias of the
	// distances past RelExtent, for a backend that gathers the bias itself
	// (engine.RelAttention).
	RelProjFar engine.Tensor

	// Sliding layers.
	KConv, VConv engine.Tensor // [channels, kernel, 1]
	// Full layers: the static part of the filter, float32
	// [kv heads, head dim, kernel], and the projection of the dynamic part.
	KStatic, VStatic engine.Tensor
	KDyn, VDyn       projection

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
	if c.rel != nil {
		// The clips' scales, one per output of the projection before them,
		// go into that projection's rows: no multiply a token.
		if q, ok := scaledRows(c, a.Q, a.QScale); ok {
			a.Q, a.QScale = q, nil
		}
		if r, ok := scaledRows(c, a.R, a.RelScale); ok {
			a.R, a.RelScale = r, nil
		}
	}
	if c.rel != nil && ld.err == nil {
		if j := join(c, true, a.Q, a.R, a.Gate, a.Sink); j != nil {
			a.qIn = j
			a.Q, a.R, a.Gate, a.Sink = j.part(0), j.part(1), j.part(2), j.part(3)
		}
	}
	if c.rel != nil && a.RelProj != nil {
		rows := a.RelProj.Shape()[0]
		a.RelProjFar = c.Concat(c.f32(a.RelProj), c.FromFloats(make([]float32, rows), rows, 1), 1)
	}
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
	if c.rel != nil && ld.err == nil && sliding {
		a.KVConv = c.Concat(c.f32(a.KConv), c.f32(a.VConv), 0)
		a.kv = join(c, false, a.K, a.V)
	}
	return a, ld.err
}

// joined is several quantized projections of one input as one: the rows of
// their weights one after the other, each projection a cut of the output.
// The product is computed once per input and kept for the cuts that follow.
type joined struct {
	all  *qwen.QuantLinear
	ends []int // each part's last row + 1

	in, out engine.Tensor // the last input and its product
}

// join joins ps when each is a quantized projection with no bias, of the
// same input width, bits and group size, and the backend reads tensors
// back raw; nil otherwise. With free the parts' own tensors are freed: the
// parts are then used only as cuts of the whole.
func join(c *ctx, free bool, ps ...projection) *joined {
	rr, ok := c.Backend.(engine.RawReader)
	if !ok {
		return nil
	}
	var ls []*qwen.QuantLinear
	for _, p := range ps {
		l, ok := p.(*qwen.QuantLinear)
		if !ok || l.Bias != nil || len(l.Weight.Shape()) != 2 {
			return nil
		}
		if len(ls) > 0 {
			f := ls[0]
			if l.Bits != f.Bits || l.GroupSize != f.GroupSize || l.Weight.Shape()[1] != f.Weight.Shape()[1] ||
				l.Scales.Shape()[1] != f.Scales.Shape()[1] || rr.DTypeOf(l.Weight) != rr.DTypeOf(f.Weight) {
				return nil
			}
		}
		ls = append(ls, l)
	}
	j := &joined{}
	var words []byte
	var scales, biases []float32
	rows := 0
	for _, l := range ls {
		words = append(words, rr.Bytes(l.Weight)...)
		scales = append(scales, c.Floats(l.Scales)...)
		biases = append(biases, c.Floats(l.Biases)...)
		rows += l.Weight.Shape()[0]
		j.ends = append(j.ends, rows)
	}
	f := ls[0]
	all := *f
	all.Weight = c.FromRaw(rr.DTypeOf(f.Weight), words, rows, f.Weight.Shape()[1])
	all.Scales = c.FromFloats(scales, rows, f.Scales.Shape()[1])
	all.Biases = c.FromFloats(biases, rows, f.Scales.Shape()[1])
	j.all = &all
	if fr, ok := c.Backend.(engine.Freer); ok && free {
		for _, l := range ls {
			fr.Free(l.Weight, l.Scales, l.Biases)
		}
	}
	return j
}

// part is the i-th projection joined.
func (j *joined) part(i int) projection { return joinedPart{j, i} }

type joinedPart struct {
	j *joined
	i int
}

func (p joinedPart) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	j := p.j
	if j.in != x {
		j.in, j.out = x, j.all.Forward(b, x)
	}
	from := 0
	if p.i > 0 {
		from = j.ends[p.i-1]
	}
	return b.Slice(j.out, len(x.Shape())-1, from, j.ends[p.i])
}

// scaledRows is quantized projection p with its output row i multiplied by
// scale's i-th value: an affine weight is q*scale + bias a group, so the
// row's scales and biases take the factor. False when p is not quantized.
func scaledRows(c *ctx, p projection, scale engine.Tensor) (projection, bool) {
	l, ok := p.(*qwen.QuantLinear)
	if !ok || scale == nil || l.Bias != nil {
		return nil, false
	}
	f := c.Floats(scale)
	sShape := l.Scales.Shape()
	if len(sShape) != 2 || sShape[0] != len(f) {
		return nil, false
	}
	sc, bi := c.Floats(l.Scales), c.Floats(l.Biases)
	groups := sShape[1]
	for r := range f {
		for g := 0; g < groups; g++ {
			sc[r*groups+g] *= f[r]
			bi[r*groups+g] *= f[r]
		}
	}
	out := *l
	out.Scales, out.Biases = c.FromFloats(sc, sShape...), c.FromFloats(bi, sShape...)
	return &out, true
}

// dynamicConv is the full layers' short convolution: the filter of each
// position is a static weight plus one its own input chooses, per head.
// x is [1, L, C], in the layer input the projection x was made from.
func (a *Attention) dynamicConv(c *ctx, x, in, static engine.Tensor, dyn projection, cache *VisitCache, slot int) engine.Tensor {
	L, K, H, D := x.Shape()[1], a.Kernel, a.KVHeads, a.HeadDim
	chosen := c.f32(dyn.Forward(c, in)) // [1, L, H*K]
	if window := c.tapWindow(x, cache.Conv[slot]); window != nil {
		if out := c.rel.ConvTapsDyn(window, static, chosen); out != nil {
			cache.Conv[slot] = window
			return c.model(out)
		}
	}
	weights := c.Reshape(chosen, 1, L, H, 1, K)
	weights = c.Add(weights, static) // [1, L, H, D, K]
	window := c.convWindow(x, cache.tail(c, slot), K)
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
	cache.slide(c, slot, window, K)
	return c.model(out)
}

// Forward runs attention for L new tokens at positions pos.. on the query,
// key and value inputs [1, L, hidden] (the same tensor on a sliding layer
// outside the loop), reading and extending the visit's cache.
func (a *Attention) Forward(c *ctx, qIn, kIn, vIn engine.Tensor, cache *VisitCache, pos int, masks *maskMaker, eps float32) engine.Tensor {
	L, H, G, D := qIn.Shape()[1], a.Heads, a.KVHeads, a.HeadDim
	var keys, values engine.Tensor
	switch {
	case a.KVConv != nil:
		// The keys' and the values' convolutions as one, over both sets of
		// channels: slot 0 holds the tail of the two, slot 1 nothing.
		var kv engine.Tensor
		if a.kv != nil && kIn == vIn {
			kv = a.kv.all.Forward(c, kIn)
		} else {
			kv = c.Concat(c.f32(a.K.Forward(c, kIn)), c.f32(a.V.Forward(c, vIn)), 2)
		}
		kv = c.shortConv(kv, a.KVConv, cache, 0)
		ch := G * D
		keys, values = c.Slice(kv, 2, 0, ch), c.Slice(kv, 2, ch, 2*ch)
	case a.Window > 0:
		keys = c.shortConv(a.K.Forward(c, kIn), a.KConv, cache, 0)
		values = c.shortConv(a.V.Forward(c, vIn), a.VConv, cache, 1)
	default:
		keys = a.dynamicConv(c, a.K.Forward(c, kIn), kIn, a.KStatic, a.KDyn, cache, 0)
		values = a.dynamicConv(c, a.V.Forward(c, vIn), vIn, a.VStatic, a.VDyn, cache, 1)
	}
	gate := a.Gate.Forward(c, qIn)
	sinks := a.Sink.Forward(c, qIn) // [1, L, H]

	qRaw := c.Reshape(a.Q.Forward(c, qIn), 1, L, H, D)
	relRaw := c.Reshape(a.R.Forward(c, qIn), 1, L, H, masks.cfg.DRel)
	if a.qIn != nil {
		a.qIn.in, a.qIn.out = nil, nil // the four cuts are taken
	}
	k := c.model(c.l2clip(c.Reshape(keys, 1, L, G, D), nil, masks.cfg.QKClip))
	v := c.model(c.rms(c.f32(c.Reshape(values, 1, L, G, D)), eps))
	k, v = cache.update(c, c.Transpose(k, 0, 2, 1, 3), c.Transpose(v, 0, 2, 1, 3), max(a.Window-1, 0))
	S := k.Shape()[2]
	scale := float32(1 / math.Sqrt(float64(D)))

	// A backend that has the attention as an operation: it reads the caches
	// in place, gathers the bias itself and takes the two clips' thresholds
	// with its scales.
	if c.rel != nil && a.RelProjFar != nil {
		q, rel := c.l2unit(qRaw, a.QScale, masks.cfg.QKClip), c.l2unit(relRaw, a.RelScale, masks.cfg.RelClip)
		if a.Window == 0 {
			tau := masks.tau(pos, L)
			q, rel = c.Mul(q, tau), c.Mul(rel, tau)
		}
		q = c.Transpose(q, 0, 2, 1, 3)
		if out := c.rel.RelAttend(q, k, v, rel, a.RelProjFar, c.f32(sinks), pos, a.Window, scale*masks.cfg.QKClip, masks.cfg.RelClip); out != nil {
			out = c.Reshape(c.Transpose(out, 0, 2, 1, 3), 1, L, H*D)
			out = c.Mul(out, c.Sigmoid(c.f32(gate)))
			return a.O.Forward(c, c.model(out))
		}
	}
	q := c.model(c.l2clip(qRaw, a.QScale, masks.cfg.QKClip))
	rel := c.model(c.l2clip(relRaw, a.RelScale, masks.cfg.RelClip))
	if a.Window == 0 {
		tau := masks.tau(pos, L)
		q = c.model(c.Mul(c.f32(q), tau))
		rel = c.model(c.Mul(c.f32(rel), tau))
	}
	q = c.Transpose(q, 0, 2, 1, 3)
	bias := c.MatMul(rel, a.RelProj) // [1, L, H, RelExtent]

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
	at := func(pos int) []float32 {
		t := make([]float32, L)
		for i := range t {
			r := float64(pos+i+1) / float64(mm.cfg.NFloor)
			t[i] = float32(1 + float64(mm.cfg.Alpha)*math.Log(math.Max(r, 1)))
		}
		return t
	}
	if mm.c.rel != nil && L == 1 {
		mm.taus = mm.c.rel.StepFloats(pos, at, 1, L, 1, 1)
	} else {
		mm.taus = mm.c.FromFloats(at(pos), 1, L, 1, 1)
	}
	return mm.taus
}
