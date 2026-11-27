package inkling2

import (
	"math"

	"github.com/guregodevo/janis/engine"
)

// ctx is what every part of a forward pass needs: the backend, its extra
// operations, the dtype the residual stream is kept in and a few constants.
type ctx struct {
	engine.Backend
	ops   engine.LoopOps
	rel   engine.RelAttention // nil on a backend without it
	dtype engine.DType
	one   engine.Tensor // [1] float32
	ones  map[int]engine.Tensor

	experts *expertCache // nil when the experts are resident

	// stacks holds, for one forward pass, the block outputs already put
	// side by side for a depth attention: each mix adds its own state to a
	// stack the mixes before it built.
	stacks map[stackKey]engine.Tensor
}

// stackKey names a run of block outputs by its ends and its length.
type stackKey struct {
	first, last engine.Tensor
	n           int
}

// stack is blocks side by side, [1, L, len(blocks), D], oldest first.
func (c *ctx) stack(blocks []engine.Tensor) engine.Tensor {
	n := len(blocks)
	key := stackKey{blocks[0], blocks[n-1], n}
	if s, ok := c.stacks[key]; ok {
		return s
	}
	s := c.ExpandDims(blocks[n-1], 2)
	if n > 1 {
		s = c.Concat(c.stack(blocks[:n-1]), s, 2)
	}
	if c.stacks == nil {
		c.stacks = map[stackKey]engine.Tensor{}
	}
	c.stacks[key] = s
	return s
}

func (c *ctx) f32(x engine.Tensor) engine.Tensor   { return c.Cast(x, engine.F32) }
func (c *ctx) model(x engine.Tensor) engine.Tensor { return c.Cast(x, c.dtype) }

// sumLast sums over the last axis, which stays with size 1.
func (c *ctx) sumLast(x engine.Tensor) engine.Tensor {
	last := len(x.Shape()) - 1
	return c.ExpandDims(c.ScalarMul(c.Mean(x, last), float32(x.Shape()[last])), last)
}

// rms is RMS normalization with no weight, in float32.
func (c *ctx) rms(x engine.Tensor, eps float32) engine.Tensor {
	if c.rel != nil {
		return c.rel.RMS(x, eps)
	}
	s := x.Shape()
	return c.RMSNorm(x, c.ones[s[len(s)-1]], eps)
}

// defaultEps is the reference's epsilon where it does not pass the config's.
const defaultEps = 1e-6

// l2clip scales x (by scale when set) and shrinks each row whose L2 norm
// over the last axis is above threshold down to it. Float32 in and out.
func (c *ctx) l2clip(x, scale engine.Tensor, threshold float32) engine.Tensor {
	s := c.f32(x)
	if scale != nil {
		s = c.Mul(s, scale)
	}
	if c.rel != nil {
		return c.rel.L2Clip(s, threshold)
	}
	norm := c.ops.Sqrt(c.sumLast(c.Mul(s, s)))
	limit := c.FromFloats([]float32{threshold}, 1)
	factor := c.ops.Minimum(c.ops.Div(limit, c.ScalarAdd(norm, 1e-6)), c.one)
	return c.Mul(s, factor)
}

// l2unit is l2clip divided by threshold, on a backend whose attention takes
// the threshold with its own scale.
func (c *ctx) l2unit(x, scale engine.Tensor, threshold float32) engine.Tensor {
	s := c.f32(x)
	if scale != nil {
		s = c.Mul(s, scale)
	}
	return c.rel.L2Unit(s, threshold)
}

// pick takes row i of the queries axis of a depth attention output
// [1, L, Q, D] -> [1, L, D].
func (c *ctx) pick(x engine.Tensor, i int) engine.Tensor {
	s := x.Shape()
	return c.Reshape(c.Slice(x, 2, i, i+1), s[0], s[1], s[3])
}

// depthAttention mixes the block outputs so far with the current state,
// once per query: a softmax over the states of each position, scored by
// the query. states are [1, L, D] float32, queryT is [D, Q]; the result is
// [1, L, Q, D].
func (c *ctx) depthAttention(blocks []engine.Tensor, current, queryT engine.Tensor) engine.Tensor {
	stacked := c.ExpandDims(current, 2)
	if len(blocks) > 0 {
		stacked = c.Concat(c.stack(blocks), stacked, 2)
	}
	if c.rel != nil {
		if out := c.rel.DepthMix(stacked, queryT); out != nil {
			return out
		}
	}
	dims := current.Shape()[2]
	scores := c.ScalarMul(c.MatMul(stacked, queryT), float32(1/math.Sqrt(float64(dims))))
	weights := c.Softmax(scores, 2)
	return c.MatMul(c.Transpose(weights, 0, 1, 3, 2), stacked)
}

// shortConv is the static short convolution: a causal depthwise filter over
// the last kernel inputs, added to the input. x is [1, L, C] in the model
// dtype, w [C, K, 1]; the last K-1 inputs of the steps before are slot's
// tail in cache, which is slid on to this forward's.
func (c *ctx) shortConv(x, w engine.Tensor, cache *VisitCache, slot int) engine.Tensor {
	k := w.Shape()[1]
	if window := c.tapWindow(x, cache.Conv[slot]); window != nil {
		if out := c.rel.ConvTaps(window, w); out != nil {
			cache.Conv[slot] = window
			return c.model(out)
		}
	}
	window := c.convWindow(x, cache.tail(c, slot), k)
	out := c.f32(c.Conv1dDepthwise(c.model(window), w))
	out = c.ScalarMul(c.Add(out, c.f32(x)), float32(1/math.Sqrt(float64(k+1))))
	cache.slide(c, slot, window, k)
	return c.model(out)
}

// tapWindow is one token's convolution input per channel, [1, C, K] with
// the newest last, on a backend that keeps the last window so
// (engine.RelAttention): the token is shifted into it. Nil for more than
// one token, at the first one, and on any other backend.
func (c *ctx) tapWindow(x, last engine.Tensor) engine.Tensor {
	if c.rel == nil || x.Shape()[1] != 1 || last == nil {
		return nil
	}
	return c.rel.ShiftIn(last, c.f32(x))
}

func (c *ctx) convWindow(x, tail engine.Tensor, k int) engine.Tensor {
	if tail == nil {
		ch := x.Shape()[2]
		tail = c.FromFloats(make([]float32, (k-1)*ch), 1, k-1, ch)
	}
	return c.Concat(tail, c.f32(x), 1)
}
