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
	dtype engine.DType
	one   engine.Tensor // [1] float32
	ones  map[int]engine.Tensor
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
	norm := c.ops.Sqrt(c.sumLast(c.Mul(s, s)))
	limit := c.FromFloats([]float32{threshold}, 1)
	factor := c.ops.Minimum(c.ops.Div(limit, c.ScalarAdd(norm, 1e-6)), c.one)
	return c.Mul(s, factor)
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
	for i := len(blocks) - 1; i >= 0; i-- {
		stacked = c.Concat(c.ExpandDims(blocks[i], 2), stacked, 2)
	}
	dims := current.Shape()[2]
	scores := c.ScalarMul(c.MatMul(stacked, queryT), float32(1/math.Sqrt(float64(dims))))
	weights := c.Softmax(scores, 2)
	return c.MatMul(c.Transpose(weights, 0, 1, 3, 2), stacked)
}

// shortConv is the static short convolution: a causal depthwise filter over
// the last kernel inputs, added to the input. x is [1, L, C] in the model
// dtype, w [C, K, 1], tail the last K-1 inputs of the steps before (nil at
// the start). It returns the output and the new tail.
func (c *ctx) shortConv(x, w, tail engine.Tensor) (engine.Tensor, engine.Tensor) {
	k := w.Shape()[1]
	window := c.convWindow(x, tail, k)
	out := c.f32(c.Conv1dDepthwise(c.model(window), w))
	out = c.ScalarMul(c.Add(out, c.f32(x)), float32(1/math.Sqrt(float64(k+1))))
	return c.model(out), c.tailOf(window, k)
}

func (c *ctx) convWindow(x, tail engine.Tensor, k int) engine.Tensor {
	if tail == nil {
		ch := x.Shape()[2]
		tail = c.FromFloats(make([]float32, (k-1)*ch), 1, k-1, ch)
	}
	return c.Concat(tail, c.f32(x), 1)
}

func (c *ctx) tailOf(window engine.Tensor, k int) engine.Tensor {
	n := window.Shape()[1]
	return c.Slice(window, 1, n-(k-1), n)
}
