// Package cpu is a pure-Go, CPU implementation of engine.Backend — the
// portable counterpart to the MLX (Apple-Silicon) backend, so the same model
// forward passes (llm/qwen, llm/bge) run on Linux and Intel Macs.
//
// It is EAGER, not graph-building: every op computes immediately and returns a
// materialized tensor. Eval is therefore a no-op and the Sweeper interface is
// intentionally not implemented — Go's GC manages memory (see engine.Backend's
// docs, which anticipate exactly this pure-Go backend). Tensors store data as
// row-major []float32 regardless of source dtype; weights load via FromRaw,
// which up-converts f16/bf16 to f32 on the way in.
//
// Correctness first, speed later: ops are straightforward triple-loops. The
// hot path (MatMul, SDPA) is the obvious place to drop in BLAS / SIMD once the
// numerics match the MLX backend token-for-token.
package cpu

import (
	"encoding/binary"
	"math"

	"memdoor/llm/engine"
)

// Backend is the pure-Go engine.Backend. Stateless — every op is a pure
// function of its inputs, so one Backend is safe to share.
type Backend struct{}

// New returns a CPU backend.
func New() *Backend { return &Backend{} }

func (b *Backend) Version() string { return "cpu-go-0.1" }

// tensor is the concrete handle: a shape + dense row-major f32 data. ints holds
// integer payloads (token ids / gather indices) when the tensor is I32/U32 —
// float data still mirrors it so float ops stay uniform.
type tensor struct {
	shape []int
	data  []float32
	dt    engine.DType
}

func (t *tensor) Shape() []int { return append([]int(nil), t.shape...) }

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func newTensor(dt engine.DType, shape ...int) *tensor {
	return &tensor{shape: append([]int(nil), shape...), data: make([]float32, numel(shape)), dt: dt}
}

func as(t engine.Tensor) *tensor { return t.(*tensor) }

// ---- creation ------------------------------------------------------------

func (b *Backend) FromFloats(data []float32, shape ...int) engine.Tensor {
	t := newTensor(engine.F32, shape...)
	copy(t.data, data)
	return t
}

func (b *Backend) FromInt32(data []int32, shape ...int) engine.Tensor {
	t := newTensor(engine.I32, shape...)
	for i, v := range data {
		t.data[i] = float32(v)
	}
	return t
}

// FromRaw decodes packed weight bytes of the given dtype into f32. Covers the
// dtypes a safetensors load produces for unquantized models; 4-bit quantized
// weights are a later phase (a separate dequant path before this).
func (b *Backend) FromRaw(dt engine.DType, raw []byte, shape ...int) engine.Tensor {
	t := newTensor(dt, shape...)
	n := numel(shape)
	switch dt {
	case engine.F32:
		for i := 0; i < n; i++ {
			t.data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case engine.F16:
		for i := 0; i < n; i++ {
			t.data[i] = f16tof32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case engine.BF16:
		for i := 0; i < n; i++ {
			// bf16 is the top 16 bits of an f32.
			t.data[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case engine.I32, engine.U32:
		for i := 0; i < n; i++ {
			t.data[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
	default:
		panic("cpu.FromRaw: unsupported dtype")
	}
	return t
}

// f16tof32 converts an IEEE-754 half to a float32.
func f16tof32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0x1F:
		return math.Float32frombits(sign | 0x7F800000 | mant<<13) // inf / nan
	case exp == 0:
		// subnormal: normalize
		e := uint32(127 - 15 + 1)
		for mant&0x0400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x03FF
		return math.Float32frombits(sign | e<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
	}
}

// ---- lazy-eval no-ops (this backend is eager) ----------------------------

func (b *Backend) Eval(ts ...engine.Tensor) {}

// ---- elementwise (with NumPy broadcasting) -------------------------------

// broadcastShape returns the broadcasted result shape of a and b, right-aligned.
func broadcastShape(a, b []int) []int {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if i := len(a) - n + i; i >= 0 {
			da = a[i]
		}
		if i := len(b) - n + i; i >= 0 {
			db = b[i]
		}
		if da == 1 {
			out[i] = db
		} else {
			out[i] = da
		}
	}
	return out
}

// strides returns row-major strides for a shape.
func strides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// bcIndex maps a flat output index (in outShape) to the flat input index of a
// right-aligned operand with shape opShape (size-1 dims broadcast).
func bcIndex(flat int, outShape, opShape []int) int {
	outS := strides(outShape)
	opS := strides(opShape)
	off := len(outShape) - len(opShape)
	idx := 0
	for i := 0; i < len(outShape); i++ {
		coord := (flat / outS[i]) % outShape[i]
		j := i - off
		if j < 0 {
			continue
		}
		if opShape[j] == 1 {
			continue // broadcast: stays at 0
		}
		idx += coord * opS[j]
	}
	return idx
}

func (b *Backend) elementwise(a, bb engine.Tensor, f func(x, y float32) float32) engine.Tensor {
	ta, tb := as(a), as(bb)
	outShape := broadcastShape(ta.shape, tb.shape)
	out := newTensor(engine.F32, outShape...)
	for i := range out.data {
		out.data[i] = f(ta.data[bcIndex(i, outShape, ta.shape)], tb.data[bcIndex(i, outShape, tb.shape)])
	}
	return out
}

func (b *Backend) Add(a, bb engine.Tensor) engine.Tensor {
	return b.elementwise(a, bb, func(x, y float32) float32 { return x + y })
}

func (b *Backend) Mul(a, bb engine.Tensor) engine.Tensor {
	return b.elementwise(a, bb, func(x, y float32) float32 { return x * y })
}

// ---- unary activations ---------------------------------------------------

func (b *Backend) unary(x engine.Tensor, f func(float32) float32) engine.Tensor {
	tx := as(x)
	out := newTensor(engine.F32, tx.shape...)
	for i, v := range tx.data {
		out.data[i] = f(v)
	}
	return out
}

func (b *Backend) Tanh(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(v float32) float32 { return float32(math.Tanh(float64(v))) })
}

func (b *Backend) Erf(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(v float32) float32 { return float32(math.Erf(float64(v))) })
}

func (b *Backend) SiLU(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(v float32) float32 { return v / (1 + float32(math.Exp(float64(-v)))) })
}

// Gelu uses the tanh approximation, matching engine.Backend's contract.
func (b *Backend) Gelu(x engine.Tensor) engine.Tensor {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return b.unary(x, func(v float32) float32 {
		inner := c * (v + 0.044715*v*v*v)
		return 0.5 * v * (1 + float32(math.Tanh(float64(inner))))
	})
}

// ---- shape ops -----------------------------------------------------------

func (b *Backend) Reshape(x engine.Tensor, shape ...int) engine.Tensor {
	tx := as(x)
	// resolve a single -1 dim
	prod, neg := 1, -1
	for i, d := range shape {
		if d == -1 {
			neg = i
		} else {
			prod *= d
		}
	}
	out := append([]int(nil), shape...)
	if neg >= 0 {
		out[neg] = len(tx.data) / prod
	}
	return &tensor{shape: out, data: tx.data, dt: tx.dt}
}

func (b *Backend) ExpandDims(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape) + 1
	}
	ns := make([]int, 0, len(tx.shape)+1)
	ns = append(ns, tx.shape[:axis]...)
	ns = append(ns, 1)
	ns = append(ns, tx.shape[axis:]...)
	return &tensor{shape: ns, data: tx.data, dt: tx.dt}
}

func (b *Backend) Softmax(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape)
	}
	out := newTensor(engine.F32, tx.shape...)
	copy(out.data, tx.data)
	st := strides(tx.shape)
	axisLen, axisStride := tx.shape[axis], st[axis]
	// iterate every "line" along axis
	outer := numel(tx.shape) / axisLen
	for o := 0; o < outer; o++ {
		// compute the base offset of this line: spread o across non-axis dims
		base, rem := 0, o
		for d := len(tx.shape) - 1; d >= 0; d-- {
			if d == axis {
				continue
			}
			base += (rem % tx.shape[d]) * st[d]
			rem /= tx.shape[d]
		}
		mx := float32(math.Inf(-1))
		for i := 0; i < axisLen; i++ {
			if v := out.data[base+i*axisStride]; v > mx {
				mx = v
			}
		}
		var sum float32
		for i := 0; i < axisLen; i++ {
			e := float32(math.Exp(float64(out.data[base+i*axisStride] - mx)))
			out.data[base+i*axisStride] = e
			sum += e
		}
		for i := 0; i < axisLen; i++ {
			out.data[base+i*axisStride] /= sum
		}
	}
	return out
}
