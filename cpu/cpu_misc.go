package cpu

import (
	"math"

	"memdoor/llm/engine"
)

// Floats forces evaluation (no-op here, already eager) and returns a copy.
func (b *Backend) Floats(t engine.Tensor) []float32 {
	return append([]float32(nil), as(t).data...)
}

// Ints copies a tensor out as int32 (small index tensors).
func (b *Backend) Ints(t engine.Tensor) []int32 {
	d := as(t).data
	out := make([]int32, len(d))
	for i, v := range d {
		out[i] = int32(math.Round(float64(v)))
	}
	return out
}

func (b *Backend) ScalarMul(x engine.Tensor, v float32) engine.Tensor {
	return b.unary(x, func(e float32) float32 { return e * v })
}

func (b *Backend) ScalarAdd(x engine.Tensor, v float32) engine.Tensor {
	return b.unary(x, func(e float32) float32 { return e + v })
}

// Cast changes the recorded dtype. Data is always held as f32, so float casts
// keep full precision (more precise than MLX's f16, within diff tolerance); an
// integer cast rounds.
func (b *Backend) Cast(x engine.Tensor, dt engine.DType) engine.Tensor {
	tx := as(x)
	out := &tensor{shape: append([]int(nil), tx.shape...), data: append([]float32(nil), tx.data...), dt: dt}
	if dt == engine.I32 || dt == engine.U32 {
		for i, v := range out.data {
			out.data[i] = float32(int32(math.Round(float64(v))))
		}
	}
	return out
}

// TakeAxis gathers slices of a along `axis` by integer indices. The common case
// is embedding lookup: a=[vocab,dim], indices=[L], axis=0 -> [L,dim].
func (b *Backend) TakeAxis(a, indices engine.Tensor, axis int) engine.Tensor {
	ta, ti := as(a), as(indices)
	if axis < 0 {
		axis += len(ta.shape)
	}
	n := len(ti.data)
	outShape := make([]int, 0, len(ta.shape)-1+len(ti.shape))
	outShape = append(outShape, ta.shape[:axis]...)
	outShape = append(outShape, ti.shape...)
	outShape = append(outShape, ta.shape[axis+1:]...)
	out := newTensor(ta.dt, outShape...)
	// A U32 (packed-quant) source carries exact words; gather those too so a
	// following Dequantize sees the real packed weights, not the lossy mirror.
	if ta.u32 != nil {
		out.u32 = make([]uint32, len(out.data))
	}

	st := strides(ta.shape)
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= ta.shape[i]
	}
	inner := st[axis] // size of one slice along axis
	axisLen := ta.shape[axis]
	dst := 0
	for o := 0; o < outer; o++ {
		for k := 0; k < n; k++ {
			idx := int(ti.data[k])
			if idx < 0 {
				idx += axisLen
			}
			src := o*axisLen*inner + idx*inner
			copy(out.data[dst:dst+inner], ta.data[src:src+inner])
			if out.u32 != nil {
				copy(out.u32[dst:dst+inner], ta.u32[src:src+inner])
			}
			dst += inner
		}
	}
	return out
}

// Argmax returns the index of the max along axis (axis removed).
func (b *Backend) Argmax(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape)
	}
	ns := make([]int, 0, len(tx.shape)-1)
	ns = append(ns, tx.shape[:axis]...)
	ns = append(ns, tx.shape[axis+1:]...)
	out := newTensor(engine.I32, ns...)
	st := strides(tx.shape)
	axisLen, axisStride := tx.shape[axis], st[axis]
	for o := range out.data {
		rem, base, od := o, 0, 0
		for d := 0; d < len(tx.shape); d++ {
			if d == axis {
				continue
			}
			coord := (rem / strides(ns)[od]) % tx.shape[d]
			base += coord * st[d]
			od++
		}
		best, bi := float32(math.Inf(-1)), 0
		for i := 0; i < axisLen; i++ {
			if v := tx.data[base+i*axisStride]; v > best {
				best, bi = v, i
			}
		}
		out.data[o] = float32(bi)
	}
	return out
}

func (b *Backend) Close() {}

// ---- 4-bit affine quantization (MLX-compatible) ---------------------------
// MLX packs valsPerWord = 32/bits sub-values into each uint32, LSB first, along
// the last axis; per group of `groupSize` sub-values there is one scale + bias,
// and the float value is `scale*q + bias` (affine). Dequantize/QuantMatmul
// reproduce that exactly so a Qwen3-8B-4bit MLX checkpoint runs unchanged on the
// CPU backend — no separate f16 model needed.

// Dequantize reconstructs the full float weight from (packed words, scales,
// biases). w is [..., P] (U32); the output is [..., P*32/bits].
func (b *Backend) Dequantize(w, scales, biases engine.Tensor, groupSize, bits int) engine.Tensor {
	tw, ts, tb := as(w), as(scales), as(biases)
	if tw.u32 == nil {
		panic("cpu.Dequantize: weight tensor is not packed U32")
	}
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	G := ts.shape[len(ts.shape)-1] // groups per row (= in/groupSize)
	rows := numel(tw.shape) / P

	outShape := append([]int(nil), tw.shape...)
	outShape[len(outShape)-1] = in
	out := newTensor(engine.F32, outShape...)

	for r := 0; r < rows; r++ {
		wRow, sRow, oRow := r*P, r*G, r*in
		for j := 0; j < in; j++ {
			q := (tw.u32[wRow+j/valsPerWord] >> uint((j%valsPerWord)*bits)) & mask
			g := j / groupSize
			out.data[oRow+j] = float32(q)*ts.data[sRow+g] + tb.data[sRow+g]
		}
	}
	return out
}

// QuantMatmul computes x @ w with w affine-quantized. transpose=true (the
// QuantizedLinear convention) treats the dequantized w as [out, in] and returns
// x @ Wᵀ -> [..., out]. Correctness-first: dequantize then dense matmul.
func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	W := b.Dequantize(w, scales, biases, groupSize, bits)
	if transpose {
		W = b.Transpose(W, lastTwoSwapped(as(W).shape)...)
	}
	return b.MatMul(x, W)
}

// lastTwoSwapped returns the identity axis order with the final two axes
// swapped — Wᵀ on the matrix dims, leaving any batch dims in place.
func lastTwoSwapped(shape []int) []int {
	n := len(shape)
	axes := make([]int, n)
	for i := range axes {
		axes[i] = i
	}
	axes[n-2], axes[n-1] = n-1, n-2
	return axes
}
