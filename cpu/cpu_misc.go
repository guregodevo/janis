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
	out := newTensor(engine.F32, outShape...)

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

// ---- Phase 3 (4-bit quantization) — not yet implemented -------------------
// Linux ships f16 models first; these land with the dequant path.

func (b *Backend) Dequantize(w, scales, biases engine.Tensor, groupSize, bits int) engine.Tensor {
	panic("cpu.Dequantize: 4-bit quantization is phase 3 (use an f16 model for now)")
}

func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	panic("cpu.QuantMatmul: 4-bit quantization is phase 3 (use an f16 model for now)")
}
