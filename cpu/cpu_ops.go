package cpu

import (
	"math"

	"memdoor/llm/engine"
)

// MatMul is batched matrix multiply with NumPy broadcasting on the batch dims.
// The last two axes are the matrix dims (…,M,K) x (…,K,N) -> (…,M,N). 1-D and
// 2-D inputs are handled as the common no-batch case.
func (b *Backend) MatMul(a, bb engine.Tensor) engine.Tensor {
	ta, tb := as(a), as(bb)
	// Promote to at least 2-D.
	ash := ensure2D(ta.shape)
	bsh := ensure2D(tb.shape)
	M, K := ash[len(ash)-2], ash[len(ash)-1]
	K2, N := bsh[len(bsh)-2], bsh[len(bsh)-1]
	if K != K2 {
		panic("cpu.MatMul: inner dims disagree")
	}
	batchA, batchB := ash[:len(ash)-2], bsh[:len(bsh)-2]
	batch := broadcastShape(batchA, batchB)
	nBatch := numel(batch)

	outShape := append(append([]int(nil), batch...), M, N)
	out := newTensor(engine.F32, outShape...)
	aMat, bMat := M*K, K*N
	for bi := 0; bi < nBatch; bi++ {
		aOff := bcBatch(bi, batch, batchA) * aMat
		bOff := bcBatch(bi, batch, batchB) * bMat
		oOff := bi * M * N
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				var s float32
				for k := 0; k < K; k++ {
					s += ta.data[aOff+i*K+k] * tb.data[bOff+k*N+j]
				}
				out.data[oOff+i*N+j] = s
			}
		}
	}
	// A 1-D operand collapses the corresponding output matrix axis, but the
	// model code always feeds ≥2-D here, so keep the explicit (…,M,N) shape.
	return out
}

func ensure2D(shape []int) []int {
	if len(shape) >= 2 {
		return shape
	}
	if len(shape) == 1 {
		return []int{1, shape[0]}
	}
	return []int{1, 1}
}

// bcBatch maps a flat batch index over `batch` to the operand's flat batch
// index, honoring broadcast (size-1) dims.
func bcBatch(flat int, batch, op []int) int {
	if len(op) == 0 {
		return 0
	}
	bs := strides(batch)
	os := strides(op)
	off := len(batch) - len(op)
	idx := 0
	for i := 0; i < len(batch); i++ {
		coord := (flat / bs[i]) % batch[i]
		j := i - off
		if j < 0 {
			continue
		}
		if op[j] == 1 {
			continue
		}
		idx += coord * os[j]
	}
	return idx
}

// Transpose permutes axes by the given order.
func (b *Backend) Transpose(x engine.Tensor, axes ...int) engine.Tensor {
	tx := as(x)
	ns := make([]int, len(axes))
	for i, a := range axes {
		ns[i] = tx.shape[a]
	}
	out := newTensor(engine.F32, ns...)
	in := strides(tx.shape)
	on := strides(ns)
	for o := range out.data {
		// decompose o in output coords, map back to input flat index
		rem, idx := o, 0
		for d := 0; d < len(ns); d++ {
			coord := (rem / on[d]) % ns[d]
			idx += coord * in[axes[d]]
		}
		out.data[o] = tx.data[idx]
	}
	return out
}

// Mean averages over an axis, removing it.
func (b *Backend) Mean(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape)
	}
	ns := make([]int, 0, len(tx.shape)-1)
	ns = append(ns, tx.shape[:axis]...)
	ns = append(ns, tx.shape[axis+1:]...)
	out := newTensor(engine.F32, ns...)
	st := strides(tx.shape)
	axisLen, axisStride := tx.shape[axis], st[axis]
	for o := range out.data {
		// map output coord back to the base offset of the reduced line
		rem, base, od := o, 0, 0
		for d := 0; d < len(tx.shape); d++ {
			if d == axis {
				continue
			}
			coord := (rem / strides(ns)[od]) % tx.shape[d]
			base += coord * st[d]
			od++
		}
		var s float32
		for i := 0; i < axisLen; i++ {
			s += tx.data[base+i*axisStride]
		}
		out.data[o] = s / float32(axisLen)
	}
	return out
}

// RMSNorm: x / sqrt(mean(x^2)+eps) * weight, over the last dim.
func (b *Backend) RMSNorm(x, weight engine.Tensor, eps float32) engine.Tensor {
	tx, tw := as(x), as(weight)
	d := tx.shape[len(tx.shape)-1]
	out := newTensor(engine.F32, tx.shape...)
	rows := len(tx.data) / d
	for r := 0; r < rows; r++ {
		var ss float32
		for i := 0; i < d; i++ {
			v := tx.data[r*d+i]
			ss += v * v
		}
		inv := 1.0 / float32(math.Sqrt(float64(ss/float32(d)+eps)))
		for i := 0; i < d; i++ {
			out.data[r*d+i] = tx.data[r*d+i] * inv * tw.data[i]
		}
	}
	return out
}

// LayerNorm over the last dim with weight + bias.
func (b *Backend) LayerNorm(x, weight, bias engine.Tensor, eps float32) engine.Tensor {
	tx, tw, tb := as(x), as(weight), as(bias)
	d := tx.shape[len(tx.shape)-1]
	out := newTensor(engine.F32, tx.shape...)
	rows := len(tx.data) / d
	for r := 0; r < rows; r++ {
		var mean float32
		for i := 0; i < d; i++ {
			mean += tx.data[r*d+i]
		}
		mean /= float32(d)
		var v float32
		for i := 0; i < d; i++ {
			diff := tx.data[r*d+i] - mean
			v += diff * diff
		}
		inv := 1.0 / float32(math.Sqrt(float64(v/float32(d)+eps)))
		for i := 0; i < d; i++ {
			out.data[r*d+i] = (tx.data[r*d+i]-mean)*inv*tw.data[i] + tb.data[i]
		}
	}
	return out
}

// Concat joins a and b along axis.
func (b *Backend) Concat(a, bb engine.Tensor, axis int) engine.Tensor {
	ta, tb := as(a), as(bb)
	if axis < 0 {
		axis += len(ta.shape)
	}
	ns := append([]int(nil), ta.shape...)
	ns[axis] += tb.shape[axis]
	out := newTensor(engine.F32, ns...)
	// outer = product of dims before axis; copy contiguous blocks.
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= ta.shape[i]
	}
	inner := 1
	for i := axis + 1; i < len(ta.shape); i++ {
		inner *= ta.shape[i]
	}
	blkA, blkB := ta.shape[axis]*inner, tb.shape[axis]*inner
	for o := 0; o < outer; o++ {
		copy(out.data[o*(blkA+blkB):], ta.data[o*blkA:o*blkA+blkA])
		copy(out.data[o*(blkA+blkB)+blkA:], tb.data[o*blkB:o*blkB+blkB])
	}
	return out
}

// Slice keeps [start:end] along axis, full extent elsewhere.
func (b *Backend) Slice(x engine.Tensor, axis, start, end int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape)
	}
	ns := append([]int(nil), tx.shape...)
	ns[axis] = end - start
	out := newTensor(engine.F32, ns...)
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= tx.shape[i]
	}
	inner := 1
	for i := axis + 1; i < len(tx.shape); i++ {
		inner *= tx.shape[i]
	}
	srcBlk := tx.shape[axis] * inner
	dstBlk := (end - start) * inner
	for o := 0; o < outer; o++ {
		copy(out.data[o*dstBlk:], tx.data[o*srcBlk+start*inner:o*srcBlk+end*inner])
	}
	return out
}
