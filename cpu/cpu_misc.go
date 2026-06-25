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

	parWork(rows, in, func(rs, re int) {
		for r := rs; r < re; r++ {
			wRow, sRow, oRow := r*P, r*G, r*in
			for j := 0; j < in; j++ {
				q := (tw.u32[wRow+j/valsPerWord] >> uint((j%valsPerWord)*bits)) & mask
				g := j / groupSize
				out.data[oRow+j] = float32(q)*ts.data[sRow+g] + tb.data[sRow+g]
			}
		}
	})
	return out
}

// QuantMatmul computes x @ w with w affine-quantized. transpose=true (the
// QuantizedLinear convention) treats w as [out, in] and returns x @ Wᵀ ->
// [..., out]. Fused: parallel over output channels, each goroutine dequantizes
// one weight row into a reused buffer and dots it against every x row — so the
// unpack is amortized across the batch and NOTHING materializes the full
// [out,in] weight or a transpose (the old dequantize-then-dense-matmul path did
// both, every token, which dominated decode latency).
// GatherQuantMatmul is a correct (unoptimized) fallback for mlx's gather_qmm,
// matching the convention MoEBlock.Forward uses: x is [..lead.., 1, 1, in]
// (mlx-lm's double expand_dims, so one input vector per lead position),
// rhsIndices is [..lead.., K] selecting K experts out of the stacked
// w/scales/biases ([E, out, *]); output is [..lead.., K, 1, out]. Output slots
// are grouped by expert so each expert's weight is dequantized once (inside one
// QuantMatmul) and dotted against every token routed to it — amortizing the
// dequant across the batch without materializing a dense expert.
func (b *Backend) GatherQuantMatmul(x, w, scales, biases, rhsIndices engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	xs, ws, ss, bs := x.Shape(), w.Shape(), scales.Shape(), biases.Shape()
	rs := rhsIndices.Shape()
	in := xs[len(xs)-1]
	outDim := ws[1]
	E := ws[0]
	K := rs[len(rs)-1]
	idx := b.Ints(rhsIndices)
	nLead := 1
	for _, d := range rs[:len(rs)-1] {
		nLead *= d
	}
	// x is [..lead.., A, 1, in]: A==1 broadcasts one input vector across all K
	// experts (gate/up projections), A==K aligns a distinct vector per expert
	// slot (down projection, where x is the per-slot hidden state).
	A := xs[len(xs)-3]
	xt := as(x)
	out := newTensor(engine.F32, append(append([]int{}, rs...), 1, outDim)...)

	// Bucket the (lead, slot) output positions by the expert they route to.
	type slot struct{ src, dst int } // offsets into xt.data and out.data
	byExpert := make([][]slot, E)
	for l := 0; l < nLead; l++ {
		for k := 0; k < K; k++ {
			a := 0
			if A != 1 {
				a = k
			}
			e := int(idx[l*K+k])
			byExpert[e] = append(byExpert[e], slot{src: (l*A + a) * in, dst: (l*K + k) * outDim})
		}
	}
	for e, slots := range byExpert {
		if len(slots) == 0 {
			continue
		}
		// Slice this expert out of the stack once (relies on Slice/Reshape carrying
		// the packed u32 buffer), gather its tokens into one [R, in] batch, and run
		// a single QuantMatmul.
		we := b.Reshape(b.Slice(w, 0, e, e+1), ws[1], ws[2])
		se := b.Reshape(b.Slice(scales, 0, e, e+1), ss[1], ss[2])
		be := b.Reshape(b.Slice(biases, 0, e, e+1), bs[1], bs[2])
		xb := make([]float32, len(slots)*in)
		for i, s := range slots {
			copy(xb[i*in:], xt.data[s.src:s.src+in])
		}
		r := as(b.QuantMatmul(b.FromFloats(xb, len(slots), in), we, se, be, transpose, groupSize, bits))
		for i, s := range slots {
			copy(out.data[s.dst:s.dst+outDim], r.data[i*outDim:(i+1)*outDim])
		}
	}
	return out
}

func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	if !transpose {
		// The models never hit this; keep a correct fallback.
		return b.MatMul(x, b.Dequantize(w, scales, biases, groupSize, bits))
	}
	tx, tw, ts, tb := as(x), as(w), as(scales), as(biases)
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	outDim := tw.shape[len(tw.shape)-2] // W is [outDim, P] = [outDim, in]
	G := ts.shape[len(ts.shape)-1]
	rows := len(tx.data) / in // flattened leading dims of x

	outShape := append(append([]int(nil), tx.shape[:len(tx.shape)-1]...), outDim)
	y := newTensor(engine.F32, outShape...)

	// Decode fast path (rows == 1): the dominant cost at generation time. Walk
	// the packed words sequentially (no per-element i/valsPerWord div+mod), unpack
	// a whole word at a time, and fuse the dequant into the dot WITHOUT a wbuf
	// round-trip. scale/bias are constant per group, so within a group accumulate
	// sum(x·q) and sum(x) once and combine — turning the per-element
	// (q*scale+bias)*x into one scale-mul + one bias-mul per group. Requires
	// groupSize to be a multiple of valsPerWord (always true: groupSize∈{32,64,128},
	// valsPerWord∈{8,4,2}); otherwise fall through to the general path below.
	if rows == 1 && groupSize%valsPerWord == 0 {
		x0 := tx.data
		wordsPerGroup := groupSize / valsPerWord
		// 4-bit is the only width the models use; specialize it. Each u32 holds 8
		// nibbles — unroll them with constant shifts (no per-element shift/loop
		// counter) and use TWO accumulator lanes so the float adds aren't a single
		// dependent chain (better instruction-level parallelism). General path
		// below handles any other bit width.
		if bits == 4 && quant4SIMD {
			// SIMD per-channel kernel (NEON on arm64). One call per output
			// channel; x is shared. Falls back to the blocked-Go path below on
			// arches without a kernel (incl. linux/amd64).
			parWork(outDim, in, func(os, oe int) {
				for o := os; o < oe; o++ {
					v, _ := qChannel4Accel(&tw.u32[o*P], &x0[0], &ts.data[o*G], &tb.data[o*G], G, wordsPerGroup)
					y.data[o] = v
				}
			})
			return y
		}
		if bits == 4 {
			parWork(outDim, in, func(os, oe int) {
				o := os
				// 4-channel register blocking: process four output channels per
				// pass. x is loaded ONCE and reused across all four, and sumX (the
				// per-group Σx) is identical for every channel so it's computed
				// once and shared — only the four sumXQ accumulators differ. Four
				// independent chains also give the CPU plenty of ILP. The dominant
				// decode cost is this loop, so the x-reuse + shared-sumX is the win.
				for ; o+4 <= oe; o += 4 {
					wp0, wp1, wp2, wp3 := o*P, (o+1)*P, (o+2)*P, (o+3)*P
					sB0, sB1, sB2, sB3 := o*G, (o+1)*G, (o+2)*G, (o+3)*G
					var acc0, acc1, acc2, acc3 float32
					xi := 0
					for g := 0; g < G; g++ {
						var q0, q1, q2, q3, sx float32
						for wc := 0; wc < wordsPerGroup; wc++ {
							w0, w1 := tw.u32[wp0], tw.u32[wp1]
							w2, w3 := tw.u32[wp2], tw.u32[wp3]
							wp0++
							wp1++
							wp2++
							wp3++
							x := x0[xi : xi+8 : xi+8]
							xi += 8
							a0, a1, a2, a3 := x[0], x[1], x[2], x[3]
							a4, a5, a6, a7 := x[4], x[5], x[6], x[7]
							q0 += a0*float32(w0&0xF) + a1*float32((w0>>4)&0xF) + a2*float32((w0>>8)&0xF) + a3*float32((w0>>12)&0xF) +
								a4*float32((w0>>16)&0xF) + a5*float32((w0>>20)&0xF) + a6*float32((w0>>24)&0xF) + a7*float32((w0>>28)&0xF)
							q1 += a0*float32(w1&0xF) + a1*float32((w1>>4)&0xF) + a2*float32((w1>>8)&0xF) + a3*float32((w1>>12)&0xF) +
								a4*float32((w1>>16)&0xF) + a5*float32((w1>>20)&0xF) + a6*float32((w1>>24)&0xF) + a7*float32((w1>>28)&0xF)
							q2 += a0*float32(w2&0xF) + a1*float32((w2>>4)&0xF) + a2*float32((w2>>8)&0xF) + a3*float32((w2>>12)&0xF) +
								a4*float32((w2>>16)&0xF) + a5*float32((w2>>20)&0xF) + a6*float32((w2>>24)&0xF) + a7*float32((w2>>28)&0xF)
							q3 += a0*float32(w3&0xF) + a1*float32((w3>>4)&0xF) + a2*float32((w3>>8)&0xF) + a3*float32((w3>>12)&0xF) +
								a4*float32((w3>>16)&0xF) + a5*float32((w3>>20)&0xF) + a6*float32((w3>>24)&0xF) + a7*float32((w3>>28)&0xF)
							sx += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
						}
						acc0 += ts.data[sB0+g]*q0 + tb.data[sB0+g]*sx
						acc1 += ts.data[sB1+g]*q1 + tb.data[sB1+g]*sx
						acc2 += ts.data[sB2+g]*q2 + tb.data[sB2+g]*sx
						acc3 += ts.data[sB3+g]*q3 + tb.data[sB3+g]*sx
					}
					y.data[o], y.data[o+1], y.data[o+2], y.data[o+3] = acc0, acc1, acc2, acc3
				}
				// Remainder channels (outDim not a multiple of 4): single channel,
				// dual-lane to keep some ILP.
				for ; o < oe; o++ {
					wp, sBase := o*P, o*G
					var acc float32
					xi := 0
					for g := 0; g < G; g++ {
						var xq0, xq1, sx0, sx1 float32
						for wc := 0; wc < wordsPerGroup; wc++ {
							word := tw.u32[wp]
							wp++
							x := x0[xi : xi+8 : xi+8]
							xi += 8
							a0, a1, a2, a3 := x[0], x[1], x[2], x[3]
							a4, a5, a6, a7 := x[4], x[5], x[6], x[7]
							xq0 += a0*float32(word&0xF) + a2*float32((word>>8)&0xF) +
								a4*float32((word>>16)&0xF) + a6*float32((word>>24)&0xF)
							xq1 += a1*float32((word>>4)&0xF) + a3*float32((word>>12)&0xF) +
								a5*float32((word>>20)&0xF) + a7*float32((word>>28)&0xF)
							sx0 += a0 + a2 + a4 + a6
							sx1 += a1 + a3 + a5 + a7
						}
						acc += ts.data[sBase+g]*(xq0+xq1) + tb.data[sBase+g]*(sx0+sx1)
					}
					y.data[o] = acc
				}
			})
			return y
		}
		parWork(outDim, in, func(os, oe int) {
			for o := os; o < oe; o++ {
				wp, sBase := o*P, o*G
				var acc float32
				xi := 0
				for g := 0; g < G; g++ {
					s, bb := ts.data[sBase+g], tb.data[sBase+g]
					var sumXQ, sumX float32
					for wc := 0; wc < wordsPerGroup; wc++ {
						word := tw.u32[wp]
						wp++
						sh := uint(0)
						for v := 0; v < valsPerWord; v++ {
							xv := x0[xi]
							xi++
							sumXQ += xv * float32((word>>sh)&mask)
							sumX += xv
							sh += uint(bits)
						}
					}
					acc += s*sumXQ + bb*sumX
				}
				y.data[o] = acc
			}
		})
		return y
	}

	// General/prefill path (rows > 1): unpack each weight row once into a reused
	// buffer and dot it against every x row, amortizing the unpack across the batch.
	parWork(outDim, in*rows, func(os, oe int) {
		wbuf := make([]float32, in) // reused across channels in this chunk
		for o := os; o < oe; o++ {
			wRow, sRow := o*P, o*G
			for i := 0; i < in; i++ {
				q := (tw.u32[wRow+i/valsPerWord] >> uint((i%valsPerWord)*bits)) & mask
				g := i / groupSize
				wbuf[i] = float32(q)*ts.data[sRow+g] + tb.data[sRow+g]
			}
			for r := 0; r < rows; r++ {
				xRow := tx.data[r*in : r*in+in]
				var acc float32
				for i := 0; i < in; i++ {
					acc += xRow[i] * wbuf[i]
				}
				y.data[r*outDim+o] = acc
			}
		}
	})
	return y
}
