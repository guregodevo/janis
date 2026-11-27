//go:build ggml

package ggml

/*
#include "ggml.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/guregodevo/janis/engine"
)

// The operations of engine.LoopOps (Inkling 2), on the nodes llama.cpp's
// own graph of the model uses: argsort for the router's top k, batched
// get_rows for a gather along an axis, mul_mat_id over NVFP4 experts.

func (b *Backend) Sqrt(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_sqrt(c, t)
	})
}

func (b *Backend) Div(x, y engine.Tensor) engine.Tensor {
	return b.binary(x, y, func(ctx *C.struct_ggml_context, p, q *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_div(ctx, p, q)
	}, false)
}

// Minimum is x - relu(x - y): this ggml has no elementwise minimum.
func (b *Backend) Minimum(x, y engine.Tensor) engine.Tensor {
	return b.binary(x, y, func(ctx *C.struct_ggml_context, p, q *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_sub(ctx, p, C.ggml_relu(ctx, C.ggml_sub(ctx, p, q)))
	}, false)
}

// TopK is a descending argsort of each row cut at k, as llama.cpp picks a
// MoE layer's experts.
func (b *Backend) TopK(x engine.Tensor, k int) engine.Tensor {
	tx := as(x)
	shape := append([]int(nil), tx.shape...)
	shape[len(shape)-1] = k
	return b.op(engine.I32, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		top := C.ggml_argsort_top_k(ctx, contig(ctx, f32(ctx, tx)), C.int(k))
		if C.ggml_is_contiguous(top) { // one row: its first k lie together
			m := ne(shape)
			return C.ggml_reshape_4d(ctx, top, m[0], m[1], m[2], m[3])
		}
		return C.ggml_cont(ctx, top)
	})
}

// TakeAlong gathers along the last axis with ggml's batched get_rows: the
// axes the indices have are the batch, the axes they broadcast over (size
// 1 in indices) are the row that is taken whole.
func (b *Backend) TakeAlong(a, indices engine.Tensor, axis int) engine.Tensor {
	ta, ti := as(a), as(indices)
	nd := len(ta.shape)
	if axis < 0 {
		axis += nd
	}
	if axis != nd-1 || len(ti.shape) != nd {
		panic(fmt.Sprintf("ggml: TakeAlong of %v by %v along axis %d: only the last axis, at the same rank", ta.shape, ti.shape, axis))
	}
	var batch, wide []int
	T, W := 1, 1
	for d := 0; d < nd-1; d++ {
		switch {
		case ti.shape[d] == ta.shape[d]:
			batch = append(batch, d)
			T *= ta.shape[d]
		case ti.shape[d] == 1:
			wide = append(wide, d)
			W *= ta.shape[d]
		default:
			panic(fmt.Sprintf("ggml: TakeAlong of %v by %v: the indices are wider than the values", ta.shape, ti.shape))
		}
	}
	N, K := ta.shape[nd-1], ti.shape[nd-1]
	order := append(append(append([]int(nil), batch...), nd-1), wide...)
	src := a
	if len(wide) > 0 {
		src = b.Transpose(a, order...)
	}
	ts := as(src)
	moved := make([]int, nd) // the result's shape in the order of src
	for i, d := range order {
		moved[i] = ta.shape[d]
	}
	moved[len(batch)] = K
	got := b.op(engine.F32, moved, []*tensor{ts, ti}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		rows := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, ts)), C.int64_t(W), C.int64_t(N), C.int64_t(T))
		ids := C.ggml_reshape_2d(ctx, contig(ctx, ti.t), C.int64_t(K), C.int64_t(T))
		m := ne(moved)
		return C.ggml_reshape_4d(ctx, C.ggml_get_rows(ctx, rows, ids), m[0], m[1], m[2], m[3])
	})
	if len(wide) == 0 {
		return got
	}
	inv := make([]int, nd)
	for i, d := range order {
		inv[d] = i
	}
	return b.Transpose(got, inv...)
}

// e2m1 is a 4-bit float: sign, two bits of exponent, one of mantissa.
var e2m1 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// e4m3 is an 8-bit float: sign, four bits of exponent, three of mantissa.
func e4m3(x byte) float32 {
	exp, man := int(x>>3)&0xF, float64(x&7)
	var v float64
	if exp == 0 {
		v = math.Ldexp(man, -9)
	} else {
		v = math.Ldexp(1+man/8, exp-7)
	}
	if x&0x80 != 0 {
		v = -v
	}
	return float32(v)
}

// modeStack is a stack of experts in a floating-point format, ready for
// mul_mat_id: ne (in, out, E). nvfp4 rows of whole 64-value blocks become
// ggml's NVFP4, the same numbers in another byte order (4 scale bytes then
// 32 bytes of nibbles a block; value j of a group of 16 in the low nibble of
// byte j for j < 8, the high nibble of byte j-8 after). Anything else is
// decoded to f16. Built once per stack and pinned; the words go.
func (b *Backend) modeStack(tw, ts *tensor, groupSize, bits int, mode string) *tensor {
	if tw.q4 != nil {
		return tw.q4
	}
	words, scales := b.Bytes(tw), b.Bytes(ts)
	P := tw.shape[len(tw.shape)-1]
	in := P * 32 / bits
	rows := numel(tw.shape) / P
	G := in / groupSize
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	var typ C.enum_ggml_type
	var raw []byte
	var row func(r int)
	if mode == "nvfp4" && in%64 == 0 && os.Getenv("JANIS_GGML_DENSE_EXPERTS") != "1" {
		typ = C.GGML_TYPE_NVFP4
		raw = make([]byte, rows*(in/64)*36)
		row = func(r int) {
			wr, sr := words[r*P*4:], scales[r*G:]
			for blk := 0; blk < in/64; blk++ {
				o := (r*(in/64) + blk) * 36
				copy(raw[o:o+4], sr[blk*4:blk*4+4])
				for s := 0; s < 4; s++ {
					m := wr[(blk*4+s)*8:]
					for j := 0; j < 8; j++ {
						lo := m[j/2] >> (4 * uint(j%2)) & 0xF
						hi := m[4+j/2] >> (4 * uint(j%2)) & 0xF
						raw[o+4+s*8+j] = lo | hi<<4
					}
				}
			}
		}
	} else {
		typ = C.GGML_TYPE_F16
		raw = make([]byte, rows*in*2)
		var codes [256]float32 // a weight's code as a number
		for c := range codes {
			if bits == 4 {
				codes[c] = e2m1[c&0xF]
			} else {
				codes[c] = e4m3(byte(c))
			}
		}
		row = func(r int) {
			for g := 0; g < G; g++ {
				sb := scales[r*G+g]
				var d float32
				if mode == "nvfp4" {
					d = e4m3(sb & 0x7F)
				} else {
					d = float32(math.Ldexp(1, int(sb)-127))
				}
				for j := 0; j < groupSize; j++ {
					i := g*groupSize + j
					c := words[r*P*4+i]
					if bits == 4 {
						c = words[r*P*4+i/2] >> (4 * uint(i%2))
					}
					h := f16bits(codes[c] * d)
					raw[(r*in+i)*2], raw[(r*in+i)*2+1] = byte(h), byte(h>>8)
				}
			}
		}
	}
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for r := w; r < rows; r += workers {
				row(r)
			}
		}(w)
	}
	wg.Wait()
	q := b.newWeight(typ, shape, raw, mode+" experts")
	tw.q4 = q
	b.dropWords(tw)
	return q
}

// f16bits is v as IEEE half precision, rounded to nearest even; past the
// largest half it is that half, as a clipped weight.
func f16bits(v float32) uint16 {
	x := math.Float32bits(v)
	sign := uint16(x>>16) & 0x8000
	exp := int(x>>23&0xFF) - 127 + 15
	man := x & 0x7FFFFF
	switch {
	case exp >= 31:
		return sign | 0x7BFF
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint(14 - exp)
		h := man >> shift
		if rem, half := man&(1<<shift-1), uint32(1)<<(shift-1); rem > half || rem == half && h&1 == 1 {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(exp)<<10 | man>>13
	if rem := man & 0x1FFF; rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	return sign | uint16(h)
}

// GatherModeMatmul is GatherQuantMatmul for experts with no zero points.
func (b *Backend) GatherModeMatmul(x, w, scales, rhsIndices engine.Tensor, groupSize, bits int, mode string) engine.Tensor {
	tx, ti := as(x), as(rhsIndices)
	q := b.modeStack(as(w), as(scales), groupSize, bits, mode)
	xs, rs := tx.shape, ti.shape
	in, A := xs[len(xs)-1], xs[len(xs)-3]
	K := rs[len(rs)-1]
	nTok := numel(rs) / K
	shape := append(append([]int(nil), rs...), 1, q.shape[1])
	return b.op(engine.F32, shape, []*tensor{tx, ti, q, as(w)}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		xb := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tx)), C.int64_t(in), C.int64_t(A), C.int64_t(nTok))
		ids := C.ggml_reshape_2d(ctx, contig(ctx, ti.t), C.int64_t(K), C.int64_t(nTok))
		r := C.ggml_mul_mat_id(ctx, q.t, xb, ids) // ne (out, K, nTok)
		m := ne(shape)
		return C.ggml_reshape_4d(ctx, r, m[0], m[1], m[2], m[3])
	})
}

var _ engine.LoopOps = (*Backend)(nil)

// ggmlHalf is ggml's own conversion, which f16bits is tested against.
func ggmlHalf(v float32) uint16 { return uint16(C.ggml_fp32_to_fp16(C.float(v))) }
