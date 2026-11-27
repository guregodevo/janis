//go:build ggml

package ggml

/*
#include "ggml.h"
*/
import "C"

import (
	"github.com/guregodevo/janis/engine"
	"os"
)

// AppendValuesT appends to a value cache kept transposed, tokens innermost
// (engine.RelAttention): RelAttend multiplies by it in place, as SDPA does
// by AppendValues' cache where attention is not flash attention.
func (b *Backend) AppendValuesT(cache, x engine.Tensor) engine.Tensor {
	ta, tb := as(cache), as(x)
	if ta.dt != engine.F16 || tb.dt != engine.F16 || len(ta.shape) != 4 {
		return b.Concat(cache, x, 2)
	}
	return b.appendCache(ta, tb, 2, true)
}

// StepFloats is f(pos) as a tensor; in a recorded step, an input the step
// fills with f of each position it runs at (engine.RelAttention).
func (b *Backend) StepFloats(pos int, f func(pos int) []float32, shape ...int) engine.Tensor {
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	if rs == nil || rs.offs != nil {
		return b.FromFloats(f(pos), shape...)
	}
	if pos != rs.offset {
		rs.spoil("a position input away from the step's position")
		return b.FromFloats(f(pos), shape...)
	}
	t := b.replayInput(engine.F32, shape...)
	setFloat32(t, f(pos))
	b.mu.Lock()
	rs.floats = append(rs.floats, stepFloats{t: t, f: f})
	b.mu.Unlock()
	return t
}

// L2Clip is x*min(1, t/|x|) = t*x/max(|x|, t), and ggml's l2_norm(x, eps)
// is x/max(|x|, eps): two nodes (engine.RelAttention).
func (b *Backend) L2Clip(x engine.Tensor, threshold float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_scale(c, C.ggml_l2_norm(c, t, C.float(threshold)), C.float(threshold))
	})
}

// RMS is RMSNorm with no weight (engine.RelAttention).
func (b *Backend) RMS(x engine.Tensor, eps float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_rms_norm(c, t, C.float(eps))
	})
}

// relInputsFor is the distances and the mask of a query at pos over n
// cache slots: inputs of the step being recorded when the caches are its
// padded views, tensors made for this call otherwise.
func (b *Backend) relInputsFor(replay bool, pos, window, extent, n int) (idx, hidden *tensor) {
	if replay {
		key := [3]int{window, extent, n}
		b.mu.Lock()
		rs := b.rec0
		in := rs.rel[key]
		b.mu.Unlock()
		if in != nil {
			return in.idx, in.hidden
		}
		in = &relInputs{idx: b.replayInput(engine.I32, 1, n), hidden: b.replayInput(engine.F32, 1, n+1)}
		i, h := relMask(rs.offset, window, extent, n)
		setInt32(in.idx, i)
		setFloat32(in.hidden, h)
		b.mu.Lock()
		if rs.rel == nil {
			rs.rel = map[[3]int]*relInputs{}
		}
		rs.rel[key] = in
		rs.limit = min(rs.limit, n)
		b.mu.Unlock()
		return in.idx, in.hidden
	}
	i, h := relMask(pos, window, extent, n)
	return as(b.FromInt32(i, 1, n)), as(b.FromFloats(h, 1, n+1))
}

// RelAttend is Inkling 2's attention for one query (engine.RelAttention),
// on the nodes llama.cpp's graph of the model uses: the bias table is
// gathered by distance with one get_rows over the cache slots, the scores
// are k·q plus that bias, the sink logit joins the softmax as one more
// column, and the values are read transposed in place. The caches are read
// padded to the bucket a recorded step reads them at, the slots past the
// query masked: the graph is the same for every position in the bucket.
func (b *Backend) RelAttend(q, k, v, table, sinks engine.Tensor, pos, window int, scale float32) engine.Tensor {
	tq, tk, tv, tt, tsk := as(q), as(k), as(v), as(table), as(sinks)
	H, D := tq.shape[1], tq.shape[3]
	G, S := tk.shape[1], tk.shape[2]
	extent := tt.shape[3] - 1
	replay := tk.replay && tv.replay
	n := S
	if replay {
		b.mu.Lock()
		rs := b.rec0
		b.mu.Unlock()
		if rs == nil || rs.offs != nil {
			return nil
		}
		if pos != rs.offset {
			rs.spoil("attention away from the step's position")
			return nil
		}
	} else {
		if S != pos+1 {
			return nil
		}
		if n = tk.replayPad(tv); n <= S {
			return nil
		}
	}
	if tk.grp == nil || tv.grp == nil || !tv.grp.vt || tk.grp.vt {
		return nil
	}
	idx, hidden := b.relInputsFor(replay, pos, window, extent, n)
	per := H / G
	shape := []int{1, H, 1, D}
	return b.op(engine.F32, shape, []*tensor{tq, tk, tv, tt, tsk, idx, hidden}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		kk, vv := tk.t, tv.t
		if !replay {
			kk = cachePrefix(ctx, tk.grp, padTokens(tk.shape, n))
			vv = cachePrefix(ctx, tv.grp, padTokens(tv.shape, n))
		}
		if os.Getenv("JANIS_GGML_REL_F32") == "1" {
			// For tests: the CPU's f16 matmul rounds the query to f16 too,
			// which a tiny model of random weights turns into another reply.
			kk = C.ggml_cast(ctx, C.ggml_cont(ctx, kk), C.GGML_TYPE_F32)
			vv = C.ggml_cast(ctx, C.ggml_cont(ctx, vv), C.GGML_TYPE_F32)
		}
		cn, cH, cG, cper, cD := C.int64_t(n), C.int64_t(H), C.int64_t(G), C.int64_t(per), C.int64_t(D)
		// The bias of each head for each slot: rows of the table by distance.
		rows := C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tt)), C.int64_t(extent+1), cH, 1), 1, 0, 2, 3)) // ne (H, extent+1, 1)
		bias := C.ggml_get_rows(ctx, rows, C.ggml_reshape_2d(ctx, idx.t, cn, 1))                                                                 // ne (H, n, 1)
		bias = C.ggml_cont(ctx, C.ggml_permute(ctx, bias, 2, 0, 1, 3))                                                                           // ne (n, 1, H)
		// The scores of head g*per+p are key head g against the query.
		qq := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tq)), cD, cper, cG)
		kq := C.ggml_reshape_3d(ctx, C.ggml_mul_mat(ctx, kk, qq), cn, 1, cH) // ne (n, per, G) -> (n, 1, H)
		scores := C.ggml_add(ctx, C.ggml_scale(ctx, kq, C.float(scale)), bias)
		sink := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tsk)), 1, 1, cH)
		p := C.ggml_soft_max_ext(ctx, C.ggml_concat(ctx, scores, sink, 0), hidden.t, 1, 0) // ne (n+1, 1, H)
		p = C.ggml_cont(ctx, C.ggml_view_3d(ctx, p, cn, 1, cH, p.nb[1], p.nb[2], 0))
		vT := C.ggml_transpose(ctx, vv) // ne (n, D, G)
		if !C.ggml_is_transposed(vv) {
			vT = C.ggml_cont(ctx, vT)
		}
		out := C.ggml_mul_mat(ctx, vT, C.ggml_reshape_3d(ctx, p, cn, cper, cG)) // ne (D, per, G)
		return C.ggml_cont(ctx, C.ggml_reshape_4d(ctx, out, cD, 1, cH, 1))
	})
}

var _ engine.RelAttention = (*Backend)(nil)
