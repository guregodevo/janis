//go:build ggml

package ggml

/*
#include "ggml.h"
#include "ggml-backend.h"
static struct ggml_tensor * janis_new4(struct ggml_context * c, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	return ggml_new_tensor_4d(c, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
}
*/
import "C"

import (
	"fmt"
	"github.com/guregodevo/janis/engine"
	"math"
	"os"
	"strconv"
	"strings"
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

// TrimFront is the cache without its first drop tokens, copied out of its
// buffer (engine.RelAttention): the next append moves it into a buffer of
// its own, whose first token is then at the position counted here.
func (b *Backend) TrimFront(cache engine.Tensor, drop int) engine.Tensor {
	tx := as(cache)
	shape := append([]int(nil), tx.shape...)
	shape[2] -= drop
	out := b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		t := tx.t
		if C.ggml_is_transposed(t) { // a value cache kept tokens innermost
			t = C.ggml_cont(ctx, t)
		}
		m := ne(shape)
		return C.ggml_cont(ctx, C.ggml_view_4d(ctx, t, m[0], m[1], m[2], m[3], t.nb[1], t.nb[2], t.nb[3], C.size_t(drop)*C.size_t(t.nb[1])))
	})
	out.off = tx.firstPos() + drop
	return out
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

// PrefillFit is the window by the device's free memory
// (engine.RelAttention), on CUDA; the CPU backend reads a prompt at the
// global window.
func (b *Backend) PrefillFit(perToken, perContext, perCached int64, tokens int) int {
	if !strings.HasPrefix(b.name, "CUDA") {
		return 0
	}
	return prefillFit(b.devFree()-perCached*int64(tokens), perToken, perContext, tokens)
}

func prefillFit(free, perToken, perContext int64, tokens int) int {
	each := perToken + perContext*int64(tokens)
	c := 64
	for c < 2048 && int64(2*c)*each <= free/2 {
		c *= 2
	}
	return c
}

// L2Clip is x*min(1, t/|x|) = t*x/max(|x|, t), and ggml's l2_norm(x, eps)
// is x/max(|x|, eps): two nodes (engine.RelAttention).
func (b *Backend) L2Clip(x engine.Tensor, threshold float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_scale(c, C.ggml_l2_norm(c, t, C.float(threshold)), C.float(threshold))
	})
}

// L2Unit is L2Clip without its last scale (engine.RelAttention): what
// reads it takes the threshold with a scale of its own.
func (b *Backend) L2Unit(x engine.Tensor, threshold float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_l2_norm(c, t, C.float(threshold))
	})
}

// RouteWeights is the chosen experts' scores over their sum, scaled
// (engine.RelAttention).
func (b *Backend) RouteWeights(scores, idx engine.Tensor, scale float32) engine.Tensor {
	ts, ti := as(scores), as(idx)
	L, E, K := ts.shape[1], ts.shape[2], ti.shape[2]
	return b.op(engine.F32, []int{1, L, K}, []*tensor{ts, ti}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cL, cK := C.int64_t(L), C.int64_t(K)
		rows := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, ts)), 1, C.int64_t(E), cL)
		w := C.ggml_reshape_2d(ctx, C.ggml_get_rows(ctx, rows, C.ggml_reshape_2d(ctx, contig(ctx, ti.t), cK, cL)), cK, cL)
		w = C.ggml_scale(ctx, C.ggml_div(ctx, w, C.ggml_sum_rows(ctx, w)), C.float(scale))
		return C.ggml_reshape_3d(ctx, w, cK, cL, 1)
	})
}

// CappedSwiGLU is the experts' activation on both halves at once
// (engine.RelAttention): one multiply by the two caps' inverses, one tanh
// and one sigmoid over gate and up together, then the halves multiplied
// as views of those.
func (b *Backend) CappedSwiGLU(gu engine.Tensor, gateCap, upCap float32) engine.Tensor {
	tg := as(gu)
	L, K, I := tg.shape[1], tg.shape[2], tg.shape[4]/2
	key := [3]float32{float32(I), gateCap, upCap}
	b.mu.Lock()
	caps := b.caps[key]
	b.mu.Unlock()
	if caps == nil {
		v := make([]float32, 2*I)
		for i := 0; i < I; i++ {
			v[i], v[I+i] = 1/gateCap, 1/upCap
		}
		caps = b.kept(as(b.FromFloats(v, 2*I)))
		b.mu.Lock()
		if b.caps == nil {
			b.caps = map[[3]float32]*tensor{}
		}
		b.caps[key] = caps
		b.mu.Unlock()
	}
	return b.op(engine.F32, []int{1, L, K, 1, I}, []*tensor{tg, caps}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cI, cK := C.int64_t(I), C.int64_t(K*L)
		g := C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tg)), 2*cI, cK)
		t := C.ggml_tanh(ctx, C.ggml_mul(ctx, g, caps.t))
		s := C.ggml_sigmoid(ctx, g)
		half := C.size_t(I) * C.size_t(t.nb[0])
		act := C.ggml_mul(ctx, C.ggml_view_2d(ctx, t, cI, cK, t.nb[1], 0), C.ggml_view_2d(ctx, t, cI, cK, t.nb[1], half))
		act = C.ggml_mul(ctx, act, C.ggml_view_2d(ctx, s, cI, cK, s.nb[1], 0))
		return C.ggml_reshape_4d(ctx, act, cI, 1, C.int64_t(K), C.int64_t(L))
	})
}

// PairRows marks a stack of experts not yet repacked: its repack puts the
// even output rows first, then the odd ones (engine.RelAttention).
func (b *Backend) PairRows(w engine.Tensor) bool {
	tw := as(w)
	if tw.q4 != nil || len(tw.shape) != 3 || tw.shape[1]%2 != 0 {
		return false
	}
	tw.pairRows = true
	return true
}

// RMS is RMSNorm with no weight (engine.RelAttention).
func (b *Backend) RMS(x engine.Tensor, eps float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_rms_norm(c, t, C.float(eps))
	})
}

// zeroSinks is a sink logit of zero for each of H heads.
func (b *Backend) zeroSinks(H int) *tensor {
	b.mu.Lock()
	z := b.zeros[H]
	b.mu.Unlock()
	if z == nil {
		z = b.kept(as(b.FromFloats(make([]float32, H), H)))
		b.mu.Lock()
		if b.zeros == nil {
			b.zeros = map[int]*tensor{}
		}
		b.zeros[H] = z
		b.mu.Unlock()
	}
	return z
}

// relInputsFor is the distances and the mask of a query at pos over the n
// cache slots from base: inputs of the step being recorded when the caches are its
// padded views, tensors made for this call otherwise.
func (b *Backend) relInputsFor(replay bool, pos, window, extent, base, n int) (idx, hidden *tensor) {
	if replay {
		key := [4]int{window, extent, base, n}
		b.mu.Lock()
		rs := b.rec0
		in := rs.rel[key]
		b.mu.Unlock()
		if in != nil {
			return in.idx, in.hidden
		}
		in = &relInputs{idx: b.replayInput(engine.I32, 1, n), hidden: b.replayInput(engine.F32, 1, n)}
		i, h := relMask(rs.offset, window, extent, base, n)
		setInt32(in.idx, i)
		setFloat32(in.hidden, h)
		b.mu.Lock()
		if rs.rel == nil {
			rs.rel = map[[4]int]*relInputs{}
		}
		rs.rel[key] = in
		rs.limit = min(rs.limit, base+n)
		b.mu.Unlock()
		return in.idx, in.hidden
	}
	i, h := relMask(pos, window, extent, base, n)
	return as(b.FromInt32(i, 1, n)), as(b.FromFloats(h, 1, n))
}

// RelAttend is Inkling 2's attention for one query (engine.RelAttention),
// on the nodes llama.cpp's graph of the model uses: the bias table is
// gathered by distance with one get_rows over the cache slots, the scores
// are k·q plus that bias, the sink logit joins the softmax as one more
// column, and the values are read transposed in place. The caches are read
// padded to the bucket a recorded step reads them at, the slots past the
// query masked: the graph is the same for every position in the bucket.
func (b *Backend) RelAttend(q, k, v, rel, relProj, sinks engine.Tensor, pos, window int, scale, relScale float32) engine.Tensor {
	tq, tk, tv, tr, tsk := as(q), as(k), as(v), as(rel), as(sinks)
	// relProj [d_rel, R+1] as ggml_mul_mat's first operand, [R+1, d_rel]:
	// made once of a weight, else copied in the graph.
	tp := as(relProj)
	dr, R1 := tp.shape[0], tp.shape[1]
	proj := b.madeOf(tp, fmt.Sprintf("T*%g", relScale), func(v []float32) ([]float32, []int) {
		out := make([]float32, len(v))
		for k := 0; k < dr; k++ {
			for n := 0; n < R1; n++ {
				out[n*dr+k] = v[k*R1+n] * relScale
			}
		}
		return out, []int{R1, dr}
	})
	H, D := tq.shape[1], tq.shape[3]
	G, S := tk.shape[1], tk.shape[2]
	extent := tp.shape[1] - 1
	if tq.shape[2] > 1 {
		return b.relAttendRows(tq, tk, tv, tr, tp, proj, tsk, pos, window, scale, relScale)
	}
	replay := tk.replay && tv.replay
	// The keys' and the values' buffers are sized apart, from the memory
	// free when each was made (reservedSpare): a recorded step reads no
	// further than the shorter of the two. Read to the keys' bucket, the
	// values' view ran past its buffer (an abort at 11,008 tokens of a
	// reply, RTX 5000 Ada, 2026-11-27).
	n := min(S, tv.shape[2])
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
		if S+tk.firstPos() != pos+1 {
			return nil
		}
		if n = tk.replayPad(tv); n <= S {
			return nil
		}
	}
	if tk.grp == nil || tv.grp == nil || tk.grp.base == nil || tv.grp.base == nil || !tv.grp.vt || tk.grp.vt || tk.grp.off != tv.grp.off {
		return nil
	}
	// A sliding layer reads the slots its window reaches from here to the
	// end of the bucket, not the whole cache: a recorded step runs at this
	// position and the ones after it. base is the first position read,
	// slot the cache's slot that holds it (a cache cut at its front,
	// TrimFront, starts past position 0).
	off := tk.grp.off
	base := off
	if window > 0 {
		base = max(off, pos-(window-1))
	}
	slot := base - off
	n -= slot
	idx, hidden := b.relInputsFor(replay, pos, window, extent, base, n)
	zero := b.zeroSinks(H)
	per := H / G
	shape := []int{1, H, 1, D}
	return b.op(engine.F32, shape, []*tensor{tq, tk, tv, tr, tp, proj, tsk, idx, hidden, zero}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cn, cH, cG, cper, cD := C.int64_t(n), C.int64_t(H), C.int64_t(G), C.int64_t(per), C.int64_t(D)
		// The slots base.. of the caches, where they are: the keys
		// ne (D, n, G), the values kept tokens innermost ne (n, D, G).
		kb, vb := tk.grp.base, tv.grp.base
		kk := C.ggml_view_4d(ctx, kb, cD, cn, cG, 1, kb.nb[1], kb.nb[2], kb.nb[3], C.size_t(slot)*C.size_t(kb.nb[1]))
		C.ggml_backend_view_init(kk)
		vT := C.ggml_view_4d(ctx, vb, cn, cD, cG, 1, vb.nb[1], vb.nb[2], vb.nb[3], C.size_t(slot)*C.size_t(vb.nb[0]))
		C.ggml_backend_view_init(vT)
		if os.Getenv("JANIS_GGML_REL_F32") == "1" {
			// For tests: the CPU's f16 matmul rounds the query to f16 too,
			// which a tiny model of random weights turns into another reply.
			kk = C.ggml_cast(ctx, C.ggml_cont(ctx, kk), C.GGML_TYPE_F32)
			vT = C.ggml_cast(ctx, C.ggml_cont(ctx, vT), C.GGML_TYPE_F32)
		}
		// The bias of each head at each distance, ne (H, extent+1), then of
		// each head for each slot: rows of that table by the slot's distance.
		var pT *C.struct_ggml_tensor
		if proj != nil {
			pT = proj.t
		} else {
			pT = C.ggml_scale(ctx, C.ggml_cont(ctx, C.ggml_transpose(ctx, f32(ctx, tp))), C.float(relScale))
		}
		rows := C.ggml_mul_mat(ctx, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tr)), C.int64_t(dr), cH), pT)
		// The sink logit t joins each head's softmax: exp(s - t)/(1 + sum
		// exp(s - t)) is exp(s)/(exp(t) + sum exp(s)), so t leaves the
		// table and the softmax takes a sink of zero, as llama.cpp does.
		rows = C.ggml_sub(ctx, rows, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tsk)), cH, 1))
		bias := C.ggml_get_rows(ctx, C.ggml_reshape_3d(ctx, rows, cH, C.int64_t(extent+1), 1), C.ggml_reshape_2d(ctx, idx.t, cn, 1)) // ne (H, n, 1)
		bias = C.ggml_cont(ctx, C.ggml_permute(ctx, bias, 2, 0, 1, 3))                                                               // ne (n, 1, H)
		// The scores of head g*per+p are key head g against the query.
		qq := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tq)), cD, cper, cG)
		kq := C.ggml_reshape_3d(ctx, C.ggml_mul_mat(ctx, kk, qq), cn, 1, cH) // ne (n, per, G) -> (n, 1, H)
		scores := C.ggml_add(ctx, C.ggml_scale(ctx, kq, C.float(scale)), bias)
		p := C.ggml_soft_max_ext(ctx, scores, hidden.t, 1, 0) // ne (n, 1, H)
		C.ggml_soft_max_add_sinks(p, zero.t)
		out := C.ggml_mul_mat(ctx, vT, C.ggml_reshape_3d(ctx, p, cn, cper, cG)) // ne (D, per, G)
		return C.ggml_reshape_4d(ctx, out, cD, 1, cH, 1)
	})
}

// relPieceBytes bounds what a chunk's attention holds at its peak: the
// queries are taken in pieces that keep it under this.
const relPieceBytes = 512 << 20

// relSlidingPiece is how many queries of a chunk a sliding layer takes at a
// time: each piece reads the slots its own queries' windows reach, window-1
// before its first query and its own. A whole chunk of 1,024 queries reads
// 2,047 slots for each, half of them masked; pieces of 128 read 1,151. A
// 3.2k-token prompt read at 868, 956 and 1,006 tok/s in pieces of 512, 256
// and 128 (RTX 5000 Ada, 2026-11-27).
const relSlidingPiece = 128

// relRows are the inputs of one piece of a chunk's attention, kept for the
// forward pass that made them: every layer of a kind reads the same.
type relRows struct {
	idx, hidden *tensor
}

// relRowsFor is, for the ql queries at qpos.. over the n cache slots that
// hold positions base..base+n-1: each slot's bias row for each query (its
// distance, extent from there on), I32 [ql, n], and the mask [ql, n] (-inf
// where the query does not see the slot). pass names
// the forward pass, its position and length: the inputs of the pass before
// are freed when another begins.
func (b *Backend) relRowsFor(pass [2]int, qpos, ql, window, extent, base, n int) (idx, hidden *tensor) {
	key := [6]int{qpos, ql, window, extent, base, n}
	b.mu.Lock()
	var old map[[6]int]*relRows
	if b.relPass != pass {
		old, b.relRows, b.relPass = b.relRows, map[[6]int]*relRows{}, pass
	}
	in := b.relRows[key]
	b.mu.Unlock()
	if len(old) > 0 {
		b.flush() // the pass before may still be computing with them
		for _, o := range old {
			o.idx.free()
			o.hidden.free()
		}
	}
	if in != nil {
		return in.idx, in.hidden
	}
	iv, hv := make([]int32, ql*n), make([]float32, ql*n)
	ninf := float32(math.Inf(-1))
	for i := 0; i < ql; i++ {
		for j := 0; j < n; j++ {
			d := qpos + i - (base + j)
			iv[i*n+j] = int32(extent)
			switch {
			case d < 0 || (window > 0 && d >= window):
				hv[i*n+j] = ninf
			case d < extent:
				iv[i*n+j] = int32(d)
			}
		}
	}
	in = &relRows{idx: b.kept(as(b.FromInt32(iv, ql, n))), hidden: b.kept(as(b.FromFloats(hv, ql, n)))}
	b.mu.Lock()
	b.relRows[key] = in
	b.mu.Unlock()
	return in.idx, in.hidden
}

// relAttendRows is RelAttend for a chunk of L queries at pos.., the keys
// and values read in place from their caches. The queries are taken in
// pieces: on a sliding layer each piece reads the slots its windows reach;
// on a full one each reads every slot, and the pieces bound the tensors of
// slots by queries by heads the attention goes through (the gathered bias,
// the scores, the weights), which are the graph's to reuse inside one
// operation. Nil when the caches are not buffers it can read in place (the
// first chunk of a prompt), which the model's own operations compute.
func (b *Backend) relAttendRows(tq, tk, tv, tr, tp, proj, tsk *tensor, pos, window int, scale, relScale float32) engine.Tensor {
	H, L, D := tq.shape[1], tq.shape[2], tq.shape[3]
	G, S := tk.shape[1], tk.shape[2]
	dr, extent := tp.shape[0], tp.shape[1]-1
	if tk.replay || tv.replay || tk.grp == nil || tv.grp == nil || tk.grp.base == nil || tv.grp.base == nil ||
		tk.grp.vt || !tv.grp.vt || tk.cacheLen != S || tv.cacheLen != S || tk.grp.off != tv.grp.off || S+tk.grp.off != pos+L {
		return nil
	}
	// A cache cut at its front (TrimFront) holds the positions from off.
	off := tk.grp.off
	if (window == 0 && off != 0) || (window > 0 && off > max(0, pos-(window-1))) {
		return nil
	}
	per := H / G
	// Five tensors of slots by queries by heads floats are alive at the
	// peak: a piece is the queries that keep them under relPieceBytes.
	piece := max(16, min(L, relPieceBytes/(5*4*H*S)))
	if window > 0 {
		piece = min(piece, relSlidingPiece)
	}
	if v, err := strconv.Atoi(os.Getenv("JANIS_GGML_REL_PIECE")); err == nil && v > 0 {
		piece = v // for measuring, and for tests
	}
	type part struct {
		l0, ql, base, n int
		idx, hidden     *tensor
	}
	var parts []part
	inputs := []*tensor{tq, tk, tv, tr, tp, proj, tsk}
	for l0 := 0; l0 < L; l0 += piece {
		pt := part{l0: l0, ql: min(piece, L-l0)}
		end := pos + l0 + pt.ql // one past the piece's last query
		pt.base = off
		if window > 0 {
			pt.base = max(off, pos+l0-(window-1))
		}
		pt.n = end - pt.base
		pt.idx, pt.hidden = b.relRowsFor([2]int{pos, L}, pos+l0, pt.ql, window, extent, pt.base, pt.n)
		parts = append(parts, pt)
		inputs = append(inputs, pt.idx, pt.hidden)
	}
	f32read := os.Getenv("JANIS_GGML_REL_F32") == "1"
	zero := b.zeroSinks(H)
	inputs = append(inputs, zero)
	return b.op(engine.F32, []int{1, H, L, D}, inputs, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cL, cH, cG, cD, cper := C.int64_t(L), C.int64_t(H), C.int64_t(G), C.int64_t(D), C.int64_t(per)
		kb, vb := tk.grp.base, tv.grp.base
		var pT *C.struct_ggml_tensor
		if proj != nil {
			pT = proj.t
		} else {
			pT = C.ggml_scale(ctx, C.ggml_cont(ctx, C.ggml_transpose(ctx, f32(ctx, tp))), C.float(relScale))
		}
		// Each head's bias at each distance for each query, ne (H, L,
		// extent+1), turned to (H, extent+1, L).
		rows := C.ggml_mul_mat(ctx, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tr)), C.int64_t(dr), cH*cL), pT)
		// Less each query's sink logit, which the softmax then takes as a
		// sink of zero (RelAttend).
		rows = C.ggml_sub(ctx, rows, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tsk)), cH*cL, 1))
		rows = C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, rows, cH, cL, C.int64_t(extent+1)), 0, 2, 1, 3))
		// The queries with the scale in them: one pass over the scores less.
		q4 := C.ggml_reshape_4d(ctx, C.ggml_scale(ctx, contig(ctx, f32(ctx, tq)), C.float(scale)), cD, cL, cper, cG)
		var out *C.struct_ggml_tensor
		for _, pt := range parts {
			cn, ql, o := C.int64_t(pt.n), C.int64_t(pt.ql), C.size_t(pt.l0)
			// The slots base.. of the caches, where they are: the keys
			// ne (D, n, G), the values kept tokens innermost ne (n, D, G).
			kk := C.ggml_view_4d(ctx, kb, cD, cn, cG, 1, kb.nb[1], kb.nb[2], kb.nb[3], C.size_t(pt.base-off)*C.size_t(kb.nb[1]))
			C.ggml_backend_view_init(kk)
			vT := C.ggml_view_4d(ctx, vb, cn, cD, cG, 1, vb.nb[1], vb.nb[2], vb.nb[3], C.size_t(pt.base-off)*C.size_t(vb.nb[0]))
			C.ggml_backend_view_init(vT)
			if f32read {
				kk = C.ggml_cast(ctx, C.ggml_cont(ctx, kk), C.GGML_TYPE_F32)
				vT = C.ggml_cast(ctx, C.ggml_cont(ctx, vT), C.GGML_TYPE_F32)
			}
			rp := C.ggml_view_3d(ctx, rows, cH, C.int64_t(extent+1), ql, rows.nb[1], rows.nb[2], o*C.size_t(rows.nb[2]))
			bias := C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_get_rows(ctx, rp, pt.idx.t), 2, 0, 1, 3)) // ne (n, ql, H)
			qp := C.ggml_view_4d(ctx, q4, cD, ql, cper, cG, q4.nb[1], q4.nb[2], q4.nb[3], o*C.size_t(q4.nb[1]))
			qp = C.ggml_reshape_3d(ctx, contig(ctx, qp), cD, ql*cper, cG)
			scores := C.ggml_add(ctx, C.ggml_reshape_3d(ctx, C.ggml_mul_mat(ctx, kk, qp), cn, ql, cH), bias)
			p := C.ggml_soft_max_ext(ctx, scores, pt.hidden.t, 1, 0)
			C.ggml_soft_max_add_sinks(p, zero.t)
			op := C.ggml_reshape_4d(ctx, C.ggml_mul_mat(ctx, vT, C.ggml_reshape_3d(ctx, p, cn, ql*cper, cG)), cD, ql, cper, cG)
			if out == nil {
				out = op
			} else {
				out = C.ggml_concat(ctx, out, op, 1)
			}
		}
		return C.ggml_reshape_4d(ctx, out, cD, cL, cH, 1)
	})
}

// DepthMix is the mix of the states for each query (engine.RelAttention),
// as llama.cpp builds it: the scores are one matmul against the queries,
// the softmax takes the scale, and the states are copied transposed once
// for the product with the weights.
func (b *Backend) DepthMix(stack, queryT engine.Tensor) engine.Tensor {
	ts := as(stack)
	S, D := ts.shape[2], ts.shape[3]
	tq := as(queryT)
	q := b.transposedWeight(tq) // [Q, D]
	Q := tq.shape[1]
	if L := ts.shape[1]; L > 1 {
		// A chunk: the scores of every token's states in one matmul, the
		// states as its columns; the product with the weights a batch.
		return b.op(engine.F32, []int{1, L, Q, D}, []*tensor{ts, tq, q}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
			cD, cS, cL, cQ := C.int64_t(D), C.int64_t(S), C.int64_t(L), C.int64_t(Q)
			st := contig(ctx, f32(ctx, ts))
			scores := C.ggml_mul_mat(ctx, transposed(ctx, tq, q), C.ggml_reshape_2d(ctx, st, cD, cS*cL)) // ne (Q, S*L)
			if Q == 1 {
				scores = C.ggml_reshape_3d(ctx, scores, cS, 1, cL)
			} else {
				scores = C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, scores, cQ, cS, cL), 1, 0, 2, 3))
			}
			p := C.ggml_soft_max_ext(ctx, scores, nil, C.float(1/math.Sqrt(float64(D))), 0)                  // ne (S, Q, L)
			stT := C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, st, cD, cS, cL), 1, 0, 2, 3)) // ne (S, D, L)
			return C.ggml_reshape_4d(ctx, C.ggml_mul_mat(ctx, stT, p), cD, cQ, cL, 1)
		})
	}
	return b.op(engine.F32, []int{1, 1, Q, D}, []*tensor{ts, tq, q}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		st := C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, ts)), C.int64_t(D), C.int64_t(S))
		p := C.ggml_soft_max_ext(ctx, C.ggml_mul_mat(ctx, st, transposed(ctx, tq, q)), nil, C.float(1/math.Sqrt(float64(D))), 0) // ne (S, Q)
		out := C.ggml_mul_mat(ctx, C.ggml_cont(ctx, C.ggml_transpose(ctx, st)), p)                                               // ne (D, Q)
		return C.ggml_reshape_4d(ctx, out, C.int64_t(D), C.int64_t(Q), 1, 1)
	})
}

// WeightedSum is y's k rows summed by w (engine.RelAttention): the rows
// copied transposed, then one matmul.
func (b *Backend) WeightedSum(y, w engine.Tensor) engine.Tensor {
	ty, tw := as(y), as(w)
	L, K, D := ty.shape[1], ty.shape[2], ty.shape[3]
	return b.op(engine.F32, []int{1, L, D}, []*tensor{ty, tw}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cL, cK, cD := C.int64_t(L), C.int64_t(K), C.int64_t(D)
		yT := C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, ty)), cD, cK, cL), 1, 0, 2, 3)) // ne (K, D, L)
		ww := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tw)), cK, 1, cL)
		return C.ggml_reshape_3d(ctx, C.ggml_mul_mat(ctx, yT, ww), cD, cL, 1)
	})
}

// ConvTaps is one position of a short convolution (engine.RelAttention):
// the window times the taps, summed over the K inputs of each channel.
// The taps are made once from the filter, with the residual (one more on
// the newest input) and the 1/sqrt(K+1) in them.
func (b *Backend) ConvTaps(window, taps engine.Tensor) engine.Tensor {
	tw := as(window)
	C_, K := tw.shape[1], tw.shape[2]
	w := b.madeOf(as(taps), "taps", func(v []float32) ([]float32, []int) {
		s := float32(1 / math.Sqrt(float64(K+1)))
		out := make([]float32, len(v))
		for c := 0; c < C_; c++ {
			for k := 0; k < K; k++ {
				out[c*K+k] = v[c*K+k] * s
			}
			out[c*K+K-1] += s
		}
		return out, []int{1, C_, K}
	})
	if w == nil {
		return nil
	}
	return b.op(engine.F32, []int{1, 1, C_}, []*tensor{tw, w}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		sum := C.ggml_sum_rows(ctx, C.ggml_mul(ctx, f32(ctx, tw), w.t)) // ne (1, C)
		return C.ggml_reshape_3d(ctx, sum, C.int64_t(C_), 1, 1)
	})
}

// ShiftIn is the window of the next position (engine.RelAttention): one
// concat of the K-1 newest inputs, read where they are, and x.
func (b *Backend) ShiftIn(window, x engine.Tensor) engine.Tensor {
	tw, tx := as(window), as(x)
	C_, K := tw.shape[1], tw.shape[2]
	return b.op(engine.F32, []int{1, C_, K}, []*tensor{tw, tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		w := f32(ctx, tw)
		newest := C.ggml_view_2d(ctx, w, C.int64_t(K-1), C.int64_t(C_), w.nb[1], w.nb[0])
		return C.ggml_concat(ctx, newest, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tx)), 1, C.int64_t(C_)), 0)
	})
}

// ConvTapsDyn is ConvTaps with a filter the input chooses per head: the
// static part is made once in the window's order (newest last, so by lag
// reversed) with the residual in it, the dynamic part is put in that
// order by a gather, and the sum is scaled by 1/sqrt(2K+1).
func (b *Backend) ConvTapsDyn(window, static, dyn engine.Tensor) engine.Tensor {
	tw, td := as(window), as(dyn)
	C_, K := tw.shape[1], tw.shape[2]
	ts := as(static)
	H := ts.shape[0]
	D := C_ / H
	w := b.madeOf(ts, "taps", func(v []float32) ([]float32, []int) {
		out := make([]float32, len(v))
		for c := 0; c < C_; c++ {
			for k := 0; k < K; k++ {
				out[c*K+k] = v[c*K+K-1-k]
			}
			out[c*K+K-1]++
		}
		return out, []int{H, D, K}
	})
	if w == nil {
		return nil
	}
	b.mu.Lock()
	order := b.lagOrder[[2]int{H, K}]
	b.mu.Unlock()
	if order == nil {
		idx := make([]int32, H*K)
		for h := 0; h < H; h++ {
			for k := 0; k < K; k++ {
				idx[h*K+k] = int32(h*K + K - 1 - k)
			}
		}
		order = b.kept(as(b.FromInt32(idx, H*K)))
		b.mu.Lock()
		if b.lagOrder == nil {
			b.lagOrder = map[[2]int]*tensor{}
		}
		b.lagOrder[[2]int{H, K}] = order
		b.mu.Unlock()
	}
	scale := float32(1 / math.Sqrt(float64(2*K+1)))
	return b.op(engine.F32, []int{1, 1, C_}, []*tensor{tw, w, td, order}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cK, cD, cH := C.int64_t(K), C.int64_t(D), C.int64_t(H)
		d := C.ggml_get_rows(ctx, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, td)), 1, cH*cK), order.t) // ne (1, H*K), newest last
		filter := C.ggml_add(ctx, w.t, C.ggml_reshape_3d(ctx, d, cK, 1, cH))                            // ne (K, D, H)
		win := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tw)), cK, cD, cH)
		sum := C.ggml_scale(ctx, C.ggml_sum_rows(ctx, C.ggml_mul(ctx, win, filter)), C.float(scale)) // ne (1, D, H)
		return C.ggml_reshape_3d(ctx, sum, C.int64_t(C_), 1, 1)
	})
}

// GateMix is a looped layer's mix of its input with the message of the
// pass before (engine.RelAttention), as llama.cpp's control: the two as
// rows of one rms_norm, one matmul down, one matmul for both gates (the
// forget gate's weights negated: tanh is odd), and the two products added
// to the input.
func (b *Backend) GateMix(h, message, downT, forgetT, admitT engine.Tensor, eps float32) engine.Tensor {
	th, tm, tf, ta := as(h), as(message), as(forgetT), as(admitT)
	D := th.shape[2]
	down := b.transposedWeight(as(downT)) // [rank, 2D]
	if down == nil || !isWeight(ta) {
		return nil
	}
	rank := down.shape[0]
	// Both gates as one matrix [2D, rank]: -forget's columns, then admit's.
	gates := b.madeOf(tf, "gates", func(f []float32) ([]float32, []int) {
		a := weightFloats(ta)
		if a == nil {
			return nil, nil
		}
		out := make([]float32, 2*D*rank)
		for r := 0; r < rank; r++ {
			for i := 0; i < D; i++ {
				out[i*rank+r] = -f[r*D+i]
				out[(D+i)*rank+r] = a[r*D+i]
			}
		}
		return out, []int{2 * D, rank}
	})
	if gates == nil {
		return nil
	}
	return b.op(engine.F32, []int{1, 1, D}, []*tensor{th, tm, down, gates}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cD := C.int64_t(D)
		hh := C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, th)), cD, 1)
		both := C.ggml_concat(ctx, hh, C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tm)), cD, 1), 1) // ne (D, 2)
		s := C.ggml_mul_mat(ctx, down.t, C.ggml_reshape_2d(ctx, C.ggml_rms_norm(ctx, both, C.float(eps)), 2*cD, 1))
		g := C.ggml_reshape_2d(ctx, C.ggml_tanh(ctx, C.ggml_mul_mat(ctx, gates.t, s)), cD, 2) // -forget, admit
		prod := C.ggml_mul(ctx, g, both)
		out := C.ggml_add(ctx, hh, C.ggml_view_2d(ctx, prod, cD, 1, prod.nb[1], 0))
		out = C.ggml_add(ctx, out, C.ggml_view_2d(ctx, prod, cD, 1, prod.nb[1], prod.nb[1]))
		return C.ggml_reshape_3d(ctx, out, cD, 1, 1)
	})
}

// GateMixRows is GateMix for n inputs as one batch (engine.RelAttention):
// the same eight nodes, the two matmuls batched over the n gates' weights
// stacked once.
func (b *Backend) GateMixRows(h, messages engine.Tensor, downT, forgetT, admitT []engine.Tensor, eps float32) engine.Tensor {
	th, tm := as(h), as(messages)
	n := len(downT)
	D := th.shape[len(th.shape)-1]
	d0 := as(downT[0])
	if len(d0.shape) != 2 {
		return nil
	}
	rank := d0.shape[1]
	for i := 0; i < n; i++ {
		if !isWeight(as(downT[i])) || !isWeight(as(forgetT[i])) || !isWeight(as(admitT[i])) {
			return nil
		}
	}
	// The n down projections [2D, rank] as [n, rank, 2D], ggml_mul_mat's
	// first operand, and the n pairs of gates as [n, 2D, rank]: -forget's
	// columns, then admit's.
	down := b.madeOf(d0, fmt.Sprintf("down*%d", n), func([]float32) ([]float32, []int) {
		out := make([]float32, n*rank*2*D)
		for i := 0; i < n; i++ {
			v := weightFloats(as(downT[i]))
			if v == nil {
				return nil, nil
			}
			for r := 0; r < rank; r++ {
				for k := 0; k < 2*D; k++ {
					out[(i*rank+r)*2*D+k] = v[k*rank+r]
				}
			}
		}
		return out, []int{n, rank, 2 * D}
	})
	gates := b.madeOf(as(forgetT[0]), fmt.Sprintf("gates*%d", n), func([]float32) ([]float32, []int) {
		out := make([]float32, n*2*D*rank)
		for i := 0; i < n; i++ {
			f, a := weightFloats(as(forgetT[i])), weightFloats(as(admitT[i]))
			if f == nil || a == nil {
				return nil, nil
			}
			for r := 0; r < rank; r++ {
				for k := 0; k < D; k++ {
					out[(i*2*D+k)*rank+r] = -f[r*D+k]
					out[(i*2*D+D+k)*rank+r] = a[r*D+k]
				}
			}
		}
		return out, []int{n, 2 * D, rank}
	})
	if down == nil || gates == nil {
		return nil
	}
	// h is [1, L, n, D], or [1, L, D] for one input shared by the n gates.
	shared := len(th.shape) == 3
	L := th.shape[1]
	return b.op(engine.F32, []int{1, L, n, D}, []*tensor{th, tm, down, gates}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		cD, cL, cn := C.int64_t(D), C.int64_t(L), C.int64_t(n)
		// Each gate's tokens as columns, the gates the batch: ne (D, 1, L, n).
		rows := func(t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
			if L == 1 {
				return C.ggml_reshape_4d(ctx, t, cD, 1, 1, cn)
			}
			return C.ggml_reshape_4d(ctx, C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, t, cD, cn, cL), 0, 2, 1, 3)), cD, 1, cL, cn)
		}
		var hh *C.struct_ggml_tensor
		if shared {
			hh = C.ggml_repeat(ctx, C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, th)), cD, 1, cL, 1), C.janis_new4(ctx, cD, 1, cL, cn))
		} else {
			hh = rows(contig(ctx, f32(ctx, th)))
		}
		mm := rows(contig(ctx, f32(ctx, tm)))
		both := C.ggml_concat(ctx, hh, mm, 1) // ne (D, 2, L, n)
		s := C.ggml_mul_mat(ctx, down.t, C.ggml_reshape_3d(ctx, C.ggml_rms_norm(ctx, both, C.float(eps)), 2*cD, cL, cn))
		g := C.ggml_reshape_4d(ctx, C.ggml_tanh(ctx, C.ggml_mul_mat(ctx, gates.t, s)), cD, 2, cL, cn) // -forget, admit
		prod := C.ggml_mul(ctx, g, both)
		out := C.ggml_add(ctx, hh, C.ggml_view_4d(ctx, prod, cD, 1, cL, cn, prod.nb[1], prod.nb[2], prod.nb[3], 0))
		out = C.ggml_add(ctx, out, C.ggml_view_4d(ctx, prod, cD, 1, cL, cn, prod.nb[1], prod.nb[2], prod.nb[3], prod.nb[1]))
		if L == 1 {
			return C.ggml_reshape_2d(ctx, out, cD, cn)
		}
		// Back to each token's n rows together: ne (D, n, L).
		return C.ggml_cont(ctx, C.ggml_permute(ctx, C.ggml_reshape_3d(ctx, out, cD, cL, cn), 0, 2, 1, 3))
	})
}

// transposed is matrix w transposed: made, its copy made once, else a
// copy in the graph.
func transposed(ctx *C.struct_ggml_context, w, made *tensor) *C.struct_ggml_tensor {
	if made != nil {
		return made.t
	}
	return C.ggml_cont(ctx, C.ggml_transpose(ctx, f32(ctx, w)))
}

var _ engine.RelAttention = (*Backend)(nil)
