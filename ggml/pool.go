//go:build ggml

package ggml

/*
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
*/
import "C"

import (
	"github.com/guregodevo/janis/engine"
)

// A batched decode step's attention read each conversation's own cache:
// about nine kernels a conversation in every attention layer, 1.9 ms a
// conversation a step on the 27B at 32 (2026-11-22). A pool keeps the B
// conversations' keys and values of one layer in one cache buffer, B rows
// of room tokens, as llama.cpp keeps its sequences' streams: one
// set_rows writes every row's new key, one the values, and one k·q, one
// softmax with a [B, L] mask and one v·scores read every row.

// kvPool is one attention layer's keys and values for B conversations.
type kvPool struct {
	b          *Backend
	k, v       *tensor // views of the whole buffers, [B, Hk, room, D]; they hold the buffers
	rows, room int
	hk, d, dv  int
}

// PoolKV copies conversation j's keys ks[j] and values vs[j] [1, Hk, Lj, D]
// (nil for a conversation with none yet) into a new pool of room tokens a
// row (engine.KVPooler).
func (b *Backend) PoolKV(ks, vs []engine.Tensor, hk, d, dv, room int) engine.Pool {
	B := len(ks)
	room = (room + kvPad - 1) / kvPad * kvPad
	p := &kvPool{b: b, rows: B, room: room, hk: hk, d: d, dv: dv}
	p.k = b.poolBuffer([]int{B, hk, room, d}, false)
	p.v = b.poolBuffer([]int{B, hk, room, dv}, !b.flash)
	b.mu.Lock()
	if b.rec == nil {
		b.newRec()
	}
	for j := range ks {
		if ks[j] == nil {
			continue
		}
		b.writeCacheRow(p.k.grp, as(ks[j]), j)
		b.writeCacheRow(p.v.grp, as(vs[j]), j)
	}
	b.mu.Unlock()
	return p
}

// writeCacheRow copies x [1, H, L, D] into row j of a pool buffer, at its
// first token. Called with b.mu held.
func (b *Backend) writeCacheRow(grp *batch, x *tensor, j int) {
	m := ne(x.shape)
	base := grp.base
	var dst *C.struct_ggml_tensor
	if grp.vt {
		d := C.ggml_view_4d(b.rec, base, m[1], m[0], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], C.size_t(j)*C.size_t(base.nb[3]))
		C.ggml_backend_view_init(d)
		dst = C.ggml_permute(b.rec, d, 1, 0, 2, 3)
	} else {
		dst = C.ggml_view_4d(b.rec, base, m[0], m[1], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], C.size_t(j)*C.size_t(base.nb[3]))
	}
	C.ggml_backend_view_init(dst)
	src := x.t
	if x.dt != engine.F16 {
		src = C.ggml_cast(b.rec, src, C.GGML_TYPE_F16)
	}
	b.extra = append(b.extra, C.ggml_cpy(b.rec, src, dst))
	b.extraDeps = append(b.extraDeps, x)
}

func (p *kvPool) Rows() int { return p.rows }
func (p *kvPool) Room() int { return p.room }

// Tensors are what holds the pool, for pinning.
func (p *kvPool) Tensors() []engine.Tensor { return []engine.Tensor{p.k, p.v} }

// Append writes k, v [B, Hk, 1, D] at row j's token at[j].
func (p *kvPool) Append(k, v engine.Tensor, at []int) {
	b := p.b
	tk, tv := as(k), as(v)
	B := p.rows
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	recorded := rs != nil && rs.offs != nil && len(rs.offs) == B
	if rs != nil && !recorded {
		rs.spoil("a pool append outside a batched step")
	}
	// Keys: one row a conversation, at its token.
	var kIdx *tensor
	if recorded {
		kIdx = rs.posB
	} else {
		pos := make([]int32, B)
		for j, a := range at {
			pos[j] = int32(a)
		}
		kIdx = as(b.FromInt32(pos, B))
	}
	// Values, tokens innermost: element (t, d, h, j) is
	// t + room*(d + Dv*(h + Hk*j)); a token's elements in order are its
	// Dv*Hk values of each row.
	m := p.dv * p.hk
	var vIdx *tensor
	if p.v.grp.vt {
		if recorded {
			vIdx = b.replayPoolCols(p.room, m)
		} else {
			vIdx = as(b.FromInt32(poolColsAt(at, p.room, m), B*m))
		}
	}
	b.mu.Lock()
	if b.rec == nil {
		b.newRec()
	}
	ctx := b.rec
	src := C.ggml_cont(ctx, f32(ctx, tk)) // ne (D, 1, Hk, B)
	idx := C.ggml_reshape_3d(ctx, kIdx.t, 1, 1, C.int64_t(B))
	b.extra = append(b.extra, C.ggml_set_rows(ctx, p.k.grp.base, src, idx))
	b.extraDeps = append(b.extraDeps, tk, kIdx)
	vs := C.ggml_cont(ctx, f32(ctx, tv)) // ne (Dv, 1, Hk, B)
	if p.v.grp.vt {
		base := p.v.grp.base
		dst := C.ggml_view_2d(ctx, base, 1, C.int64_t(numel(shapeOf(base))), base.nb[0], 0)
		C.ggml_backend_view_init(dst)
		b.extra = append(b.extra, C.ggml_set_rows(ctx, dst, C.ggml_reshape_2d(ctx, vs, 1, C.int64_t(B*m)), vIdx.t))
		b.extraDeps = append(b.extraDeps, tv, vIdx)
	} else {
		b.extra = append(b.extra, C.ggml_set_rows(ctx, p.v.grp.base, vs, idx))
		b.extraDeps = append(b.extraDeps, tv)
	}
	b.mu.Unlock()
}

// poolColsAt is the elements of each row's token at[j] in a transposed
// value pool of room tokens, m elements a token a row.
func poolColsAt(at []int, room, m int) []int32 {
	v := make([]int32, len(at)*m)
	for j, a := range at {
		for i := 0; i < m; i++ {
			v[j*m+i] = int32(a + room*(i+m*j))
		}
	}
	return v
}

// Attend is attention for q [B, H, 1, D], row j over its first lens[j]
// keys -> [B, H, 1, Dv]. The rows read the pool padded to one length, the
// keys past each row's masked.
func (p *kvPool) Attend(q engine.Tensor, lens []int, scale float32) engine.Tensor {
	b := p.b
	tq := as(q)
	B, H := tq.shape[0], tq.shape[1]
	longest := 0
	for _, n := range lens {
		longest = max(longest, n)
	}
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	recorded := rs != nil && rs.offs != nil && len(rs.offs) == B
	var L int
	var mask *tensor
	if recorded {
		// A recorded step reads a bucket, recorded again when a row fills it.
		L = min(p.room, (longest-1+replayBucket)/replayBucket*replayBucket)
		for j := range rs.limits {
			rs.limits[j] = min(rs.limits[j], L)
		}
		mask = b.replayPoolMask(L)
	} else {
		if rs != nil {
			rs.spoil("pool attention outside a batched step")
		}
		L = min(p.room, (longest+kvPad-1)/kvPad*kvPad)
		if padDecode() {
			// Read as a recorded step reads: padded to its bucket, masked.
			L = min(p.room, (longest-1+replayBucket)/replayBucket*replayBucket)
		}
		mask = as(b.FromFloats(poolMaskAt(lens, L), B, 1, 1, L))
	}
	kShape, vShape := []int{B, p.hk, L, p.d}, []int{B, p.hk, L, p.dv}
	return b.op(engine.F32, []int{B, H, 1, p.dv}, []*tensor{tq, p.k, p.v, mask}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		kk := cachePrefix(ctx, p.k.grp, kShape)     // ne (D, L, Hk, B)
		vv := cachePrefix(ctx, p.v.grp, vShape)     // ne (Dv, L, Hk, B)
		kq := C.ggml_mul_mat(ctx, kk, f32(ctx, tq)) // ne (L, 1, H, B)
		pr := C.ggml_soft_max_ext(ctx, kq, mask.t, C.float(scale), 0)
		vT := C.ggml_transpose(ctx, vv)
		if !C.ggml_is_transposed(vv) {
			vT = C.ggml_cont(ctx, vT)
		}
		return C.ggml_mul_mat(ctx, vT, pr) // ne (Dv, 1, H, B)
	})
}

// poolMaskAt is the [B, n] mask: row j 0 for its first lens[j] keys, -inf
// past them.
func poolMaskAt(lens []int, n int) []float32 {
	v := make([]float32, 0, len(lens)*n)
	for _, l := range lens {
		v = append(v, maskAt(l-1, n)...)
	}
	return v
}

// Row is conversation j's keys and values at n tokens [1, Hk, n, D], as
// caches of their own: a later append to them must not write into the
// pool, where another row may be.
func (p *kvPool) Row(j, n int) (engine.Tensor, engine.Tensor) {
	b := p.b
	row := func(t *tensor, D int) engine.Tensor {
		shape := []int{1, p.hk, n, D}
		return b.op(engine.F16, shape, []*tensor{t}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
			// The row is viewed on the buffer's own layout: a view takes its
			// innermost stride to be the element size, which a value cache
			// kept transposed (tokens innermost) only has before it is
			// permuted back.
			base := t.grp.base
			off := C.size_t(j) * C.size_t(base.nb[3])
			if t.grp.vt {
				v := C.ggml_view_4d(ctx, base, C.int64_t(n), C.int64_t(D), C.int64_t(p.hk), 1, base.nb[1], base.nb[2], base.nb[3], off)
				return C.ggml_cont(ctx, C.ggml_permute(ctx, v, 1, 0, 2, 3))
			}
			v := C.ggml_view_4d(ctx, base, C.int64_t(D), C.int64_t(n), C.int64_t(p.hk), 1, base.nb[1], base.nb[2], base.nb[3], off)
			return C.ggml_cont(ctx, v)
		})
	}
	if n == 0 {
		return nil, nil
	}
	return row(p.k, p.d), row(p.v, p.dv)
}
