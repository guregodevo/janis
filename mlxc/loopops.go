//go:build darwin && arm64 && cgo

package mlxc

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
#include "compat.h"
*/
import "C"

import (
	"unsafe"

	"github.com/guregodevo/janis/engine"
)

// The operations of engine.LoopOps.

func (b *Backend) Sqrt(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_sqrt(&out, tx.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Div(x, y engine.Tensor) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	out := C.mlx_array_new()
	C.mlx_divide(&out, tx.arr, ty.arr, b.stream)
	return b.newT(out, broadcastShapes(tx.shape, ty.shape))
}

func (b *Backend) Minimum(x, y engine.Tensor) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	out := C.mlx_array_new()
	C.mlx_minimum(&out, tx.arr, ty.arr, b.stream)
	return b.newT(out, broadcastShapes(tx.shape, ty.shape))
}

// TopK partitions the negated values, as mlx-lm's routers do: the first k
// places of the partition hold the k largest.
func (b *Backend) TopK(x engine.Tensor, k int) engine.Tensor {
	tx := x.(*tensor)
	last := len(tx.shape) - 1
	neg := C.mlx_array_new()
	C.mlx_negative(&neg, tx.arr, b.stream)
	part := C.mlx_array_new()
	C.mlx_argpartition_axis(&part, neg, C.int(k-1), C.int(last), b.stream)
	C.mlx_array_free(neg)
	all := b.newT(part, cloneShape(tx.shape))
	return b.Slice(all, last, 0, k)
}

func (b *Backend) TakeAlong(a, indices engine.Tensor, axis int) engine.Tensor {
	ta, ti := a.(*tensor), indices.(*tensor)
	shape := broadcastShapes(ta.shape, ti.shape)
	shape[axis] = ti.shape[axis]
	// MLX wants both at their full shape off the axis.
	src := cloneShape(shape)
	src[axis] = ta.shape[axis]
	cs, _ := cShape(src)
	wide := C.mlx_array_new()
	C.mlx_broadcast_to(&wide, ta.arr, &cs[0], C.size_t(len(cs)), b.stream)
	ci, _ := cShape(shape)
	idx := C.mlx_array_new()
	C.mlx_broadcast_to(&idx, ti.arr, &ci[0], C.size_t(len(ci)), b.stream)
	out := C.mlx_array_new()
	C.mlx_take_along_axis(&out, wide, idx, C.int(axis), b.stream)
	C.mlx_array_free(wide)
	C.mlx_array_free(idx)
	return b.newT(out, shape)
}

func (b *Backend) GatherModeMatmul(x, w, scales, rhsIndices engine.Tensor, groupSize, bits int, mode string) engine.Tensor {
	tx, tw, ts, tr := x.(*tensor), w.(*tensor), scales.(*tensor), rhsIndices.(*tensor)
	out := C.mlx_array_new()
	cmode := C.CString(mode)
	defer C.free(unsafe.Pointer(cmode))
	var none C.mlx_array // no zero points, no left gather
	C.janis_gather_qmm(&out, tx.arr, tw.arr, ts.arr, none, none, tr.arr,
		C.bool(true), optInt(groupSize), optInt(bits), cmode, C.bool(false), b.stream)
	return b.newT(out, append(cloneShape(tr.shape), 1, tw.shape[1]))
}

var _ engine.LoopOps = (*Backend)(nil)
