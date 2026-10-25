//go:build ggml

// Package ggml is janis's third backend: the engine's tensor interface on
// ggml, llama.cpp's tensor library, linked from its prebuilt releases
// (scripts/fetch-ggml.sh). It is what makes a CUDA flavor possible: the same
// Go code runs on ggml's CPU backend here, and on Metal or CUDA by loading
// that device's library, because every op is a ggml graph node and ggml
// owns the kernels.
//
// This is the spike (2026-10-25): eager, one graph per op, each result in its
// own backend buffer; correct first, batched graphs later. Affine 4-bit
// weights (MLX's layout) are repacked once into ggml's Q4_1, whose block is
// the same formula (scale*q + min), so quantized matmuls run on ggml's
// quantized kernels rather than through a dequantized copy.
package ggml

/*
#cgo CFLAGS: -I${SRCDIR}/third_party/macos-arm64/include -I${SRCDIR}/third_party/ubuntu-x64/include -I${SRCDIR}/third_party/ubuntu-arm64/include
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/third_party/macos-arm64/lib -lggml -lggml-base -lggml-cpu -Wl,-rpath,${SRCDIR}/third_party/macos-arm64/lib
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/ubuntu-x64/lib -lggml -lggml-base -lggml-cpu -Wl,-rpath,${SRCDIR}/third_party/ubuntu-x64/lib
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/third_party/ubuntu-arm64/lib -lggml -lggml-base -lggml-cpu -Wl,-rpath,${SRCDIR}/third_party/ubuntu-arm64/lib
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static struct ggml_context * janis_ctx(size_t mem) {
	struct ggml_init_params p = { mem, NULL, true };
	return ggml_init(p);
}
static struct ggml_tensor * janis_new(struct ggml_context * c, enum ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	return ggml_new_tensor_4d(c, t, ne0, ne1, ne2, ne3);
}
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/guregodevo/janis/engine"
)

// Backend is one ggml backend (CPU, or the device JANIS_GGML_DEVICE names)
// behind the engine's interface. One generation at a time, as the others.
type Backend struct {
	be   C.ggml_backend_t
	buft C.ggml_backend_buffer_type_t
	name string
	mu   sync.Mutex
}

// New opens the CPU backend, or a GPU one when JANIS_GGML_DEVICE=gpu and a
// device library (libggml-metal, libggml-cuda) sits next to the CPU one.
func New() *Backend {
	b := &Backend{}
	if strings.EqualFold(os.Getenv("JANIS_GGML_DEVICE"), "gpu") {
		for _, lib := range []string{"libggml-metal.dylib", "libggml-cuda.so"} {
			p := C.CString(libDir() + "/" + lib)
			C.ggml_backend_load(p)
			C.free(unsafe.Pointer(p))
		}
		b.be = C.ggml_backend_init_by_type(C.GGML_BACKEND_DEVICE_TYPE_GPU, nil)
	}
	if b.be == nil {
		b.be = C.ggml_backend_cpu_init()
		threads := runtime.NumCPU()
		if n, err := strconv.Atoi(os.Getenv("JANIS_GGML_THREADS")); err == nil && n > 0 {
			threads = n
		}
		C.ggml_backend_cpu_set_n_threads(b.be, C.int(threads))
	}
	b.buft = C.ggml_backend_get_default_buffer_type(b.be)
	b.name = C.GoString(C.ggml_backend_name(b.be))
	return b
}

func libDir() string {
	_, file, _, _ := runtime.Caller(0)
	dir := file[:strings.LastIndex(file, "/")] + "/third_party/"
	switch runtime.GOOS + "-" + runtime.GOARCH {
	case "darwin-arm64":
		return dir + "macos-arm64/lib"
	case "linux-amd64":
		return dir + "ubuntu-x64/lib"
	default:
		return dir + "ubuntu-arm64/lib"
	}
}

func (b *Backend) Version() string { return "ggml-" + b.name }

func (b *Backend) Close() {
	if b.be != nil {
		C.ggml_backend_free(b.be)
		b.be = nil
	}
}

// ---- tensors ----

// tensor is one ggml tensor in its own buffer. shape is janis's row-major
// shape; ggml's ne is the same list reversed (ne0 innermost).
type tensor struct {
	t     *C.struct_ggml_tensor
	ctx   *C.struct_ggml_context
	buf   C.ggml_backend_buffer_t
	shape []int
	dt    engine.DType
	q4    *tensor // the Q4_1 repack of a packed weight, built once
}

func (t *tensor) Shape() []int { return append([]int(nil), t.shape...) }

func (t *tensor) free() {
	if t.buf != nil {
		C.ggml_backend_buffer_free(t.buf)
		t.buf = nil
	}
	if t.ctx != nil {
		C.ggml_free(t.ctx)
		t.ctx = nil
	}
	if t.q4 != nil {
		t.q4.free()
		t.q4 = nil
	}
}

func as(t engine.Tensor) *tensor { return t.(*tensor) }

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func ne(shape []int) [4]C.int64_t {
	out := [4]C.int64_t{1, 1, 1, 1}
	for i, d := range shape {
		out[len(shape)-1-i] = C.int64_t(d)
	}
	return out
}

func ggmlType(dt engine.DType) C.enum_ggml_type {
	switch dt {
	case engine.F16:
		return C.GGML_TYPE_F16
	case engine.BF16:
		return C.GGML_TYPE_BF16
	case engine.I32, engine.U32:
		return C.GGML_TYPE_I32
	default:
		return C.GGML_TYPE_F32
	}
}

const ctxMem = 64 * 1024 // tensors and graph metadata, no data (no_alloc)

// newFrom allocates a tensor of dt/shape in its own buffer and fills it.
func (b *Backend) newFrom(dt engine.DType, shape []int, raw unsafe.Pointer, size int) *tensor {
	if len(shape) > 4 {
		panic(fmt.Sprintf("ggml: %d-d tensors are not supported", len(shape)))
	}
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, ggmlType(dt), n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if size > 0 {
		C.ggml_backend_tensor_set(t, raw, 0, C.size_t(size))
	}
	out := &tensor{t: t, ctx: ctx, buf: buf, shape: append([]int(nil), shape...), dt: dt}
	runtime.SetFinalizer(out, func(x *tensor) { x.free() })
	return out
}

func (b *Backend) FromFloats(data []float32, shape ...int) engine.Tensor {
	if len(data) == 0 {
		return b.newFrom(engine.F32, shape, nil, 0)
	}
	return b.newFrom(engine.F32, shape, unsafe.Pointer(&data[0]), len(data)*4)
}

func (b *Backend) FromInt32(data []int32, shape ...int) engine.Tensor {
	if len(data) == 0 {
		return b.newFrom(engine.I32, shape, nil, 0)
	}
	return b.newFrom(engine.I32, shape, unsafe.Pointer(&data[0]), len(data)*4)
}

func (b *Backend) FromRaw(dt engine.DType, raw []byte, shape ...int) engine.Tensor {
	if len(raw) == 0 {
		return b.newFrom(dt, shape, nil, 0)
	}
	return b.newFrom(dt, shape, unsafe.Pointer(&raw[0]), len(raw))
}

// ---- eager graph execution ----

// op builds a graph around the node fn makes and computes it; the node's
// data lives in the returned tensor's own buffer. The node must not be a
// view (wrap views in cont).
//
// The inputs are kept alive until the compute is done: a temporary only the
// closure refers to would otherwise be collected, and its finalizer free the
// buffer the graph is about to read (the first full forward crashed so).
func (b *Backend) op(dt engine.DType, shape []int, inputs []*tensor, fn func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor) *tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	if trace {
		_, file, line, _ := runtime.Caller(1)
		var ins []string
		for _, in := range inputs {
			if in != nil {
				ins = append(ins, fmt.Sprintf("%v/%d", in.shape, in.dt))
			}
		}
		fmt.Fprintf(os.Stderr, "ggml op %s:%d %s -> %v\n", file[strings.LastIndex(file, "/")+1:], line, strings.Join(ins, " "), shape)
	}
	ctx := C.janis_ctx(ctxMem + C.size_t(C.ggml_graph_overhead()))
	node := fn(ctx)
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	g := C.ggml_new_graph_custom(ctx, 256, false)
	C.ggml_build_forward_expand(g, node)
	st := C.ggml_backend_graph_compute(b.be, g)
	for _, in := range inputs {
		runtime.KeepAlive(in)
	}
	if st != C.GGML_STATUS_SUCCESS {
		panic(fmt.Sprintf("ggml: compute failed (%d)", int(st)))
	}
	// Detach: a computed node still points at its sources, and a later
	// graph built on it would walk (and recompute) the whole upstream
	// history across every earlier context. With its data in place it is a
	// leaf from here on.
	node.op = C.GGML_OP_NONE
	for i := range node.src {
		node.src[i] = nil
	}
	node.view_src = nil
	node.flags = 0
	out := &tensor{t: node, ctx: ctx, buf: buf, shape: append([]int(nil), shape...), dt: dt}
	runtime.SetFinalizer(out, func(x *tensor) { x.free() })
	return out
}

// trace (JANIS_GGML_TRACE=1) prints every op before it runs: the last line
// names the one a native crash happened in.
var trace = os.Getenv("JANIS_GGML_TRACE") == "1"

// f32 is x as an F32 node in ctx (a cast when it is not).
func f32(ctx *C.struct_ggml_context, x *tensor) *C.struct_ggml_tensor {
	if x.dt == engine.F32 {
		return x.t
	}
	return C.ggml_cast(ctx, x.t, C.GGML_TYPE_F32)
}

func (b *Backend) Eval(ts ...engine.Tensor) {}

func (b *Backend) Floats(t engine.Tensor) []float32 {
	x := as(t)
	n := numel(x.shape)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	switch x.dt {
	case engine.F32:
		C.ggml_backend_tensor_get(x.t, unsafe.Pointer(&out[0]), 0, C.size_t(n*4))
	case engine.I32, engine.U32:
		for i, v := range b.Ints(t) {
			out[i] = float32(v)
		}
	default:
		c := b.op(engine.F32, x.shape, []*tensor{x}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor { return f32(ctx, x) })
		C.ggml_backend_tensor_get(c.t, unsafe.Pointer(&out[0]), 0, C.size_t(n*4))
	}
	return out
}

func (b *Backend) Ints(t engine.Tensor) []int32 {
	x := as(t)
	n := numel(x.shape)
	out := make([]int32, n)
	if n == 0 {
		return out
	}
	if x.dt == engine.I32 || x.dt == engine.U32 {
		C.ggml_backend_tensor_get(x.t, unsafe.Pointer(&out[0]), 0, C.size_t(n*4))
		return out
	}
	for i, v := range b.Floats(t) {
		out[i] = int32(v)
	}
	return out
}

// Bytes and DTypeOf let the loader read a tensor back raw (engine.RawReader).
func (b *Backend) DTypeOf(t engine.Tensor) engine.DType { return as(t).dt }
func (b *Backend) Bytes(t engine.Tensor) []byte {
	x := as(t)
	n := int(C.ggml_nbytes(x.t))
	out := make([]byte, n)
	if n > 0 {
		C.ggml_backend_tensor_get(x.t, unsafe.Pointer(&out[0]), 0, C.size_t(n))
	}
	return out
}

// Free releases tensors early (engine.Freer); the finalizer covers the rest.
func (b *Backend) Free(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			as(t).free()
		}
	}
}

// ---- elementwise ----

func broadcastShape(a, bb []int) []int {
	n := max(len(a), len(bb))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if i < len(a) {
			da = a[len(a)-1-i]
		}
		if i < len(bb) {
			db = bb[len(bb)-1-i]
		}
		out[n-1-i] = max(da, db)
	}
	return out
}

// binary applies fn(a, b) with ggml's broadcasting (b repeats into a): the
// operand with the larger shape goes first.
func (b *Backend) binary(a, bb engine.Tensor, fn func(ctx *C.struct_ggml_context, x, y *C.struct_ggml_tensor) *C.struct_ggml_tensor, commutative bool) engine.Tensor {
	ta, tb := as(a), as(bb)
	shape := broadcastShape(ta.shape, tb.shape)
	return b.op(engine.F32, shape, []*tensor{ta, tb}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		x, y := f32(ctx, ta), f32(ctx, tb)
		if numel(ta.shape) < numel(shape) {
			if numel(tb.shape) == numel(shape) && commutative {
				x, y = y, x
			} else {
				// Neither side has the full shape: repeat the first up to it.
				n := ne(shape)
				x = C.ggml_repeat(ctx, x, C.janis_new(ctx, C.GGML_TYPE_F32, n[0], n[1], n[2], n[3]))
			}
		}
		return fn(ctx, x, y)
	})
}

func (b *Backend) Add(a, bb engine.Tensor) engine.Tensor {
	return b.binary(a, bb, func(ctx *C.struct_ggml_context, x, y *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_add(ctx, x, y)
	}, true)
}

func (b *Backend) Mul(a, bb engine.Tensor) engine.Tensor {
	return b.binary(a, bb, func(ctx *C.struct_ggml_context, x, y *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_mul(ctx, x, y)
	}, true)
}

func (b *Backend) unary(x engine.Tensor, fn func(ctx *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor) engine.Tensor {
	tx := as(x)
	return b.op(engine.F32, tx.shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor { return fn(ctx, f32(ctx, tx)) })
}

func (b *Backend) SiLU(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_silu(c, t)
	})
}
func (b *Backend) Gelu(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_gelu(c, t)
	})
}
func (b *Backend) Tanh(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_tanh(c, t)
	})
}
func (b *Backend) Exp(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor { return C.ggml_exp(c, t) })
}
func (b *Backend) Sigmoid(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_sigmoid(c, t)
	})
}
func (b *Backend) Softplus(x engine.Tensor) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_softplus(c, t)
	})
}
func (b *Backend) Erf(x engine.Tensor) engine.Tensor {
	// No erf node in this ggml; a host pass (the models do not use it).
	v := b.Floats(x)
	for i := range v {
		v[i] = float32(math.Erf(float64(v[i])))
	}
	return b.FromFloats(v, as(x).shape...)
}
func (b *Backend) ScalarMul(x engine.Tensor, v float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_scale(c, t, C.float(v))
	})
}
func (b *Backend) ScalarAdd(x engine.Tensor, v float32) engine.Tensor {
	return b.unary(x, func(c *C.struct_ggml_context, t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
		return C.ggml_scale_bias(c, t, 1, C.float(v))
	})
}

func (b *Backend) Cast(x engine.Tensor, dt engine.DType) engine.Tensor {
	tx := as(x)
	if tx.dt == dt {
		return x
	}
	if dt == engine.I32 || dt == engine.U32 {
		v := b.Floats(x)
		out := make([]int32, len(v))
		for i := range v {
			out[i] = int32(math.Round(float64(v[i])))
		}
		return b.FromInt32(out, tx.shape...)
	}
	return b.op(dt, tx.shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_cast(ctx, f32(ctx, tx), ggmlType(dt))
	})
}

// ---- shape ----

func (b *Backend) Reshape(x engine.Tensor, shape ...int) engine.Tensor {
	tx := as(x)
	n := numel(tx.shape)
	for i, d := range shape {
		if d == -1 {
			known := 1
			for j, e := range shape {
				if j != i {
					known *= e
				}
			}
			shape[i] = n / known
		}
	}
	return b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		m := ne(shape)
		return C.ggml_cont(ctx, C.ggml_reshape_4d(ctx, tx.t, m[0], m[1], m[2], m[3]))
	})
}

func (b *Backend) ExpandDims(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	if axis < 0 {
		axis += len(tx.shape) + 1
	}
	shape := append(append(append([]int(nil), tx.shape[:axis]...), 1), tx.shape[axis:]...)
	return b.Reshape(x, shape...)
}

// Transpose permutes axes by the given order (numpy's): out dim j is in dim
// axes[j]. ggml's permute says where each source dim goes, on reversed axes.
func (b *Backend) Transpose(x engine.Tensor, axes ...int) engine.Tensor {
	tx := as(x)
	nd := len(tx.shape)
	shape := make([]int, nd)
	for j, a := range axes {
		shape[j] = tx.shape[a]
	}
	// ggml axis of janis axis i is nd-1-i. Source ggml axis s = nd-1-axes[j]
	// goes to destination ggml axis nd-1-j.
	dst := [4]C.int{0, 1, 2, 3}
	for j, a := range axes {
		dst[nd-1-a] = C.int(nd - 1 - j)
	}
	return b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_cont(ctx, C.ggml_permute(ctx, tx.t, dst[0], dst[1], dst[2], dst[3]))
	})
}

func (b *Backend) Concat(a, bb engine.Tensor, axis int) engine.Tensor {
	ta, tb := as(a), as(bb)
	nd := len(ta.shape)
	if axis < 0 {
		axis += nd
	}
	shape := append([]int(nil), ta.shape...)
	shape[axis] += tb.shape[axis]
	return b.op(engine.F32, shape, []*tensor{ta, tb}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_concat(ctx, f32(ctx, ta), f32(ctx, tb), C.int(nd-1-axis))
	})
}

func (b *Backend) Slice(x engine.Tensor, axis, start, end int) engine.Tensor {
	tx := as(x)
	nd := len(tx.shape)
	if axis < 0 {
		axis += nd
	}
	if end < 0 {
		end += tx.shape[axis] + 1
	}
	if end > tx.shape[axis] {
		end = tx.shape[axis]
	}
	shape := append([]int(nil), tx.shape...)
	shape[axis] = end - start
	g := nd - 1 - axis
	return b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		m := ne(shape)
		t := tx.t
		offset := C.size_t(start) * C.size_t(t.nb[g])
		return C.ggml_cont(ctx, C.ggml_view_4d(ctx, t, m[0], m[1], m[2], m[3], t.nb[1], t.nb[2], t.nb[3], offset))
	})
}

// TakeAxis gathers along axis by integer indices; axis 0 on a matrix is the
// embedding lookup (ggml_get_rows). Other axes go through a transpose.
func (b *Backend) TakeAxis(a, indices engine.Tensor, axis int) engine.Tensor {
	ta, ti := as(a), as(indices)
	nd := len(ta.shape)
	if axis < 0 {
		axis += nd
	}
	if axis != 0 {
		order := make([]int, nd)
		order[0] = axis
		k := 1
		for i := 0; i < nd; i++ {
			if i != axis {
				order[k] = i
				k++
			}
		}
		g := b.TakeAxis(b.Transpose(a, order...), indices, 0)
		inv := make([]int, nd)
		for i, o := range order {
			inv[o] = i
		}
		return b.Transpose(g, inv...)
	}
	n := numel(ti.shape)
	shape := append([]int{n}, ta.shape[1:]...)
	rows := numel(ta.shape[1:])
	return b.op(ta.dt, shape, []*tensor{ta, ti}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		src := C.ggml_reshape_2d(ctx, ta.t, C.int64_t(rows), C.int64_t(ta.shape[0]))
		idx := C.ggml_reshape_1d(ctx, ti.t, C.int64_t(n))
		got := C.ggml_get_rows(ctx, src, idx) // F32 out for float types; same type for I32
		m := ne(shape)
		out := C.ggml_reshape_4d(ctx, got, m[0], m[1], m[2], m[3])
		if ta.dt != engine.F32 && ta.dt != engine.I32 && ta.dt != engine.U32 {
			return C.ggml_cast(ctx, out, ggmlType(ta.dt))
		}
		return C.ggml_cont(ctx, out)
	})
}

// ---- reductions and norms ----

func (b *Backend) Softmax(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	nd := len(tx.shape)
	if axis < 0 {
		axis += nd
	}
	if axis != nd-1 {
		order := make([]int, nd)
		for i := range order {
			order[i] = i
		}
		order[axis], order[nd-1] = order[nd-1], order[axis]
		return b.Transpose(b.Softmax(b.Transpose(x, order...), -1), order...)
	}
	return b.op(engine.F32, tx.shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor { return C.ggml_soft_max(ctx, f32(ctx, tx)) })
}

func (b *Backend) Mean(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	nd := len(tx.shape)
	if axis < 0 {
		axis += nd
	}
	if axis != nd-1 {
		order := make([]int, nd)
		for i := range order {
			order[i] = i
		}
		order[axis], order[nd-1] = order[nd-1], order[axis]
		m := b.Mean(b.Transpose(x, order...), -1)
		// Put the remaining axes back in their order.
		rest := append(append([]int(nil), tx.shape[:axis]...), tx.shape[axis+1:]...)
		return b.Reshape(m, rest...)
	}
	shape := append([]int(nil), tx.shape[:nd-1]...)
	return b.op(engine.F32, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		m := ne(shape)
		return C.ggml_cont(ctx, C.ggml_reshape_4d(ctx, C.ggml_mean(ctx, f32(ctx, tx)), m[0], m[1], m[2], m[3]))
	})
}

func (b *Backend) RMSNorm(x, weight engine.Tensor, eps float32) engine.Tensor {
	tx, tw := as(x), as(weight)
	return b.op(engine.F32, tx.shape, []*tensor{tx, tw}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_mul(ctx, C.ggml_rms_norm(ctx, f32(ctx, tx), C.float(eps)), f32(ctx, tw))
	})
}

func (b *Backend) LayerNorm(x, weight, bias engine.Tensor, eps float32) engine.Tensor {
	tx, tw, tb := as(x), as(weight), as(bias)
	return b.op(engine.F32, tx.shape, []*tensor{tx, tw, tb}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_add(ctx, C.ggml_mul(ctx, C.ggml_norm(ctx, f32(ctx, tx), C.float(eps)), f32(ctx, tw)), f32(ctx, tb))
	})
}

func (b *Backend) Argmax(x engine.Tensor, axis int) engine.Tensor {
	tx := as(x)
	nd := len(tx.shape)
	if axis < 0 {
		axis += nd
	}
	if axis != nd-1 {
		order := make([]int, nd)
		for i := range order {
			order[i] = i
		}
		order[axis], order[nd-1] = order[nd-1], order[axis]
		return b.Argmax(b.Transpose(x, order...), -1)
	}
	shape := append([]int(nil), tx.shape[:nd-1]...)
	return b.op(engine.I32, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		m := ne(shape)
		return C.ggml_cont(ctx, C.ggml_reshape_4d(ctx, C.ggml_argmax(ctx, f32(ctx, tx)), m[0], m[1], m[2], m[3]))
	})
}

// ---- matmul and attention ----

// MatMul: (…,M,K) x (…,K,N) -> (…,M,N). ggml_mul_mat(A, B) is B·Aᵀ with A
// [k,n] and B [k,m] in ne order, so A is b transposed.
func (b *Backend) MatMul(a, bb engine.Tensor) engine.Tensor {
	ta, tb := as(a), as(bb)
	na, nb := len(ta.shape), len(tb.shape)
	shape := append([]int(nil), broadcastShape(ta.shape[:na-2], tb.shape[:nb-2])...)
	shape = append(shape, ta.shape[na-2], tb.shape[nb-1])
	return b.op(engine.F32, shape, []*tensor{ta, tb}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		wT := C.ggml_cont(ctx, C.ggml_transpose(ctx, f32(ctx, tb)))
		return C.ggml_mul_mat(ctx, wT, f32(ctx, ta))
	})
}

// RoPE on [B, H, L, D] with positions offset..offset+L along L. ggml wants
// ne = (D, H, L, B) with one position per ne2, so the tensor is permuted in
// and out. NeoX is MLX's non-traditional mode.
func (b *Backend) RoPE(x engine.Tensor, dims int, traditional bool, base, scale float32, offset int) engine.Tensor {
	return b.rope(as(x), dims, traditional, base, scale, offset, nil)
}

// RoPEFreqs: theta_i = pos*scale/freqs[i]. With freq_base 1 ggml's own
// factor is 1 and its freq_factors divide, so freqs is passed as is.
func (b *Backend) RoPEFreqs(x engine.Tensor, dims int, traditional bool, scale float32, offset int, freqs engine.Tensor) engine.Tensor {
	return b.rope(as(x), dims, traditional, 1, scale, offset, as(freqs))
}

func (b *Backend) rope(tx *tensor, dims int, traditional bool, base, scale float32, offset int, freqs *tensor) engine.Tensor {
	nd := len(tx.shape)
	L := tx.shape[nd-2]
	pos := make([]int32, L)
	for i := range pos {
		pos[i] = int32(offset + i)
	}
	tpos := as(b.FromInt32(pos, L))
	mode := C.int(C.GGML_ROPE_TYPE_NEOX)
	if traditional {
		mode = C.GGML_ROPE_TYPE_NORMAL
	}
	return b.op(engine.F32, tx.shape, []*tensor{tx, tpos, freqs}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		x := f32(ctx, tx)
		// janis [B,H,L,D] is ne (D,L,H,B); rope wants (D,H,L,B).
		x = C.ggml_cont(ctx, C.ggml_permute(ctx, x, 0, 2, 1, 3))
		var fr *C.struct_ggml_tensor
		if freqs != nil {
			fr = f32(ctx, freqs)
		}
		r := C.ggml_rope_ext(ctx, x, tpos.t, fr, C.int(dims), mode, 0, C.float(base), C.float(scale), 0, 1, 32, 1)
		return C.ggml_cont(ctx, C.ggml_permute(ctx, r, 0, 2, 1, 3))
	})
}

// SDPA over [B, H, L, D] with GQA and the causal mask offset by Lk-Lq: the
// classic three ops, scores = k·q, softmax with the mask and scale, then
// v·scores, with ggml's head broadcasting doing the GQA share.
func (b *Backend) SDPA(q, k, v engine.Tensor, scale float32, causal bool) engine.Tensor {
	tq, tk, tv := as(q), as(k), as(v)
	B, H, Lq := tq.shape[0], tq.shape[1], tq.shape[2]
	Lk := tk.shape[2]
	Dv := tv.shape[3]
	var tmask *tensor
	if causal {
		m := make([]float32, Lk*Lq)
		for i := 0; i < Lq; i++ {
			for j := 0; j < Lk; j++ {
				if j > i+(Lk-Lq) {
					m[i*Lk+j] = float32(math.Inf(-1))
				}
			}
		}
		tmask = as(b.FromFloats(m, Lq, Lk))
	}
	shape := []int{B, H, Lq, Dv}
	return b.op(engine.F32, shape, []*tensor{tq, tk, tv, tmask}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		qq, kk, vv := f32(ctx, tq), f32(ctx, tk), f32(ctx, tv)
		kq := C.ggml_mul_mat(ctx, kk, qq) // ne (Lk, Lq, H, B)
		var mask *C.struct_ggml_tensor
		if tmask != nil {
			mask = tmask.t
		}
		p := C.ggml_soft_max_ext(ctx, kq, mask, C.float(scale), 0)
		vT := C.ggml_cont(ctx, C.ggml_transpose(ctx, vv)) // ne (Lk, Dv, Hk, B)
		return C.ggml_mul_mat(ctx, vT, p)                 // ne (Dv, Lq, H, B)
	})
}

// ---- affine 4-bit weights ----

// Dequantize reconstructs the float weight from MLX's (packed words,
// scales, biases): value = q*scale + bias per group. Host-side, as the CPU
// backend does; the matmul path never needs it.
func (b *Backend) Dequantize(w, scales, biases engine.Tensor, groupSize, bits int) engine.Tensor {
	tw := as(w)
	words := b.Ints(w)
	sc, bi := b.Floats(scales), b.Floats(biases)
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	G := as(scales).shape[len(as(scales).shape)-1]
	rows := numel(tw.shape) / P
	out := make([]float32, rows*in)
	for r := 0; r < rows; r++ {
		for j := 0; j < in; j++ {
			q := (uint32(words[r*P+j/valsPerWord]) >> uint((j%valsPerWord)*bits)) & mask
			g := j / groupSize
			out[r*in+j] = float32(q)*sc[r*G+g] + bi[r*G+g]
		}
	}
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	return b.FromFloats(out, shape...)
}

// q4_1 repacks a 4-bit, group-64-or-32 affine weight [out, in] into ggml's
// Q4_1 (blocks of 32: f16 scale d, f16 min m, 16 bytes of nibbles, element
// j in the low nibble of byte j for j<16 and the high nibble of byte j-16
// after), which is the same formula: d*q + m. Built once per weight.
func (b *Backend) q4_1(tw, ts, tb *tensor, groupSize, bits int) *tensor {
	if tw.q4 != nil {
		return tw.q4
	}
	if bits != 4 || groupSize%32 != 0 {
		panic(fmt.Sprintf("ggml: %d-bit group %d weights are not repackable to Q4_1", bits, groupSize))
	}
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	P := tw.shape[len(tw.shape)-1]
	in := P * 8
	G := ts.shape[len(ts.shape)-1]
	rows := numel(tw.shape) / P
	const blk = 20 // sizeof(block_q4_1): 2+2+16
	raw := make([]byte, rows*(in/32)*blk)
	for r := 0; r < rows; r++ {
		for blkIdx := 0; blkIdx < in/32; blkIdx++ {
			g := (blkIdx * 32) / groupSize
			o := (r*(in/32) + blkIdx) * blk
			binary.LittleEndian.PutUint16(raw[o:], f32tof16(sc[r*G+g]))
			binary.LittleEndian.PutUint16(raw[o+2:], f32tof16(bi[r*G+g]))
			for j := 0; j < 32; j++ {
				e := blkIdx*32 + j
				q := byte((uint32(words[r*P+e/8]) >> uint((e%8)*4)) & 0xF)
				if j < 16 {
					raw[o+4+j] |= q
				} else {
					raw[o+4+j-16] |= q << 4
				}
			}
		}
	}
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q4_1, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	q := &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16}
	tw.q4 = q
	return q
}

// QuantMatmul: x @ Wᵀ with W [out, in] affine 4-bit, on ggml's Q4_1 kernel.
func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	if !transpose {
		return b.MatMul(x, b.Dequantize(w, scales, biases, groupSize, bits))
	}
	tx := as(x)
	q := b.q4_1(as(w), as(scales), as(biases), groupSize, bits)
	nx := len(tx.shape)
	shape := append(append([]int(nil), tx.shape[:nx-1]...), q.shape[0])
	return b.op(engine.F32, shape, []*tensor{tx, q, as(w)}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_mul_mat(ctx, q.t, f32(ctx, tx)) // ne (out, M, …)
	})
}

// GatherQuantMatmul (MoE) is not in the spike: one QuantMatmul per expert.
func (b *Backend) GatherQuantMatmul(x, w, scales, biases, rhsIndices engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	panic("ggml: GatherQuantMatmul (MoE experts) is not implemented in the spike")
}

// Conv1dDepthwise and GatedDeltaScan belong to the Qwen3.5 hybrid; not in
// the spike (Qwen3 dense first).
func (b *Backend) Conv1dDepthwise(x, w engine.Tensor) engine.Tensor {
	panic("ggml: Conv1dDepthwise is not implemented in the spike")
}
func (b *Backend) GatedDeltaScan(q, k, v, g, beta, state engine.Tensor) (engine.Tensor, engine.Tensor) {
	panic("ggml: GatedDeltaScan is not implemented in the spike")
}

func f32tof16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := uint16(mant >> shift)
		if (mant>>(shift-1))&1 == 1 {
			h++
		}
		return sign | h
	case exp >= 31:
		return sign | 0x7C00
	}
	h := sign | uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 { // round to nearest
		h++
	}
	return h
}

var _ engine.Backend = (*Backend)(nil)
