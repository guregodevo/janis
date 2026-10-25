//go:build ggml

// Package ggml is janis's third backend: the engine's tensor interface on
// ggml, llama.cpp's tensor library, linked from its prebuilt releases
// (scripts/fetch-ggml.sh). It is what makes a CUDA flavor possible: the same
// Go code runs on ggml's CPU backend here, and on Metal or CUDA by loading
// that device's library, because every op is a ggml graph node and ggml
// owns the kernels.
//
// Ops record nodes into a shared context and nothing runs until something
// needs a value: Eval, a readback, or a full context flushes the recorded
// nodes as ONE graph into ONE buffer that the batch shares (refcounted, so a
// tensor keeps its batch alive and no more), and every computed node is
// then detached from its sources so later graphs see a leaf with data. The
// engine's Pin/Unpin/Sweep protocol (engine.Sweeper, what MLX uses) frees
// a step's temporaries deterministically; weak references let the garbage
// collector cover code paths that never sweep. Affine 4-bit weights (MLX's
// layout) are repacked once into ggml's Q4_1, whose block is the same
// formula (scale*q + min), so quantized matmuls run on ggml's quantized
// kernels rather than through a dequantized copy.
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
	"time"
	"unsafe"
	"weak"

	"github.com/guregodevo/janis/engine"
)

// Backend is one ggml backend (CPU, or the device JANIS_GGML_DEVICE names)
// behind the engine's interface. One generation at a time, as the others.
type Backend struct {
	be   C.ggml_backend_t
	buft C.ggml_backend_buffer_type_t
	name string
	mu   sync.Mutex

	rec     *C.struct_ggml_context // the recording context: nodes not yet computed
	pending []*tensor              // their result tensors, in order
	// tracked is every tensor made since Open, weakly: Sweep frees the ones
	// that are alive and not pinned; a collected one needs nothing.
	tracked map[weak.Pointer[tensor]]struct{}
	// Small constants every layer re-asks for: the RoPE positions of a step
	// and the causal mask of a shape. Uploaded once, pinned.
	positions map[[2]int]*tensor
	masks     map[[2]int]*tensor
}

// A flush records at most this many results before computing: a decode
// step of a 28-layer model is one or two graphs instead of hundreds
// (JANIS_GGML_FLUSH overrides, for measuring).
var flushEvery = func() int {
	if n, err := strconv.Atoi(os.Getenv("JANIS_GGML_FLUSH")); err == nil && n > 0 {
		return n
	}
	return 512
}()

// maxNodes bounds one recording context (results and intermediates).
const maxNodes = 8192

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
	b.tracked = map[weak.Pointer[tensor]]struct{}{}
	b.positions = map[[2]int]*tensor{}
	b.masks = map[[2]int]*tensor{}
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
	b.mu.Lock()
	defer b.mu.Unlock()
	b.flushLocked()
	for wp := range b.tracked {
		if t := wp.Value(); t != nil {
			t.free()
		}
	}
	b.tracked = map[weak.Pointer[tensor]]struct{}{}
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
	ctx   *C.struct_ggml_context  // a leaf's own context and buffer …
	buf   C.ggml_backend_buffer_t //
	grp   *batch                  // … or the batch a computed result lives in
	deps  []*tensor               // the inputs a recorded node reads, held until it is computed
	shape []int
	dt    engine.DType
	where string  // the op's call site, kept when tracing or checking
	q4    *tensor // the Q4_1 repack of a packed weight, built once
	pin   bool    // survives Sweep
	freed bool
	// gathered: rows of this packed weight were taken (the embedding), so
	// its words stay after the repack; a weight only ever multiplied keeps
	// the Q4_1 copy alone.
	gathered bool
	dropped  bool // the words are gone: only QuantMatmul may use it now
}

// batch is one flushed graph's context and buffer, shared by its results and
// released when the last of them goes.
type batch struct {
	ctx  *C.struct_ggml_context
	buf  C.ggml_backend_buffer_t
	refs int
}

func (g *batch) release() {
	g.refs--
	if g.refs > 0 {
		return
	}
	if g.buf != nil {
		C.ggml_backend_buffer_free(g.buf)
	}
	if g.ctx != nil {
		C.ggml_free(g.ctx)
	}
	g.buf, g.ctx = nil, nil
}

func (t *tensor) Shape() []int { return append([]int(nil), t.shape...) }

// free releases what the tensor holds; a recorded, never computed node has
// nothing of its own yet and is left to its flush. Idempotent.
func (t *tensor) free() {
	if t.freed || t.grp == nil && t.ctx == nil && t.buf == nil && t.q4 == nil {
		return
	}
	t.freed = true
	if t.grp != nil {
		t.grp.release()
		t.grp = nil
	}
	if t.dropped {
		t.t = nil
	}
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
	t.deps = nil
}

func as(t engine.Tensor) *tensor { return t.(*tensor) }

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// ne is ggml's dimension list for a janis shape: the same list reversed
// (ne0 innermost). ggml has four dimensions; a janis shape with more (the
// MoE gather's [lead…, K, 1, out]) folds its leading dims into ne3, which
// is right for every elementwise op and reshape, the only ops such tensors
// meet.
func ne(shape []int) [4]C.int64_t {
	out := [4]C.int64_t{1, 1, 1, 1}
	if len(shape) > 4 {
		lead := 1
		for _, d := range shape[:len(shape)-3] {
			lead *= d
		}
		shape = append([]int{lead}, shape[len(shape)-3:]...)
	}
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
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, ggmlType(dt), n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if size > 0 {
		C.ggml_backend_tensor_set(t, raw, 0, C.size_t(size))
	}
	out := &tensor{t: t, ctx: ctx, buf: buf, shape: append([]int(nil), shape...), dt: dt}
	b.track(out)
	return out
}

// track registers a tensor for Sweep (weakly) and for the collector.
func (b *Backend) track(t *tensor) {
	runtime.SetFinalizer(t, func(x *tensor) { x.free() })
	b.mu.Lock()
	b.tracked[weak.Make(t)] = struct{}{}
	b.mu.Unlock()
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

// op records the node fn makes in the recording context; nothing runs until
// a flush. The inputs are held by the result until then, so a temporary only
// the closure referred to stays alive for the compute. The node must not be
// a view (wrap views in cont).
func (b *Backend) op(dt engine.DType, shape []int, inputs []*tensor, fn func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor) *tensor {
	b.mu.Lock()
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
	if b.rec == nil {
		b.rec = C.janis_ctx(C.size_t(C.ggml_tensor_overhead())*maxNodes + C.ggml_graph_overhead_custom(maxNodes, false))
	}
	node := fn(b.rec)
	out := &tensor{t: node, shape: append([]int(nil), shape...), dt: dt}
	if check {
		_, file, line, _ := runtime.Caller(1)
		out.where = fmt.Sprintf("%s:%d", file[strings.LastIndex(file, "/")+1:], line)
	}
	for _, in := range inputs {
		if in != nil {
			out.deps = append(out.deps, in)
		}
	}
	b.pending = append(b.pending, out)
	full := len(b.pending) >= flushEvery
	if full {
		b.flushLocked()
	}
	b.mu.Unlock()
	b.track(out)
	return out
}

// flushLocked computes every recorded node as one graph into one buffer the
// batch shares, then detaches the results: a computed node still pointing at
// its sources would make a later graph walk (and recompute) the whole
// history across every earlier context. Called with b.mu held.
func (b *Backend) flushLocked() {
	if len(b.pending) == 0 {
		return
	}
	ctx, pending := b.rec, b.pending
	b.rec, b.pending = nil, nil
	t0 := time.Now()
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	t1 := time.Now()
	g := C.ggml_new_graph_custom(ctx, maxNodes, false)
	for _, p := range pending {
		C.ggml_build_forward_expand(g, p.t)
	}
	t2 := time.Now()
	st := C.ggml_backend_graph_compute(b.be, g)
	if check {
		for _, p := range pending {
			if p.dt != engine.F32 {
				continue
			}
			n := numel(p.shape)
			vals := make([]float32, n)
			if n > 0 {
				C.ggml_backend_tensor_get(p.t, unsafe.Pointer(&vals[0]), 0, C.size_t(n*4))
			}
			bad := 0
			for _, v := range vals {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					bad++
				}
			}
			if bad > 0 {
				fmt.Fprintf(os.Stderr, "ggml check: %d of %d values NaN/Inf in a result of %s %v\n", bad, n, p.where, p.shape)
				var at []int
				for i, v := range vals {
					if (math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)) && len(at) < 12 {
						at = append(at, i)
					}
				}
				fmt.Fprintf(os.Stderr, "ggml check:   first bad indices %v (node type %d, op %d, contiguous %v)\n", at, int(p.t._type), int(p.t.op), bool(C.ggml_is_contiguous(p.t)))
				for _, d := range p.deps {
					if d != nil && d.dt == engine.F32 && d.t != nil && d.t.data != nil {
						dv := make([]float32, numel(d.shape))
						if len(dv) > 0 {
							C.ggml_backend_tensor_get(d.t, unsafe.Pointer(&dv[0]), 0, C.size_t(len(dv)*4))
						}
						maxAbs, dbad := 0.0, 0
						for _, v := range dv {
							f := float64(v)
							if math.IsNaN(f) || math.IsInf(f, 0) {
								dbad++
							} else if math.Abs(f) > maxAbs {
								maxAbs = math.Abs(f)
							}
						}
						var atv []float32
						for _, i := range at {
							if i < len(dv) {
								atv = append(atv, dv[i])
							}
						}
						fmt.Fprintf(os.Stderr, "ggml check:   input %v: %d NaN/Inf, max |v| %.4g, values at the bad indices %v\n", d.shape, dbad, maxAbs, atv)
					}
				}
				os.Exit(3)
			}
		}
	}
	if timing {
		t3 := time.Now()
		fmt.Fprintf(os.Stderr, "ggml flush: %d results, %d nodes, %s alloc, %s build, %s compute\n",
			len(pending), int(C.ggml_graph_n_nodes(g)), t1.Sub(t0).Round(10*time.Microsecond), t2.Sub(t1).Round(10*time.Microsecond), t3.Sub(t2).Round(10*time.Microsecond))
	}
	grp := &batch{ctx: ctx, buf: buf, refs: len(pending)}
	for _, p := range pending {
		p.t.op = C.GGML_OP_NONE
		for i := range p.t.src {
			p.t.src[i] = nil
		}
		p.t.view_src = nil
		p.t.flags = 0
		p.grp = grp
		p.deps = nil
	}
	if st != C.GGML_STATUS_SUCCESS {
		panic(fmt.Sprintf("ggml: compute failed (%d)", int(st)))
	}
}

// flush computes whatever is recorded (a readback, Eval, Sweep need it).
func (b *Backend) flush() {
	b.mu.Lock()
	b.flushLocked()
	b.mu.Unlock()
}

// trace (JANIS_GGML_TRACE=1) prints every op as it is recorded; timing
// (JANIS_GGML_TIMING=1) prints what each flush cost.
var (
	trace  = os.Getenv("JANIS_GGML_TRACE") == "1"
	timing = os.Getenv("JANIS_GGML_TIMING") == "1"
	// check (JANIS_GGML_CHECK=1) reads every result back after each flush
	// and stops at the first NaN or Inf, naming the op that made it.
	check = os.Getenv("JANIS_GGML_CHECK") == "1"
)

// f32 is x as an F32 node in ctx (a cast when it is not).
func f32(ctx *C.struct_ggml_context, x *tensor) *C.struct_ggml_tensor {
	if x.dt == engine.F32 {
		return x.t
	}
	return C.ggml_cast(ctx, x.t, C.GGML_TYPE_F32)
}

// Eval computes everything recorded so far; the engine calls it once per
// decode step with the step's survivors.
func (b *Backend) Eval(ts ...engine.Tensor) { b.flush() }

// AsyncEval is Eval here (ggml's compute is synchronous on the CPU; a GPU
// backend queues it).
func (b *Backend) AsyncEval(ts ...engine.Tensor) { b.flush() }

// ---- the engine's lifetime protocol (engine.Sweeper) ----

// PinAll pins everything made so far: the weights, after the load.
func (b *Backend) PinAll() {
	b.flush()
	b.mu.Lock()
	defer b.mu.Unlock()
	for wp := range b.tracked {
		if t := wp.Value(); t != nil {
			t.pin = true
		} else {
			delete(b.tracked, wp)
		}
	}
}

func (b *Backend) Pin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			as(t).pin = true
		}
	}
}

func (b *Backend) Unpin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			as(t).pin = false
		}
	}
}

// Sweep frees every tracked tensor that is not pinned: a step's temporaries.
func (b *Backend) Sweep() {
	b.flush()
	b.mu.Lock()
	defer b.mu.Unlock()
	for wp := range b.tracked {
		t := wp.Value()
		if t == nil || t.freed {
			delete(b.tracked, wp)
			continue
		}
		if !t.pin {
			t.free()
			delete(b.tracked, wp)
		}
	}
}

func (b *Backend) Floats(t engine.Tensor) []float32 {
	b.flush()
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
		b.flush()
		C.ggml_backend_tensor_get(c.t, unsafe.Pointer(&out[0]), 0, C.size_t(n*4))
	}
	return out
}

func (b *Backend) Ints(t engine.Tensor) []int32 {
	b.flush()
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
	b.flush()
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
	b.flush() // a recorded node may still read what is being freed
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
	if ta.dropped {
		panic("ggml: a weight's packed words were dropped after its Q4_1 repack; it was never gathered before")
	}
	ta.gathered = true
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
	tpos := b.positionsFor(offset, L)
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
		tmask = b.maskFor(Lq, Lk)
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

// positionsFor is the pinned I32 vector offset..offset+L, made once per
// (offset, L): every layer of a step asks for the same one.
func (b *Backend) positionsFor(offset, L int) *tensor {
	b.mu.Lock()
	t, ok := b.positions[[2]int{offset, L}]
	b.mu.Unlock()
	if ok {
		return t
	}
	pos := make([]int32, L)
	for i := range pos {
		pos[i] = int32(offset + i)
	}
	t = as(b.FromInt32(pos, L))
	t.pin = true
	b.mu.Lock()
	if len(b.positions) > 64 { // a decode step moves on: let the old ones go
		b.flushLocked()
		for _, old := range b.positions {
			old.free()
		}
		b.positions = map[[2]int]*tensor{}
	}
	b.positions[[2]int{offset, L}] = t
	b.mu.Unlock()
	return t
}

// maskFor is the pinned causal mask [Lq, Lk] (0 kept, -inf above the
// diagonal offset by Lk-Lq), made once per shape.
func (b *Backend) maskFor(Lq, Lk int) *tensor {
	b.mu.Lock()
	t, ok := b.masks[[2]int{Lq, Lk}]
	b.mu.Unlock()
	if ok {
		return t
	}
	m := make([]float32, Lk*Lq)
	for i := 0; i < Lq; i++ {
		for j := 0; j < Lk; j++ {
			if j > i+(Lk-Lq) {
				m[i*Lk+j] = float32(math.Inf(-1))
			}
		}
	}
	t = as(b.FromFloats(m, Lq, Lk))
	t.pin = true
	b.mu.Lock()
	if len(b.masks) > 64 {
		b.flushLocked()
		for _, old := range b.masks {
			old.free()
		}
		b.masks = map[[2]int]*tensor{}
	}
	b.masks[[2]int{Lq, Lk}] = t
	b.mu.Unlock()
	return t
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

// rangeCheck (under JANIS_GGML_CHECK) reports affine parameters f16 cannot
// hold: a NaN, an infinity, or a magnitude past 65504 becomes Inf in a
// Q4_1 block and NaN in the matmul.
func rangeCheck(what string, shape []int, vals []float32) {
	bad, maxAbs := 0, 0.0
	for _, v := range vals {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) || math.Abs(f) > 65504 {
			bad++
		}
		if a := math.Abs(f); a > maxAbs && !math.IsInf(f, 0) && !math.IsNaN(f) {
			maxAbs = a
		}
	}
	fmt.Fprintf(os.Stderr, "ggml check: %s of %v: %d of %d outside f16, max |v| %.4g\n", what, shape, bad, len(vals), maxAbs)
}

// packed is the device form of an affine weight, chosen once at its first
// use: Q4_1 (the same 4-bit affine blocks, ggml's quantized kernels) unless
// the input that meets it carries values past q8Threshold, in which case
// Q8_0. ggml's Q4_1 kernels quantize the activations to q8_1 blocks whose
// scale and block sum are f16, and a block holding a value past about
// 2,000 overflows the sum: NaN in the output. Qwen3's attention outputs
// have such channels (the 30B MoE answered NaN logits on the L4, hidden by
// argmax, exposed by sampling), so the projections that take them go to
// Q8_0, whose kernel has no block sum, at twice the bytes for those weights
// alone. Scaling the activations instead was tried and is wrong: dividing
// by 128 pushes small blocks' scales under f16's normal range.
func (b *Backend) packed(x, tw, ts, tb *tensor, groupSize, bits int) *tensor {
	if tw.q4 != nil {
		return tw.q4
	}
	if bits == 4 && groupSize%32 == 0 {
		maxAbs := float32(0)
		for _, v := range b.Floats(x) {
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs > q8Threshold || forceQ8 {
			if check || trace {
				fmt.Fprintf(os.Stderr, "ggml: weight %v kept as Q8_0 (its input reaches %.4g)\n", tw.shape, maxAbs)
			}
			return b.q8_0(tw, ts, tb, groupSize, bits)
		}
	}
	return b.q4_1(tw, ts, tb, groupSize, bits)
}

// q8Threshold is the input magnitude past which a weight is kept as Q8_0:
// half of where the q8_1 block sum overflows, with room for later tokens.
// JANIS_GGML_Q8=1 keeps every dense weight as Q8_0 (for measuring).
const q8Threshold = 1024

var forceQ8 = os.Getenv("JANIS_GGML_Q8") == "1"

// q8_0 repacks an affine 4-bit weight into ggml's Q8_0 (blocks of 32: f16
// scale d, 32 int8): each value q*scale+bias rounded to d = max|v|/127,
// a sixteenth of a 4-bit step at most. Built once, pinned; the words go.
func (b *Backend) q8_0(tw, ts, tb *tensor, groupSize, bits int) *tensor {
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	G := ts.shape[len(ts.shape)-1]
	rows := numel(tw.shape) / P
	const blk = 34 // sizeof(block_q8_0): 2+32
	raw := make([]byte, rows*(in/32)*blk)
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			vals := make([]float32, 32)
			for r := w; r < rows; r += workers {
				for blkIdx := 0; blkIdx < in/32; blkIdx++ {
					amax := float32(0)
					for j := 0; j < 32; j++ {
						e := blkIdx*32 + j
						q := (uint32(words[r*P+e/valsPerWord]) >> uint((e%valsPerWord)*bits)) & mask
						g := e / groupSize
						v := float32(q)*sc[r*G+g] + bi[r*G+g]
						vals[j] = v
						if v < 0 {
							v = -v
						}
						if v > amax {
							amax = v
						}
					}
					d := amax / 127
					o := (r*(in/32) + blkIdx) * blk
					binary.LittleEndian.PutUint16(raw[o:], f32tof16(d))
					for j, v := range vals {
						qv := 0
						if d > 0 {
							qv = int(math.Round(float64(v / d)))
						}
						raw[o+2+j] = byte(int8(qv))
					}
				}
			}
		}(w)
	}
	wg.Wait()
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q8_0, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	q := &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16, pin: true}
	tw.q4 = q
	if !tw.gathered && tw.buf != nil {
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
	return q
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
		return b.deviceF16(tw, ts, tb, groupSize, bits)
	}
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	if check {
		rangeCheck("scales", tw.shape, sc)
		rangeCheck("biases", tw.shape, bi)
	}
	P := tw.shape[len(tw.shape)-1]
	in := P * 8
	G := ts.shape[len(ts.shape)-1]
	rows := numel(tw.shape) / P
	const blk = 20 // sizeof(block_q4_1): 2+2+16
	raw := make([]byte, rows*(in/32)*blk)
	// Rows are independent: one goroutine per slice of them (the repack of
	// a whole model is done once, at the first use of each weight).
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for r := w; r < rows; r += workers {
				for blkIdx := 0; blkIdx < in/32; blkIdx++ {
					g := (blkIdx * 32) / groupSize
					o := (r*(in/32) + blkIdx) * blk
					binary.LittleEndian.PutUint16(raw[o:], f32tof16(sc[r*G+g]))
					binary.LittleEndian.PutUint16(raw[o+2:], f32tof16(bi[r*G+g]))
					// Eight nibbles per word, element j of the block at bits 4*(j%8).
					for wi := 0; wi < 4; wi++ {
						word := uint32(words[r*P+blkIdx*4+wi])
						for k := 0; k < 8; k++ {
							j := wi*8 + k
							q := byte((word >> uint(k*4)) & 0xF)
							if j < 16 {
								raw[o+4+j] |= q
							} else {
								raw[o+4+j-16] |= q << 4
							}
						}
					}
				}
			}
		}(w)
	}
	wg.Wait()
	if check {
		// Dequantize the blocks back and compare with the source: a bad row
		// in the matmul is either here or in the kernel.
		worst, worstRow, nan := 0.0, -1, 0
		for r := 0; r < rows; r++ {
			for blkIdx := 0; blkIdx < in/32; blkIdx++ {
				o := (r*(in/32) + blkIdx) * blk
				d := float64(f16tof32(binary.LittleEndian.Uint16(raw[o:])))
				m := float64(f16tof32(binary.LittleEndian.Uint16(raw[o+2:])))
				g := (blkIdx * 32) / groupSize
				for j := 0; j < 32; j++ {
					var q byte
					if j < 16 {
						q = raw[o+4+j] & 0xF
					} else {
						q = raw[o+4+j-16] >> 4
					}
					got := d*float64(q) + m
					e := blkIdx*32 + j
					src := float64((uint32(words[r*P+e/8])>>uint((e%8)*4))&0xF)*float64(sc[r*G+g]) + float64(bi[r*G+g])
					if math.IsNaN(got) || math.IsInf(got, 0) {
						nan++
					}
					if diff := math.Abs(got - src); diff > worst {
						worst, worstRow = diff, r
					}
				}
			}
		}
		fmt.Fprintf(os.Stderr, "ggml check: Q4_1 repack of %v: %d NaN/Inf, worst |repack-source| %.3g at row %d\n", tw.shape, nan, worst, worstRow)
	}
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q4_1, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	q := &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16, pin: true}
	tw.q4 = q
	if !tw.gathered && tw.buf != nil {
		// The words are not needed again: one copy of the weight, not two.
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
	return q
}

// deviceF16 is the device copy of an affine weight ggml has no block type
// for (8-bit with a bias: the MoE routers of the mlx-community checkpoints,
// 128 by 2048 each): dequantized on the host and kept as f16, which ggml's
// f16 matmul and mul_mat_id take. Built once, pinned, kept in q4 like the
// Q4_1 repack; the packed words go the same way after.
func (b *Backend) deviceF16(tw, ts, tb *tensor, groupSize, bits int) *tensor {
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	G := ts.shape[len(ts.shape)-1]
	rows := numel(tw.shape) / P
	raw := make([]byte, rows*in*2)
	for r := 0; r < rows; r++ {
		for j := 0; j < in; j++ {
			q := (uint32(words[r*P+j/valsPerWord]) >> uint((j%valsPerWord)*bits)) & mask
			g := j / groupSize
			binary.LittleEndian.PutUint16(raw[(r*in+j)*2:], f32tof16(float32(q)*sc[r*G+g]+bi[r*G+g]))
		}
	}
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = in
	q := b.newFrom(engine.F16, shape, unsafe.Pointer(&raw[0]), len(raw))
	q.pin = true
	tw.q4 = q
	if !tw.gathered && tw.buf != nil {
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
	return q
}

// QuantMatmul: x @ Wᵀ with W [out, in] affine 4-bit, on ggml's Q4_1 kernel.
func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	if !transpose {
		return b.MatMul(x, b.Dequantize(w, scales, biases, groupSize, bits))
	}
	tx := as(x)
	q := b.packed(tx, as(w), as(scales), as(biases), groupSize, bits)
	nx := len(tx.shape)
	shape := append(append([]int(nil), tx.shape[:nx-1]...), q.shape[0])
	return b.op(engine.F32, shape, []*tensor{tx, q, as(w)}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_mul_mat(ctx, q.t, f32(ctx, tx)) // ne (out, M, …)
	})
}

// GatherQuantMatmul is the MoE expert matmul on ggml_mul_mat_id: w, scales
// and biases are the stack [E, out, in] (repacked once to Q4_1, ne =
// (in, out, E)), x is [lead…, A, 1, in] with A one vector shared by the K
// slots or one per slot, rhsIndices [lead…, K] the expert per slot; the
// result is [lead…, K, 1, out], as the CPU and MLX backends give it.
func (b *Backend) GatherQuantMatmul(x, w, scales, biases, rhsIndices engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	if !transpose {
		panic("ggml: GatherQuantMatmul without transpose is not used by the models")
	}
	tx, ti := as(x), as(rhsIndices)
	q := b.packed(tx, as(w), as(scales), as(biases), groupSize, bits)
	xs, rs := tx.shape, ti.shape
	in, A := xs[len(xs)-1], xs[len(xs)-3]
	K := rs[len(rs)-1]
	nTok := numel(rs) / K
	out := q.shape[1]
	shape := append(append([]int(nil), rs...), 1, out)
	return b.op(engine.F32, shape, []*tensor{tx, ti, q, as(w)}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		xb := C.ggml_reshape_3d(ctx, f32(ctx, tx), C.int64_t(in), C.int64_t(A), C.int64_t(nTok))
		ids := C.ggml_reshape_2d(ctx, ti.t, C.int64_t(K), C.int64_t(nTok))
		r := C.ggml_mul_mat_id(ctx, q.t, xb, ids) // ne (out, K, nTok)
		m := ne(shape)
		return C.ggml_cont(ctx, C.ggml_reshape_4d(ctx, r, m[0], m[1], m[2], m[3]))
	})
}

// Conv1dDepthwise and GatedDeltaScan belong to the Qwen3.5 hybrid; not in
// the spike (Qwen3 dense first).
func (b *Backend) Conv1dDepthwise(x, w engine.Tensor) engine.Tensor {
	panic("ggml: Conv1dDepthwise is not implemented in the spike")
}
func (b *Backend) GatedDeltaScan(q, k, v, g, beta, state engine.Tensor) (engine.Tensor, engine.Tensor) {
	panic("ggml: GatedDeltaScan is not implemented in the spike")
}

func f16tof32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		return math.Float32frombits(sign) + float32(math.Ldexp(float64(mant), -24))*func() float32 {
			if sign != 0 {
				return -1
			}
			return 1
		}()
	case exp == 31:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return float32(math.NaN())
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | mant<<13)
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
