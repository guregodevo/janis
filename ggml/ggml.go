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
// janis_repack_buft is the CPU backend's weight-repacking buffer type
// (CPU_REPACK), or NULL: a weight placed in it is laid out once, when it is
// set, for the tiled int8 matmul kernels of this CPU.
static ggml_backend_buffer_type_t janis_repack_buft(ggml_backend_t be) {
	ggml_backend_dev_t dev = ggml_backend_get_device(be);
	if (dev == NULL) return NULL;
	ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
	ggml_backend_dev_get_extra_bufts_t fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
	if (fn == NULL) return NULL;
	for (ggml_backend_buffer_type_t * l = fn(dev); l != NULL && *l != NULL; l++) {
		if (strcmp(ggml_backend_buft_name(*l), "CPU_REPACK") == 0) return *l;
	}
	return NULL;
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
	"sync/atomic"
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
	// repack is the CPU's weight-repacking buffer type, nil on a device:
	// where a dense 4-bit weight goes as Q4_K (see q4K).
	repack C.ggml_backend_buffer_type_t
	name string
	mu   sync.Mutex

	rec     *C.struct_ggml_context // the recording context: nodes not yet computed
	pending []pendingRef           // their results, in order, held weakly
	// extra are nodes the next flush must run before the results: the
	// writes of new keys and values into a cache's spare room, which no
	// result depends on through the graph. extraDeps keeps what they read.
	extra     []*C.struct_ggml_tensor
	extraDeps []*tensor
	// tracked is every tensor made since Open, weakly: Sweep frees the ones
	// that are alive and not pinned; a collected one needs nothing.
	tracked map[weak.Pointer[tensor]]struct{}
	flushes int
	// lastBatch is what the last flush's batch took on the device, and how
	// many of its results were held: what a test of the reuse reads.
	lastBatch struct {
		bytes   C.size_t
		results int
		held    int
	}
	// Small constants every layer re-asks for: the RoPE positions of a step
	// and the causal mask of a shape. Uploaded once, pinned.
	positions map[[2]int]*tensor
	masks     map[[2]int]*tensor
	// arenas are a few fixed buffers that batches are placed into with a
	// linear allocator and reused once their results are gone: a flush is
	// not a fresh device allocation of yet another size, which is what
	// fragmented the L4's heap until a 100 MB request failed with 1.7 GB
	// nominally free. A batch that does not fit, or finds every arena busy,
	// gets a buffer of its own as before.
	arenas []*arena
}

type arena struct {
	buf  C.ggml_backend_buffer_t
	size C.size_t
	busy *batch // the batch living in it, nil when free
}

// arenaSizes are the arenas made at start: batches are small (on the 30B
// MoE a median under a megabyte, the largest 18 MB) and live a few flushes
// each, so many small slots catch nearly all of them, two large ones the
// prefill's. Placement takes the smallest free arena that fits.
var arenaSizes = append(append([]C.size_t{}, repeatSize(12, 32<<20)...), 128<<20, 128<<20)

func repeatSize(n int, sz C.size_t) []C.size_t {
	out := make([]C.size_t, n)
	for i := range out {
		out[i] = sz
	}
	return out
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
		if os.Getenv("JANIS_GGML_REPACK") != "0" {
			b.repack = C.janis_repack_buft(b.be)
		}
	}
	b.buft = C.ggml_backend_get_default_buffer_type(b.be)
	b.name = C.GoString(C.ggml_backend_name(b.be))
	b.tracked = map[weak.Pointer[tensor]]struct{}{}
	b.positions = map[[2]int]*tensor{}
	b.masks = map[[2]int]*tensor{}
	for _, sz := range arenaSizes {
		if buf := C.ggml_backend_buft_alloc_buffer(b.buft, sz); buf != nil {
			b.arenas = append(b.arenas, &arena{buf: buf, size: sz})
		}
	}
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
	for _, a := range b.arenas {
		C.ggml_backend_buffer_free(a.buf)
	}
	b.arenas = nil
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
	// via holds what a pending input reads, through that input's own holder
	// rather than the input itself (which may be a temporary the model
	// drops): a residual chain shares its ancestry by reference, not by copy
	// (a copied list doubled every layer and ate 64 GB of host memory on
	// the 27B, 2026-10-26).
	via *depHolder
	shape []int
	dt    engine.DType
	where string     // the op's call site, kept when tracing or checking
	quant *quantCall // set on a QuantMatmul result: what a dump needs to replay it
	q4    *tensor    // the Q4_1 repack of a packed weight, built once
	pin   bool       // survives Sweep
	freed bool
	// gathered: rows of this packed weight were taken (the embedding), so
	// its words stay after the repack; a weight only ever multiplied keeps
	// the Q4_1 copy alone.
	gathered bool
	dropped  bool // the words are gone: only QuantMatmul may use it now
	cacheLen int  // a cache view: tokens in use along the token axis
}

type quantCall struct{ groupSize, bits int }

// batch is one flushed graph's context and buffer, shared by its results and
// released when the last of them goes.
type batch struct {
	ctx    *C.struct_ggml_context
	buf    C.ggml_backend_buffer_t // its own buffer, when no arena took it
	in     *arena                  // the arena it lives in, else nil
	galloc C.ggml_gallocr_t        // the graph allocator that laid the batch out, owning its buffer
	refs   int
	// A cache buffer: base is the f16 tensor holding the whole capacity,
	// every cache tensor a view of a prefix of it; Pin leaves it in place.
	base *C.struct_ggml_tensor
	cap  int // capacity along the token axis
}

// memMu guards the refcounts and arena state that finalizers touch from
// their own goroutine while the engine records and flushes.
var memMu sync.Mutex

// release drops one reference; called with memMu held.
func (g *batch) release() {
	g.refs--
	if g.refs > 0 {
		return
	}
	if g.in != nil {
		g.in.busy = nil
		g.in = nil
	}
	if g.buf != nil {
		C.ggml_backend_buffer_free(g.buf)
	}
	if g.galloc != nil {
		C.ggml_gallocr_free(g.galloc)
	}
	if g.ctx != nil {
		C.ggml_free(g.ctx)
	}
	g.buf, g.galloc, g.ctx = nil, nil, nil
}

// pendingRef is a recorded, not yet computed result: its node in the
// recording context, and the result itself weakly, so the flush can tell a
// temporary the model has dropped from a result it still holds. In check
// mode every result is held, to be read back after the compute.
type pendingRef struct {
	node   *C.struct_ggml_tensor
	wp     weak.Pointer[tensor]
	strong *tensor
	bytes  C.size_t
}

// reuseAbove is the batch size, in bytes, from which a flush asks the
// collector which results are still held before laying the batch out: a
// collection costs a millisecond, which a decode step's small batch does
// not pay and a prefill chunk's hundreds of megabytes do
// (JANIS_GGML_REUSE_ABOVE, in MiB, overrides).
var reuseAbove = func() C.size_t {
	if n, err := strconv.Atoi(os.Getenv("JANIS_GGML_REUSE_ABOVE")); err == nil && n >= 0 {
		return C.size_t(n) << 20
	}
	return 64 << 20
}()

// depHolder is what a pending result keeps alive for the nodes that read
// it: its direct inputs and, by reference, the holders of its own pending
// inputs. Reachable from the result and from every pending node that reads
// it, until the flush that computes them.
type depHolder struct {
	deps []*tensor
	via  []*depHolder
}

// pendingInput reports whether a tensor is a recorded result not yet
// computed: it has no memory of its own that a dependency would keep alive,
// its node lives in the recording context.
func pendingInput(t *tensor) bool {
	return t.grp == nil && t.ctx == nil && t.buf == nil && t.t != nil && t.t.data == nil
}

func (t *tensor) Shape() []int { return append([]int(nil), t.shape...) }

// free releases what the tensor holds; a recorded, never computed node has
// nothing of its own yet and is left to its flush. Idempotent.
func (t *tensor) free() {
	memMu.Lock()
	defer memMu.Unlock()
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
		q := t.q4
		t.q4 = nil
		memMu.Unlock()
		q.free()
		memMu.Lock()
	}
	t.deps, t.via = nil, nil
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
	if buf == nil {
		C.ggml_free(ctx)
		panic(fmt.Sprintf("ggml: no room on the device for a leaf of %v", shape))
	}
	logAlloc("leaf", C.ggml_nbytes(t), shape)
	if size > 0 {
		C.ggml_backend_tensor_set(t, raw, 0, C.size_t(size))
	}
	out := &tensor{t: t, ctx: ctx, buf: buf, shape: append([]int(nil), shape...), dt: dt}
	b.track(out)
	return out
}

// track registers a tensor for Sweep (weakly) and for the collector.
func (b *Backend) track(t *tensor) {
	if freeLog && t.where == "" {
		_, file, line, _ := runtime.Caller(2)
		t.where = fmt.Sprintf("%s:%d", file[strings.LastIndex(file, "/")+1:], line)
	}
	runtime.SetFinalizer(t, func(x *tensor) {
		if freeLog && !x.freed {
			fmt.Fprintf(os.Stderr, "ggml finalizer: %v %s cache=%v grp=%v\n", x.shape, x.where, x.cacheLen, x.grp != nil)
		}
		x.free()
	})
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
		// A dependency keeps an input's memory until this node is computed.
		// An input that is itself pending has none yet, its node lives in
		// the same context: holding it would keep a dropped temporary alive
		// and make the flush give it memory of its own. What it reads is
		// held in its place: a pending input's own dependencies, so the
		// leaf at the bottom of a chain the model dropped stays until the
		// chain is computed (freed under the graph on the 27B's load, 2026-10-26).
		if in == nil {
			continue
		}
		if check || !pendingInput(in) {
			out.deps = append(out.deps, in)
		} else {
			if out.via == nil {
				out.via = &depHolder{}
			}
			if in.via == nil {
				in.via = &depHolder{deps: in.deps}
			}
			out.via.via = append(out.via.via, in.via)
		}
	}
	if out.via != nil {
		out.via.deps = out.deps
	}
	ref := pendingRef{node: node, wp: weak.Make(out), bytes: C.ggml_nbytes(node)}
	if check {
		ref.strong = out
	}
	b.pending = append(b.pending, ref)
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
	if (len(b.pending) == 0 && len(b.extra) == 0) || b.rec == nil {
		return
	}
	ctx, pending := b.rec, b.pending
	b.rec, b.pending = nil, nil
	t0 := time.Now()
	// Which results does the model still hold? Those are the batch's
	// outputs and get memory of their own; a result nothing holds is a
	// temporary, consumed inside this graph or by nothing, and ggml's graph
	// allocator lets the temporaries share memory in the order they die.
	// Every result used to get its own: a 32-token prefill chunk of the 27B
	// hybrid held 2.2 GB of them until the chunk's sweep, on an L4 with
	// 3 GB to spare beside the weights (2026-10-26).
	var total C.size_t
	for _, p := range pending {
		total += p.bytes
	}
	if !check && total >= reuseAbove {
		runtime.GC()
	}
	live := make([]*tensor, 0, len(pending))
	for _, p := range pending {
		t := p.strong
		if t == nil {
			t = p.wp.Value()
		}
		if t != nil {
			live = append(live, t)
			// A backend's op fusion (ggml-cuda folds a SiLU into the
			// multiply after it, a matmul into the add after it) skips
			// writing a fused node's own buffer unless it is flagged an
			// output; without the flag the 30B MoE read never-written
			// memory as its activations.
			C.ggml_set_output(p.node)
		}
	}
	g := C.ggml_new_graph_custom(ctx, maxNodes, false)
	extra, extraDeps := b.extra, b.extraDeps
	b.extra, b.extraDeps = nil, nil
	for _, e := range extra {
		C.ggml_build_forward_expand(g, e) // first in the graph: the cache writes
	}
	for _, t := range live {
		C.ggml_build_forward_expand(g, t.t) // a dropped result nothing reads is not computed at all
	}
	t1 := time.Now()
	galloc := C.ggml_gallocr_new(b.buft)
	if !C.ggml_gallocr_alloc_graph(galloc, g) {
		// Out of device memory. Computing the graph over unallocated
		// tensors was a SIGSEGV that took the server down; a panic is
		// caught per request by net/http and the server answers the next one.
		C.ggml_gallocr_free(galloc)
		C.ggml_free(ctx)
		panic(fmt.Sprintf("ggml: no room on the device for a batch of %d results, %d of them held (%.1f MiB without reuse)", len(pending), len(live), float64(total)/(1<<20)))
	}
	b.lastBatch.bytes, b.lastBatch.results, b.lastBatch.held = C.ggml_gallocr_get_buffer_size(galloc, 0), len(pending), len(live)
	if memLog {
		logAlloc(fmt.Sprintf("batch of %d results, %d held, %.1f MiB without reuse", len(pending), len(live), float64(total)/(1<<20)), b.lastBatch.bytes, nil)
	}
	t2 := time.Now()
	st := C.ggml_backend_graph_compute(b.be, g)
	runtime.KeepAlive(extraDeps)
	if check {
		for _, p := range live {
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
				if p.quant != nil {
					b.dumpQuantCall(p)
				}
				os.Exit(3)
			}
		}
	}
	if timing {
		t3 := time.Now()
		fmt.Fprintf(os.Stderr, "ggml flush: %d results, %d held, %d nodes, %s build, %s alloc, %s compute\n",
			len(pending), len(live), int(C.ggml_graph_n_nodes(g)), t1.Sub(t0).Round(10*time.Microsecond), t2.Sub(t1).Round(10*time.Microsecond), t3.Sub(t2).Round(10*time.Microsecond))
	}
	grp := &batch{ctx: ctx, galloc: galloc, refs: len(live)}
	// Results nobody holds any more are freed by their finalizers, and
	// their batches with them; the collector runs on heap pressure, which
	// a few thousand small Go objects never give it, so it is asked every
	// few flushes: without this a step's every intermediate stayed on the
	// device until the engine's sweep (1.4 GB on the 30B MoE at a 6k-token
	// prompt, in allocations that fragmented the L4's heap).
	b.flushes++
	if gcEvery > 0 && b.flushes%gcEvery == 0 {
		if gcSync {
			runtime.GC()
		} else {
			go runtime.GC()
		}
	}
	for _, p := range pending {
		p.node.op = C.GGML_OP_NONE
		for i := range p.node.src {
			p.node.src[i] = nil
		}
		p.node.view_src = nil
		p.node.flags = 0
	}
	for _, t := range live {
		t.grp = grp
		t.deps, t.via = nil, nil
	}
	if len(live) == 0 {
		// Nothing holds a result: the batch (its context, the allocator's
		// buffer) has no tensor to be released by and goes now.
		grp.refs = 1
		memMu.Lock()
		grp.release()
		memMu.Unlock()
	}
	if st != C.GGML_STATUS_SUCCESS {
		panic(fmt.Sprintf("ggml: compute failed (%d)", int(st)))
	}
}

// placeInArena lays the context's tensors out in a free arena with a linear
// allocator (views initialized after their sources), or returns nil when
// none is free or the batch does not fit.
func (b *Backend) placeInArena(ctx *C.struct_ggml_context) *arena {
	var need C.size_t
	align := C.ggml_backend_buft_get_alignment(b.buft)
	for t := C.ggml_get_first_tensor(ctx); t != nil; t = C.ggml_get_next_tensor(ctx, t) {
		if t.data != nil || t.view_src != nil {
			continue
		}
		sz := C.ggml_backend_buft_get_alloc_size(b.buft, t)
		need += (sz + align - 1) / align * align
	}
	var a *arena
	memMu.Lock()
	for _, cand := range b.arenas { // the smallest free one that fits
		if cand.busy == nil && need <= cand.size && (a == nil || cand.size < a.size) {
			a = cand
		}
	}
	if a != nil {
		a.busy = &batch{} // claimed until the flush sets its batch
	}
	memMu.Unlock()
	if a == nil {
		return nil
	}
	tal := C.ggml_tallocr_new(a.buf)
	var placed []*C.struct_ggml_tensor
	for t := C.ggml_get_first_tensor(ctx); t != nil; t = C.ggml_get_next_tensor(ctx, t) {
		if t.data != nil {
			// A view of a tensor that already had data was given its address
			// at creation but no buffer; ggml's own allocator still runs
			// view_init on it, and CUDA reads a source's buffer.
			if t.view_src != nil && t.buffer == nil {
				C.ggml_backend_view_init(t)
			}
			continue
		}
		var st C.enum_ggml_status
		if t.view_src != nil {
			st = C.ggml_backend_view_init(t)
		} else {
			st = C.ggml_tallocr_alloc(&tal, t)
		}
		if st != C.GGML_STATUS_SUCCESS {
			if check || trace {
				fmt.Fprintf(os.Stderr, "ggml: arena placement failed (%d) for a tensor of %d bytes (view %v) after %d placed, need %d of %d\n", int(st), int(C.ggml_nbytes(t)), t.view_src != nil, len(placed), int(need), int(a.size))
			}
			for _, p := range placed { // undo, and let the batch take a buffer of its own
				p.data, p.buffer = nil, nil
			}
			memMu.Lock()
			a.busy = nil
			memMu.Unlock()
			return nil
		}
		placed = append(placed, t)
	}
	return a
}

// flush computes whatever is recorded (a readback, Eval, Sweep need it).
func (b *Backend) flush() {
	b.mu.Lock()
	b.flushLocked()
	b.mu.Unlock()
}

// dumpQuantCall writes a NaN-producing QuantMatmul's input and weight
// (x, words, scales, biases, shapes) under /tmp/janis-nan, for
// TestDumpedQuantMatmul to replay on each backend.
func (b *Backend) dumpQuantCall(p *tensor) {
	if len(p.deps) < 5 {
		return
	}
	x, w, sc, bi := p.deps[0], p.deps[2], p.deps[3], p.deps[4]
	if w.t == nil {
		fmt.Fprintln(os.Stderr, "ggml check: the weight's words were dropped; no dump")
		return
	}
	dir := "/tmp/janis-nan"
	_ = os.MkdirAll(dir, 0o755)
	write := func(name string, v any) {
		f, err := os.Create(dir + "/" + name)
		if err != nil {
			return
		}
		defer f.Close()
		_ = binary.Write(f, binary.LittleEndian, v)
	}
	write("x.f32", b.Floats(x))
	write("w.u32", b.Ints(w))
	write("scales.f32", b.Floats(sc))
	write("biases.f32", b.Floats(bi))
	meta := fmt.Sprintf(`{"x":%v,"w":%v,"scales":%v,"group_size":%d,"bits":%d}`, jsonInts(x.shape), jsonInts(w.shape), jsonInts(sc.shape), p.quant.groupSize, p.quant.bits)
	_ = os.WriteFile(dir+"/meta.json", []byte(meta), 0o644)
	fmt.Fprintf(os.Stderr, "ggml check: dumped the call to %s (%s)\n", dir, meta)
}

func jsonInts(v []int) string {
	parts := make([]string, len(v))
	for i, d := range v {
		parts[i] = fmt.Sprint(d)
	}
	return "[" + strings.Join(parts, ",") + "]"
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

// Pin marks survivors (the KV caches, the logits) and moves each out of
// its batch into a buffer of its own: a batch holds every intermediate of
// its graph, and a survivor that stayed in it would keep hundreds of
// megabytes alive for a step or longer (the 30B MoE ran the L4 out of
// memory loading its last experts while the caches held the prefill's
// batches). The copy is device to device and the caches are small.
func (b *Backend) Pin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t == nil {
			continue
		}
		x := as(t)
		x.pin = true
		b.own(x)
	}
}

// own gives a batched result its own buffer. Anything recorded against the
// old node is computed first, so no pending graph refers to it after.
func (b *Backend) own(x *tensor) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if x.grp == nil && x.t != nil && x.t.data == nil {
		b.flushLocked() // pending: computed now, into a batch
	}
	if x.grp == nil || x.t == nil || x.grp.base != nil {
		return // a leaf, or a cache view: both already own their memory
	}
	ctx := C.janis_ctx(ctxMem)
	n := ne(x.shape)
	t := C.janis_new(ctx, x.t._type, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if buf == nil {
		C.ggml_free(ctx)
		return // no room for the copy: it stays in its batch
	}
	C.ggml_backend_tensor_copy(x.t, t)
	runtime.KeepAlive(x)
	memMu.Lock()
	x.grp.release()
	memMu.Unlock()
	x.grp, x.t, x.ctx, x.buf = nil, t, ctx, buf
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
		runtime.KeepAlive(c)
	}
	// The Go tensor is dead once its C pointer is read; without this its
	// finalizer may free the buffer while the copy-out still reads it.
	runtime.KeepAlive(x)
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
		runtime.KeepAlive(x)
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
	runtime.KeepAlive(x)
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
	if ta.dt == engine.F16 && tb.dt == engine.F16 && axis == nd-2 {
		return b.appendCache(ta, tb, axis)
	}
	return b.op(engine.F32, shape, []*tensor{ta, tb}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_concat(ctx, f32(ctx, ta), f32(ctx, tb), C.int(nd-1-axis))
	})
}

// appendCache is Concat for the f16 KV cache along its token axis, in
// place: the cache lives in a buffer with spare room (cacheGrow tokens
// beyond what is used), the new keys or values are copied into that room
// by a node the next flush runs first, and the result is a view of the
// longer prefix sharing the buffer. No pass over the whole cache a step,
// no old and new copies alive at once; a cache that has no room, or that
// came from somewhere else (a slice, the first step), moves into a new
// buffer first. The old view stays valid: it covers the prefix only.
func (b *Backend) appendCache(ta, tb *tensor, axis int) engine.Tensor {
	n, m := ta.shape[axis], tb.shape[axis]
	need := n + m
	shape := append([]int(nil), ta.shape...)
	shape[axis] = need
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.rec == nil {
		b.rec = C.janis_ctx(C.size_t(C.ggml_tensor_overhead())*maxNodes + C.ggml_graph_overhead_custom(maxNodes, false))
	}
	grp := ta.grp
	if grp == nil || grp.base == nil || ta.cacheLen != n || grp.cap < need {
		// A new buffer with room: the old cache is copied in by the flush.
		capShape := append([]int(nil), shape...)
		capShape[axis] = need + cacheGrow
		// The buffer's context also holds every prefix view made over its
		// life (one a step, up to cacheGrow of them): it must outlive the
		// recording contexts those steps flush.
		ctx := C.janis_ctx(ctxMem + C.size_t(C.ggml_tensor_overhead())*(cacheGrow+64))
		cne := ne(capShape)
		base := C.janis_new(ctx, C.GGML_TYPE_F16, cne[0], cne[1], cne[2], cne[3])
		buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
		if buf == nil {
			C.ggml_free(ctx)
			panic("ggml: no room on the device for the cache")
		}
		grp = &batch{ctx: ctx, buf: buf, base: base, cap: capShape[axis]}
		logAlloc("cache", C.ggml_nbytes(base), capShape)
		b.writeCache(grp, ta, 0)
	}
	b.writeCache(grp, tb, n)
	out := &tensor{shape: shape, dt: engine.F16, grp: grp, cacheLen: need}
	memMu.Lock()
	grp.refs++
	memMu.Unlock()
	out.t = b.cacheView(grp, shape)
	b.mu.Unlock()
	b.track(out)
	b.mu.Lock()
	return out
}

// cacheGrow is the spare room a cache buffer is made with, in tokens.
const cacheGrow = 256

// gcEvery is how many flushes pass between collections (see flushLocked);
// JANIS_GGML_GC overrides (0 turns it off), JANIS_GGML_GCSYNC=1 collects
// on the flushing goroutine itself. Both for measuring.
var (
	gcEvery = func() int {
		if v, err := strconv.Atoi(os.Getenv("JANIS_GGML_GC")); err == nil && v >= 0 {
			return v
		}
		return 4
	}()
	gcSync  = os.Getenv("JANIS_GGML_GCSYNC") == "1"
	freeLog = os.Getenv("JANIS_GGML_FREELOG") == "1"
	// memLog (JANIS_GGML_MEMLOG=1) prints every device allocation the
	// backend makes, with its size and what it is for.
	memLog = os.Getenv("JANIS_GGML_MEMLOG") == "1"
)

func logAlloc(what string, bytes C.size_t, shape []int) {
	if memLog {
		fmt.Fprintf(os.Stderr, "ggml alloc: %s %.1f MiB %v\n", what, float64(bytes)/(1<<20), shape)
	}
}

// cacheView is a view of the first shape[axis] tokens of a cache buffer.
func (b *Backend) cacheView(grp *batch, shape []int) *C.struct_ggml_tensor {
	m := ne(shape)
	base := grp.base
	v := C.ggml_view_4d(grp.ctx, base, m[0], m[1], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], 0)
	C.ggml_backend_view_init(v)
	return v
}

// writeCache records the copy of x into the cache buffer at token offset
// off along the token axis (nd-2, ggml's ne1), to run first in the next
// flush.
func (b *Backend) writeCache(grp *batch, x *tensor, off int) {
	m := ne(x.shape)
	base := grp.base
	dst := C.ggml_view_4d(b.rec, base, m[0], m[1], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], C.size_t(off)*C.size_t(base.nb[1]))
	C.ggml_backend_view_init(dst)
	srcT := x.t
	if x.dt != engine.F16 {
		srcT = C.ggml_cast(b.rec, srcT, C.GGML_TYPE_F16)
	}
	b.extra = append(b.extra, C.ggml_cpy(b.rec, srcT, dst))
	b.extraDeps = append(b.extraDeps, x)
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
		// Keys and values may be f16 (the cache): mul_mat takes an f16
		// first operand against f32 as it is, no cast over the whole cache.
		qq := f32(ctx, tq)
		kk, vv := tk.t, tv.t
		if tk.dt != engine.F32 && tk.dt != engine.F16 {
			kk = f32(ctx, tk)
		}
		if tv.dt != engine.F32 && tv.dt != engine.F16 {
			vv = f32(ctx, tv)
		}
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
// use: on the CPU Q4_K in the repack buffer (q4K), elsewhere Q4_1 (the same
// 4-bit affine blocks, ggml's quantized kernels), unless
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
	if q := b.q4K(tw, ts, tb, groupSize, bits); q != nil {
		return q
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
	logAlloc("Q8_0 weight", C.ggml_nbytes(t), shape)
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
	logAlloc("Q4_1 weight", C.ggml_nbytes(t), shape)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	q := &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16, pin: true}
	tw.q4 = q
	if !tw.gathered && tw.buf != nil && !check {
		// The words are not needed again: one copy of the weight, not two.
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
	return q
}

// q4K is the CPU form of a dense 4-bit weight: Q4_K in the CPU's repack
// buffer, which ggml lays out once for its tiled int8 kernels (AVX2, ARM
// dotprod and i8mm) — the matmul llama.cpp runs a Q4_K_M model with. Q4_1
// has no such kernel: its row-by-row dot product prefilled a 3,335-token
// prompt at 4.9 tok/s on a GitHub runner (2026-10-29), where llama.cpp
// does about 30. A weight whose input carries an outlier channel is still
// kept as Q8_0 (packed): the activations are quantized to 8 bits per block
// here too, and an outlier flattens the rest of its block. The MLX affine values are dequantized and quantized
// again by ggml's reference Q4_K quantizer (sub-blocks of 32 with their own
// scale and min, the same affine form), so the result is within Q4_K's
// rounding of the source. nil when this weight cannot go there: not on the
// CPU, not 4-bit, not 2D, rows not a multiple of 8 or a row not of whole
// 256-value blocks, or the check and trace modes, which read weights back.
func (b *Backend) q4K(tw, ts, tb *tensor, groupSize, bits int) *tensor {
	if b.repack == nil || bits != 4 || len(tw.shape) != 2 || check || trace {
		return nil
	}
	P := tw.shape[1]
	in := P * 8
	rows := tw.shape[0]
	if in%256 != 0 || rows%8 != 0 || groupSize <= 0 || in%groupSize != 0 {
		return nil
	}
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	G := ts.shape[len(ts.shape)-1]
	rowBytes := int(C.ggml_row_size(C.GGML_TYPE_Q4_K, C.int64_t(in)))
	raw := make([]byte, rows*rowBytes)
	C.ggml_quantize_init(C.GGML_TYPE_Q4_K)
	const span = 64 // rows a worker dequantizes and quantizes at a time
	var wg sync.WaitGroup
	var next atomic.Int64
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vals := make([]float32, span*in)
			for {
				r0 := int(next.Add(span)) - span
				if r0 >= rows {
					return
				}
				n := min(span, rows-r0)
				for r := r0; r < r0+n; r++ {
					row := vals[(r-r0)*in : (r-r0+1)*in]
					for e := range row {
						q := (uint32(words[r*P+e/8]) >> uint((e%8)*4)) & 0xF
						g := r*G + e/groupSize
						row[e] = float32(q)*sc[g] + bi[g]
					}
				}
				C.ggml_quantize_chunk(C.GGML_TYPE_Q4_K, (*C.float)(unsafe.Pointer(&vals[0])), unsafe.Pointer(&raw[r0*rowBytes]), 0, C.int64_t(n), C.int64_t(in), nil)
			}
		}()
	}
	wg.Wait()
	shape := []int{rows, in}
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q4_K, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors_from_buft(ctx, b.repack)
	if buf == nil {
		C.ggml_free(ctx)
		return nil
	}
	logAlloc("Q4_K weight", C.ggml_nbytes(t), shape)
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
	out := b.op(engine.F32, shape, []*tensor{tx, q, as(w), as(scales), as(biases)}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_mul_mat(ctx, q.t, f32(ctx, tx)) // ne (out, M, …)
	})
	out.quant = &quantCall{groupSize: groupSize, bits: bits}
	return out
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

// Conv1dDepthwise is ggml_ssm_conv, the causal depthwise convolution Mamba
// and the Qwen3.5 hybrid share: x [B, L, C] with one filter per channel,
// w [C, K, 1], stride 1, no padding -> [B, L-K+1, C] in f32. ggml wants the
// input with time innermost ([L, C, B] in its order) and the filters as [K, C].
// The CUDA kernel takes the channels 128 a block and asserts C is a multiple
// of 128 (the hybrid's packed projection is 10240 wide on the 27B).
func (b *Backend) Conv1dDepthwise(x, w engine.Tensor) engine.Tensor {
	tx, tw := as(x), as(w)
	B, L, Cn := tx.shape[0], tx.shape[1], tx.shape[2]
	K := tw.shape[1]
	shape := []int{B, L - K + 1, Cn}
	return b.op(engine.F32, shape, []*tensor{tx, tw}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		xs := C.ggml_reshape_3d(ctx, C.ggml_cont(ctx, f32(ctx, tx)), C.int64_t(Cn), C.int64_t(L), C.int64_t(B))
		sx := C.ggml_cont(ctx, C.ggml_transpose(ctx, xs))
		c := C.ggml_reshape_2d(ctx, C.ggml_cont(ctx, f32(ctx, tw)), C.int64_t(K), C.int64_t(Cn))
		return C.ggml_ssm_conv(ctx, sx, c)
	})
}

// GatedDeltaScan is ggml_gated_delta_net, one fused node for the whole
// recurrence (a CUDA kernel for head sizes 16, 32, 64 and 128; the CPU op for
// any). The contract is engine.Backend's; three things differ from ggml's
// and are translated here:
//   - ggml exponentiates the gate itself, so it is given log g;
//   - ggml scales the output by 1/sqrt(Dv), so q is given times sqrt(Dv)
//     (q never enters the state, the state is untouched);
//   - ggml maps value head h to key head h mod Hk, the contract to h / (Hv/Hk):
//     q and k are repeated to Hv heads in the contract's order first.
//
// The node's result packs y [B, T, Hv, Dv] and the new state [B, Hv, Dv, Dk]
// one after the other, in the layouts the contract names; they are cut out as
// two views.
func (b *Backend) GatedDeltaScan(q, k, v, g, beta, state engine.Tensor) (engine.Tensor, engine.Tensor) {
	tq, tk, tv, tg, tb, ts := as(q), as(k), as(v), as(g), as(beta), as(state)
	B, T, Hk, Dk := tq.shape[0], tq.shape[1], tq.shape[2], tq.shape[3]
	Hv, Dv := tv.shape[2], tv.shape[3]
	if Dk != Dv {
		panic(fmt.Sprintf("ggml: GatedDeltaScan needs Dk == Dv, got %d and %d", Dk, Dv))
	}
	rep := Hv / Hk
	rows := T*B + Dv*B
	res := b.op(engine.F32, []int{rows, Hv * Dv}, []*tensor{tq, tk, tv, tg, tb, ts}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		heads := func(t *tensor) *C.struct_ggml_tensor {
			x := C.ggml_reshape_4d(ctx, C.ggml_cont(ctx, f32(ctx, t)), C.int64_t(Dk), 1, C.int64_t(Hk), C.int64_t(T*B))
			if rep > 1 {
				x = C.ggml_repeat_4d(ctx, x, C.int64_t(Dk), C.int64_t(rep), C.int64_t(Hk), C.int64_t(T*B))
			}
			return C.ggml_reshape_4d(ctx, x, C.int64_t(Dk), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		}
		qh := C.ggml_scale(ctx, heads(tq), C.float(math.Sqrt(float64(Dv))))
		kh := heads(tk)
		vh := C.ggml_reshape_4d(ctx, C.ggml_cont(ctx, f32(ctx, tv)), C.int64_t(Dv), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		gh := C.ggml_reshape_4d(ctx, C.ggml_log(ctx, C.ggml_cont(ctx, f32(ctx, tg))), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		bh := C.ggml_reshape_4d(ctx, C.ggml_cont(ctx, f32(ctx, tb)), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		sh := C.ggml_reshape_4d(ctx, C.ggml_cont(ctx, f32(ctx, ts)), C.int64_t(Dk), C.int64_t(Dv), C.int64_t(Hv), C.int64_t(B))
		return C.ggml_gated_delta_net(ctx, qh, kh, vh, gh, bh, sh)
	})
	y := b.Reshape(b.Slice(res, 0, 0, T*B), B, T, Hv, Dv)
	st := b.Reshape(b.Slice(res, 0, T*B, rows), B, Hv, Dv, Dk)
	return y, st
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
