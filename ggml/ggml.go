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
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/ubuntu-x64/lib -lggml -lggml-cpu -lggml-base -Wl,-rpath,${SRCDIR}/third_party/ubuntu-x64/lib
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/third_party/ubuntu-arm64/lib -lggml -lggml-cpu -lggml-base -Wl,-rpath,${SRCDIR}/third_party/ubuntu-arm64/lib
#cgo linux LDFLAGS: -lstdc++ -lm
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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
// janis_dev_free is the free memory of be's device, 0 when it has none to name.
static int64_t janis_dev_free(ggml_backend_t be) {
	ggml_backend_dev_t dev = ggml_backend_get_device(be);
	if (dev == NULL) return 0;
	size_t free = 0, total = 0;
	ggml_backend_dev_memory(dev, &free, &total);
	return (int64_t) free;
}
static void janis_dequant_q4k(const void * x, float * y, int64_t k) {
	ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(x, y, k);
}
// janis_op_hist writes the graph's node count per op, most frequent first
// ("MUL_MAT 520 ADD 410 ..."), for JANIS_GGML_TIMING.
static void janis_op_hist(struct ggml_cgraph * g, char * out, size_t cap) {
	int n[GGML_OP_COUNT] = {0};
	for (int i = 0; i < ggml_graph_n_nodes(g); i++) n[ggml_graph_node(g, i)->op]++;
	size_t at = 0; out[0] = 0;
	for (;;) {
		int best = -1;
		for (int o = 0; o < GGML_OP_COUNT; o++) if (n[o] > 0 && (best < 0 || n[o] > n[best])) best = o;
		if (best < 0 || at + 48 > cap) break;
		at += snprintf(out + at, cap - at, "%s %d ", ggml_op_name((enum ggml_op) best), n[best]);
		n[best] = 0;
	}
}
// janis_keep marks t as a result that outlives its graph. A view has no
// memory of its own: the allocator frees its source once the view's last
// reader in the graph has run, unless the source is marked too.
static void janis_keep(struct ggml_tensor * t) {
	for (; t != NULL; t = t->view_src) ggml_set_output(t);
}
// janis_copy_hist writes, for the graph's copy nodes (CPY, CONT), how many
// there are of each kind: the op, the op of what it copies, the two types
// and the shape ("CONT<PERMUTE f32>f32 128x16x1x1 48; ..."), for
// JANIS_GGML_TIMING.
static void janis_copy_hist(struct ggml_cgraph * g, char * out, size_t cap) {
	enum { K = 96 };
	static char key[K][96]; int n[K]; int used = 0;
	for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
		struct ggml_tensor * t = ggml_graph_node(g, i);
		if (t->op != GGML_OP_CPY && t->op != GGML_OP_CONT) continue;
		struct ggml_tensor * s = t->src[0];
		char k[96];
		snprintf(k, sizeof k, "%s<%s %s>%s %lldx%lldx%lldx%lld", ggml_op_name(t->op), ggml_op_name(s->op), ggml_type_name(s->type), ggml_type_name(t->type),
			(long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
		int j = 0;
		for (; j < used; j++) if (strcmp(key[j], k) == 0) break;
		if (j == used) { if (used == K) continue; strcpy(key[used], k); n[used++] = 0; }
		n[j]++;
	}
	size_t at = 0; out[0] = 0;
	for (;;) {
		int best = -1;
		for (int j = 0; j < used; j++) if (n[j] > 0 && (best < 0 || n[j] > n[best])) best = j;
		if (best < 0 || at + 120 > cap) break;
		at += snprintf(out + at, cap - at, "%s %d; ", key[best], n[best]);
		n[best] = 0;
	}
}
static struct ggml_tensor * janis_new(struct ggml_context * c, enum ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	return ggml_new_tensor_4d(c, t, ne0, ne1, ne2, ne3);
}
// janis_group_sums is the sums of x (ne in, ...) over each group of gs
// inputs: ne (in/gs, ...).
static struct ggml_tensor * janis_group_sums(struct ggml_context * c, struct ggml_tensor * x, int64_t gs) {
	if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
	int64_t in = x->ne[0], rest = ggml_nelements(x) / in;
	struct ggml_tensor * sums = ggml_sum_rows(c, ggml_reshape_3d(c, x, gs, in / gs, rest));
	return ggml_reshape_4d(c, sums, in / gs, x->ne[1], x->ne[2], x->ne[3]);
}
// janis_affine8_input is x (ne in, ...) followed by spread times its group
// sums: what an 8-bit affine weight's code columns and offset columns
// multiply (affine8). spread is F32, ne (in/gs, extra).
static struct ggml_tensor * janis_affine8_input(struct ggml_context * c, struct ggml_tensor * x, struct ggml_tensor * spread, int64_t gs) {
	if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
	return ggml_concat(c, x, ggml_mul_mat(c, spread, janis_group_sums(c, x, gs)), 0);
}
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
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
	// spreads are affine8's spread matrices, by the number of groups.
	spreads map[int]*tensor
	be      C.ggml_backend_t
	buft    C.ggml_backend_buffer_type_t
	// repack is the CPU's weight-repacking buffer type, nil on a device:
	// where a dense 4-bit weight goes as Q4_K (see q4K).
	repack C.ggml_backend_buffer_type_t
	// cpu is whether the backend runs on the cores.
	cpu bool
	// recCap is the node capacity rec was made with (newRec).
	recCap int
	// flash is whether attention is ggml's fused flash attention (SDPA);
	// else the value cache is kept transposed (AppendValues).
	flash bool
	// reserved is the positions a cache is made for (ReserveCache).
	reserved int
	name     string
	mu       sync.Mutex

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
	masks     map[[3]int]*tensor
	// arenas are a few fixed buffers that batches are placed into with a
	// linear allocator and reused once their results are gone: a flush is
	// not a fresh device allocation of yet another size, which is what
	// fragmented the L4's heap until a 100 MB request failed with 1.7 GB
	// nominally free. A batch that does not fit, or finds every arena busy,
	// gets a buffer of its own as before.
	arenas []*arena
	// lagOrder holds, per (heads, kernel), the index that puts a dynamic
	// convolution's taps in the window's order (ConvTapsDyn).
	lagOrder map[[2]int]*tensor
	// caps holds, per (width, gate cap, up cap), the inverses of the caps
	// CappedSwiGLU multiplies both halves by.
	caps map[[3]float32]*tensor
	// zeros holds a sink logit of zero per head count (zeroSinks).
	zeros map[int]*tensor
	// relRows holds a chunk's attention inputs for the pass that made them.
	relRows map[[6]int]*relRows
	relPass [2]int
	// rec0 is the decode step being recorded (replay.go), nil otherwise.
	rec0 *recording
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
// step is one graph, and a graph is one blocking compute with the GPU idle
// while the next is recorded. At 512 a step of the Qwen3.8-27B hybrid
// (about 2,600 results, 5,200 nodes) was six graphs: 13.5 tok/s steady on an
// A6000, 17.5 as one graph (2026-11-24). JANIS_GGML_FLUSH overrides, for
// measuring.
var flushEvery = func() int {
	if n, err := strconv.Atoi(os.Getenv("JANIS_GGML_FLUSH")); err == nil && n > 0 {
		return n
	}
	return 4096
}()

// spreadAbove is the batch buffer size from which a flush considers moving
// its held results out (flushLocked).
const spreadAbove = 256 << 20

// maxNodes bounds one recording context (results and intermediates): a
// result is about two nodes, so flushEvery results leave room to spare.
const maxNodes = 32768

// batchNodes bounds the context a batched step is recorded in: the whole
// step is one graph, and 32 rows of the 27B are about 24,000 nodes.
const batchNodes = 65536

// newRec makes the recording context, the larger one while a batched step
// is recorded. Called with b.mu held.
func (b *Backend) newRec() {
	n := maxNodes
	if rs := b.rec0; rs != nil && rs.offs != nil {
		n = batchNodes
	}
	b.rec = C.janis_ctx(C.size_t(C.ggml_tensor_overhead())*C.size_t(n) + C.ggml_graph_overhead_custom(C.size_t(n), false))
	b.recCap = n
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
		b.cpu = true
		b.flash = flashOnCPU()
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
	b.masks = map[[3]int]*tensor{}
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
	memMu.Lock()
	drainOwned(b.buft)
	memMu.Unlock()
	if b.be != nil {
		C.ggml_backend_free(b.be)
		b.be = nil
	}
}

// ---- tensors ----

// tensor is one ggml tensor in its own buffer. shape is janis's row-major
// shape; ggml's ne is the same list reversed (ne0 innermost).
type tensor struct {
	t    *C.struct_ggml_tensor
	ctx  *C.struct_ggml_context  // a leaf's own context and buffer …
	buf  C.ggml_backend_buffer_t //
	grp  *batch                  // … or the batch a computed result lives in
	deps []*tensor               // the inputs a recorded node reads, held until it is computed
	// via holds what a pending input reads, through that input's own holder
	// rather than the input itself (which may be a temporary the model
	// drops): a residual chain shares its ancestry by reference, not by copy
	// (a copied list doubled every layer and ate 64 GB of host memory on
	// the 27B, 2026-10-26).
	via   *depHolder
	shape []int
	dt    engine.DType
	where string     // the op's call site, kept when tracing or checking
	quant *quantCall // set on a QuantMatmul result: what a dump needs to replay it
	q4    *tensor    // the Q4_1 repack of a packed weight, built once
	// offCols, on an 8-bit weight's Q8_0 codes: the columns after the codes
	// that hold its per-group offsets (affine8), 0 for any other tensor.
	offCols int
	// sums is this tensor followed by what an 8-bit weight's offset columns
	// multiply (affine8Input), for groups of sumsGS values along its last
	// axis: made for the first 8-bit weight that multiplies it and read by
	// the others (q, k and v share an input; so do gate and up).
	sums   *tensor
	sumsGS int
	f32w   *tensor // a small bf16 weight's f32 copy (f32Weight)
	// made holds what was made of a weight once and is kept with it, by
	// what it is for: its transpose for a matmul, a convolution's taps.
	made map[string]*tensor
	// pairRows: a stack of experts whose repack takes the even output rows
	// first, then the odd ones (PairRows).
	pairRows bool
	// input: a recorded step writes it before each run; never a weight.
	input bool
	pin   bool // survives Sweep
	freed bool
	// gathered: rows of this packed weight were taken (the embedding), so
	// its words stay after the repack; a weight only ever multiplied keeps
	// the Q4_1 copy alone.
	gathered bool
	dropped  bool // the words are gone: only QuantMatmul may use it now
	pooled   bool // buf came from ownPool and goes back to it
	replay   bool // a cache view a recorded step reads, padded to its bucket
	// host holds a small I32 leaf's values (token ids, made from Go), and
	// gather marks rows taken from a leaf at such ids: Dequantize reads
	// those rows of the weight itself instead of reading the graph back.
	host     []int32
	gather   *gatherRef
	cacheLen int // a cache view: tokens in use along the token axis
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
	// vt is a value cache laid out tokens innermost, the way attention's
	// v·scores reads it (AppendValues); its views are permuted back.
	vt bool
}

// memMu guards the refcounts and arena state that finalizers touch from
// their own goroutine while the engine records and flushes.
var memMu sync.Mutex

// ownPool keeps the buffers own gave survivors once they are freed, by
// buffer type and size, for the next ones: a decode step pins the same
// caches every token, and a buffer of its own per cache per token is a
// device allocation and a free each (cudaMalloc and cudaFree, which
// synchronize), 4.6 s of the 27B's 19.9 s over 272 tokens on an H100,
// 2026-11-22. At most ownPoolMax bytes are kept; past it a buffer is freed.
// Guarded by memMu.
var ownPool = struct {
	free  map[ownKey][]C.ggml_backend_buffer_t
	bytes C.size_t
}{free: map[ownKey][]C.ggml_backend_buffer_t{}}

type ownKey struct {
	buft C.ggml_backend_buffer_type_t
	size C.size_t
}

const ownPoolMax = 1 << 30

// takeOwned is a buffer of size from the pool, else a new one; nil when
// the device has no room. Called with memMu held.
func takeOwned(buft C.ggml_backend_buffer_type_t, size C.size_t) C.ggml_backend_buffer_t {
	k := ownKey{buft, size}
	if l := ownPool.free[k]; len(l) > 0 {
		buf := l[len(l)-1]
		ownPool.free[k] = l[:len(l)-1]
		ownPool.bytes -= size
		return buf
	}
	return C.ggml_backend_buft_alloc_buffer(buft, size)
}

// giveOwned puts a freed survivor's buffer back. Called with memMu held.
func giveOwned(buf C.ggml_backend_buffer_t) {
	size := C.ggml_backend_buffer_get_size(buf)
	if ownPool.bytes+size > ownPoolMax {
		C.ggml_backend_buffer_free(buf)
		return
	}
	k := ownKey{C.ggml_backend_buffer_get_type(buf), size}
	ownPool.free[k] = append(ownPool.free[k], buf)
	ownPool.bytes += size
}

// drainOwned frees the pooled buffers of buft. Called with memMu held.
func drainOwned(buft C.ggml_backend_buffer_type_t) {
	for k, l := range ownPool.free {
		if k.buft != buft {
			continue
		}
		for _, buf := range l {
			C.ggml_backend_buffer_free(buf)
			ownPool.bytes -= k.size
		}
		delete(ownPool.free, k)
	}
}

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
	site   string // JANIS_GGML_PROF: the backend method that recorded it
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
	if t.freed || t.grp == nil && t.ctx == nil && t.buf == nil && t.q4 == nil && t.f32w == nil {
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
		if t.pooled {
			giveOwned(t.buf)
		} else {
			C.ggml_backend_buffer_free(t.buf)
		}
		t.buf, t.pooled = nil, false
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
	t.sums = nil
	for k, w := range t.made {
		delete(t.made, k)
		memMu.Unlock()
		w.free()
		memMu.Lock()
	}
	if t.f32w != nil {
		w := t.f32w
		t.f32w = nil
		memMu.Unlock()
		w.free()
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
	case engine.U8:
		return C.GGML_TYPE_I8
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
	b.mu.Lock()
	if b.rec0 != nil {
		b.rec0.leaves = append(b.rec0.leaves, out)
	}
	b.mu.Unlock()
	return out
}

// newWeight is a pinned weight of a ggml type (a repacked quantized one),
// filled from raw.
func (b *Backend) newWeight(typ C.enum_ggml_type, shape []int, raw []byte, what string) *tensor {
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, typ, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if buf == nil {
		C.ggml_free(ctx)
		panic(fmt.Sprintf("ggml: no room on the device for the %s %v", what, shape))
	}
	logAlloc(what, C.ggml_nbytes(t), shape)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	return &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16, pin: true}
}

// dropWords frees a weight's packed words once its repack holds them.
func (b *Backend) dropWords(tw *tensor) {
	if !tw.gathered && tw.buf != nil {
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
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
	t := b.newFrom(engine.I32, shape, unsafe.Pointer(&data[0]), len(data)*4)
	if len(data) <= maxHostIDs {
		t.host = append([]int32(nil), data...)
	}
	return t
}

// maxHostIDs is the most token ids a leaf keeps on the host for gathers.
const maxHostIDs = 4096

// gatherRef is rows of a leaf weight taken at ids the host knows.
type gatherRef struct {
	src, ids *tensor
}

// leafRow is row r of a 2-D leaf as float32s, read from the device.
func leafRow(src *tensor, r int) []float32 {
	cols := src.shape[len(src.shape)-1]
	switch src.dt {
	case engine.I32, engine.U32:
		v := make([]int32, cols)
		C.ggml_backend_tensor_get(src.t, unsafe.Pointer(&v[0]), C.size_t(r)*C.size_t(src.t.nb[1]), C.size_t(4*cols))
		out := make([]float32, cols)
		for i, x := range v {
			out[i] = math.Float32frombits(uint32(x)) // the words, bit for bit
		}
		return out
	case engine.F16, engine.BF16:
		v := make([]uint16, cols)
		C.ggml_backend_tensor_get(src.t, unsafe.Pointer(&v[0]), C.size_t(r)*C.size_t(src.t.nb[1]), C.size_t(2*cols))
		out := make([]float32, cols)
		for i, h := range v {
			if src.dt == engine.BF16 {
				out[i] = math.Float32frombits(uint32(h) << 16)
			} else {
				out[i] = f16tof32(h)
			}
		}
		return out
	}
	out := make([]float32, cols)
	C.ggml_backend_tensor_get(src.t, unsafe.Pointer(&out[0]), C.size_t(r)*C.size_t(src.t.nb[1]), C.size_t(4*cols))
	return out
}

// dequantRows is rows of an affine-quantized leaf (words, scales, biases)
// dequantized on the host, one row after the other. A row asked for twice
// is read from the device once, and the rows of a prompt are dequantized on
// every core: read and dequantized one at a time, a 1,530-token prompt of
// the 27B spent 0.3 s here.
func dequantRows(w, sc, bi *tensor, rows []int32, groupSize, bits int) []float32 {
	valsPerWord := 32 / bits
	P := w.shape[len(w.shape)-1]
	in := P * valsPerWord
	out := make([]float32, len(rows)*in)
	type src struct{ words, s, b []float32 }
	first := make(map[int32]int, len(rows)) // a row's first place in rows
	var uniq []int                          // the places read from the device
	srcs := make([]src, len(rows))
	for i, r := range rows {
		if _, ok := first[r]; ok {
			continue
		}
		first[r] = i
		uniq = append(uniq, i)
		srcs[i] = src{leafRow(w, int(r)), leafRow(sc, int(r)), leafRow(bi, int(r))}
	}
	one := func(i int) {
		dequantRow(out[i*in:(i+1)*in], srcs[i].words, srcs[i].s, srcs[i].b, groupSize, bits)
	}
	if workers := min(runtime.NumCPU(), len(uniq)/8); workers > 1 {
		var wg sync.WaitGroup
		for k := 0; k < workers; k++ {
			wg.Add(1)
			go func(k int) {
				defer wg.Done()
				for u := k; u < len(uniq); u += workers {
					one(uniq[u])
				}
			}(k)
		}
		wg.Wait()
	} else {
		for _, i := range uniq {
			one(i)
		}
	}
	for i, r := range rows {
		if f := first[r]; f != i {
			copy(out[i*in:(i+1)*in], out[f*in:(f+1)*in])
		}
	}
	return out
}

// dequantRow is one row's values, q*scale + bias a group: out has a value
// for each of the words' packed codes.
func dequantRow(out, words, s, bb []float32, groupSize, bits int) {
	valsPerWord := 32 / bits
	mask := uint32((1 << bits) - 1)
	if groupSize%valsPerWord != 0 {
		for j := range out {
			q := (math.Float32bits(words[j/valsPerWord]) >> uint((j%valsPerWord)*bits)) & mask
			out[j] = float32(q)*s[j/groupSize] + bb[j/groupSize]
		}
		return
	}
	wordsPerGroup := groupSize / valsPerWord
	for g := 0; g*groupSize < len(out); g++ {
		sg, bg := s[g], bb[g]
		o := out[g*groupSize:]
		for wi, word := range words[g*wordsPerGroup : min((g+1)*wordsPerGroup, len(words))] {
			u := math.Float32bits(word)
			for k := 0; k < valsPerWord; k++ {
				o[wi*valsPerWord+k] = float32((u>>uint(k*bits))&mask)*sg + bg
			}
		}
	}
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
	for _, in := range inputs {
		if in != nil {
			b.f32Weight(in)
		}
	}
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
		b.newRec()
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
	if prof {
		ref.site = profSite()
	}
	b.pending = append(b.pending, ref)
	full := (len(b.pending) >= flushEvery || prof) && b.rec0 == nil
	if full {
		b.flushHeld()
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
	if b.rec0 != nil {
		b.rec0.spoil("a flush mid-step, from " + callers()) // a readback: the step is computed as it comes
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
			keep(p.node)
		}
	}
	g := C.ggml_new_graph_custom(ctx, C.size_t(b.recCap), false)
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
	if !allocGraph(galloc, g) {
		// Out of device memory. Computing the graph over unallocated
		// tensors was a SIGSEGV that took the server down; the server
		// recovers the panic and answers the next request.
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
	if prof && len(pending) == 1 {
		C.ggml_backend_synchronize(b.be)
		e := profAcc[pending[0].site]
		e.n, e.dur, e.bytes = e.n+1, e.dur+time.Since(t2), e.bytes+int64(pending[0].bytes)
		profAcc[pending[0].site] = e
	}
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
		if n := int(C.ggml_graph_n_nodes(g)); n > 500 && !histSeen[n] {
			histSeen[n] = true
			fmt.Fprintf(os.Stderr, "ggml flush: %d nodes by op: %s\n", n, opHist(g))
		}
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
	// A batch's buffer is the graph's peak, and it stays whole while one
	// result in it is held. Where the results held are under half of it
	// and what the rest wastes is more than a sixteenth of the device's
	// free memory, they move to buffers of their own and the batch goes: a prompt chunk of Inkling
	// 2 is three batches, each a few GB at its peak for under a tenth of
	// that held, and the three did not fit together (RTX 5000 Ada 32 GB,
	// 2026-11-27).
	if st == C.GGML_STATUS_SUCCESS && len(live) > 0 && b.lastBatch.bytes >= spreadAbove {
		var held C.size_t
		for _, t := range live {
			held += C.ggml_nbytes(t.t)
		}
		if waste := int64(b.lastBatch.bytes - min(held, b.lastBatch.bytes)); 2*held < b.lastBatch.bytes && 16*waste > b.devFree() {
			for _, t := range live {
				b.ownLocked(t)
			}
		}
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

// allocGraph lays out g's tensors in galloc's buffer; false when the
// device has no room.
func allocGraph(galloc C.ggml_gallocr_t, g *C.struct_ggml_cgraph) bool {
	if failAllocs > 0 {
		failAllocs--
		return false
	}
	return bool(C.ggml_gallocr_alloc_graph(galloc, g))
}

// failAllocs is how many graph layouts fail as on a full device: for tests.
var failAllocs int

// flush computes whatever is recorded (a readback, Eval, Sweep need it).
func (b *Backend) flush() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.flushLocked()
}

// flushHeld is flushLocked for a caller that unlocks b.mu itself after: a
// panic (no room on the device) releases the lock with it, so the server
// that recovers it can still use the backend.
func (b *Backend) flushHeld() {
	defer func() {
		if r := recover(); r != nil {
			b.mu.Unlock()
			panic(r)
		}
	}()
	b.flushLocked()
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
	// histSeen is the graph sizes whose op counts timing already printed.
	histSeen = map[int]bool{}
	// check (JANIS_GGML_CHECK=1) reads every result back after each flush
	// and stops at the first NaN or Inf, naming the op that made it.
	check = os.Getenv("JANIS_GGML_CHECK") == "1"
)

// devFree is the free memory of the backend's device in bytes, 0 when it
// names none.
func (b *Backend) devFree() int64 { return int64(C.janis_dev_free(b.be)) }

// f32 is x as an F32 node in ctx (a cast when it is not).
// contig is x, copied only when it is not contiguous: a cont node copies
// even a contiguous tensor, and the batched recurrent state the scan reads
// is about 100 MB a layer at 32 conversations of the 27B.
func contig(ctx *C.struct_ggml_context, x *C.struct_ggml_tensor) *C.struct_ggml_tensor {
	if C.ggml_is_contiguous(x) {
		return x
	}
	return C.ggml_cont(ctx, x)
}

func f32(ctx *C.struct_ggml_context, x *tensor) *C.struct_ggml_tensor {
	if x.dt == engine.F32 {
		return x.t
	}
	if x.f32w != nil {
		return x.f32w.t
	}
	return C.ggml_cast(ctx, x.t, C.GGML_TYPE_F32)
}

// keep marks t, and what it is a view of, as outliving its graph.
func keep(t *C.struct_ggml_tensor) { C.janis_keep(t) }

// opHist is the graph's node count per op, most frequent first.
func opHist(g *C.struct_ggml_cgraph) string {
	var buf [4096]C.char
	C.janis_op_hist(g, &buf[0], 4096)
	ops := C.GoString(&buf[0])
	C.janis_copy_hist(g, &buf[0], 4096)
	return ops + "\n  copies: " + C.GoString(&buf[0])
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
	b.ownLocked(x)
}

// ownLocked is own for a computed result. Called with b.mu held.
func (b *Backend) ownLocked(x *tensor) {
	if x.grp == nil || x.t == nil || x.grp.base != nil {
		return // a leaf, or a cache view: both already own their memory
	}
	ctx := C.janis_ctx(ctxMem)
	n := ne(x.shape)
	t := C.janis_new(ctx, x.t._type, n[0], n[1], n[2], n[3])
	size := C.ggml_backend_buft_get_alloc_size(b.buft, t)
	memMu.Lock()
	buf := takeOwned(b.buft, size)
	memMu.Unlock()
	if buf == nil {
		C.ggml_free(ctx)
		return // no room for the copy: it stays in its batch
	}
	if C.ggml_backend_tensor_alloc(buf, t, C.ggml_backend_buffer_get_base(buf)) != C.GGML_STATUS_SUCCESS {
		memMu.Lock()
		giveOwned(buf)
		memMu.Unlock()
		C.ggml_free(ctx)
		return
	}
	C.ggml_backend_tensor_copy(x.t, t)
	runtime.KeepAlive(x)
	memMu.Lock()
	x.grp.release()
	memMu.Unlock()
	x.grp, x.t, x.ctx, x.buf, x.pooled = nil, t, ctx, buf, true
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
	if prof {
		profPrint()
	}
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
	if !C.ggml_is_contiguous(x.t) { // a cache view: its bytes in order
		x = as(b.op(x.dt, x.shape, []*tensor{x}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor { return C.ggml_cont(ctx, x.t) }))
		b.flush()
	}
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
		return reshaped(ctx, tx.t, shape)
	})
}

// reshaped is t read in shape. Of what this graph computes it is a view: the
// same values in the same order, and a copy of every one was a fifth of a
// decode step's nodes. Of anything else it is a copy, as a slice is: a
// result holds what it reads only until it is computed, so a view of an
// earlier flush's result would read that result's memory after it was given
// back (Qwen3 on a prompt of 1,500 tokens, 2026-11-26: a CUDA illegal memory
// access, and other text on the CPU).
func reshaped(ctx *C.struct_ggml_context, t *C.struct_ggml_tensor, shape []int) *C.struct_ggml_tensor {
	m := ne(shape)
	v := C.ggml_reshape_4d(ctx, contig(ctx, t), m[0], m[1], m[2], m[3])
	if t.op != C.GGML_OP_NONE {
		return v
	}
	return C.ggml_cont(ctx, v)
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
	if nd > 4 {
		// ggml has four dimensions: the axes of one element are left out,
		// the others permuted, and the result given its full shape.
		var dims, at []int
		for i, d := range tx.shape {
			if d != 1 {
				at = append(at, i)
				dims = append(dims, d)
			}
		}
		if len(dims) > 4 {
			panic(fmt.Sprintf("ggml: Transpose of %v: more than four axes of several elements", tx.shape))
		}
		var order []int
		for _, a := range axes {
			if i := slices.Index(at, a); i >= 0 {
				order = append(order, i)
			}
		}
		return b.Reshape(b.Transpose(b.Reshape(x, dims...), order...), shape...)
	}
	// ggml axis of janis axis i is nd-1-i. Source ggml axis s = nd-1-axes[j]
	// goes to destination ggml axis nd-1-j.
	dst := [4]C.int{0, 1, 2, 3}
	for j, a := range axes {
		dst[nd-1-a] = C.int(nd - 1 - j)
	}
	// Axes of one element may move without the values moving: a token's
	// [1, heads, dim] and [heads, 1, dim] are the same memory. Then it is a
	// reshape, a view; else the permuted copy.
	if sameOrder(tx.shape, axes) {
		return b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
			return reshaped(ctx, tx.t, shape)
		})
	}
	return b.op(tx.dt, shape, []*tensor{tx}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_cont(ctx, C.ggml_permute(ctx, tx.t, dst[0], dst[1], dst[2], dst[3]))
	})
}

// sameOrder reports whether transposing shape by axes leaves the axes of
// more than one element in their order: the values then stay where they are.
func sameOrder(shape []int, axes []int) bool {
	last := -1
	for _, a := range axes {
		if shape[a] == 1 {
			continue
		}
		if a < last {
			return false
		}
		last = a
	}
	return true
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
		return b.appendCache(ta, tb, axis, false)
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
func (b *Backend) appendCache(ta, tb *tensor, axis int, vt bool) engine.Tensor {
	n, m := ta.shape[axis], tb.shape[axis]
	need := n + m
	shape := append([]int(nil), ta.shape...)
	shape[axis] = need
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.rec == nil {
		b.newRec()
	}
	grp := ta.grp
	if b.rec0 != nil {
		if grp != nil && grp.base != nil && ta.cacheLen == n && grp.cap >= need && grp.vt == vt {
			if out, ok := b.recordAppend(grp, tb, shape, axis); ok {
				b.mu.Unlock()
				b.track(out)
				b.mu.Lock()
				return out
			}
		}
		b.rec0.spoil(fmt.Sprintf("a cache append of %d at %d with no room (cap %v)", tb.shape[axis], n, grp != nil && grp.cap >= need))
	}
	if grp == nil || grp.base == nil || ta.cacheLen != n || grp.cap < need || grp.vt != vt {
		// A new buffer with room: the old cache is copied in by the flush.
		spare := reservedSpare(need, b.reserved, numel(shape)/need*2, b.devFree())
		capShape := append([]int(nil), shape...)
		capShape[axis] = (need + spare + kvPad - 1) / kvPad * kvPad
		// The buffer's context also holds every prefix view made over its
		// life (one a step, up to spare of them, two tensors each for a
		// transposed value cache): it must outlive the recording contexts
		// those steps flush.
		ctx := C.janis_ctx(ctxMem + C.size_t(C.ggml_tensor_overhead())*C.size_t(2*(spare+64)))
		cne := ne(capShape)
		if vt {
			cne[0], cne[1] = cne[1], cne[0]
		}
		base := C.janis_new(ctx, C.GGML_TYPE_F16, cne[0], cne[1], cne[2], cne[3])
		buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
		if buf == nil {
			C.ggml_free(ctx)
			panic("ggml: no room on the device for the cache")
		}
		// Zeros past the cache: a prompt chunk reads up to kvPad of them,
		// masked, and a masked garbage NaN would still poison the sum.
		C.ggml_backend_buffer_clear(buf, 0)
		grp = &batch{ctx: ctx, buf: buf, base: base, cap: capShape[axis], vt: vt}
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

// poolBuffer is a cleared f16 cache buffer of shape [B, H, room, D] and a
// view of the whole of it, which holds it.
func (b *Backend) poolBuffer(shape []int, vt bool) *tensor {
	ctx := C.janis_ctx(ctxMem + C.size_t(C.ggml_tensor_overhead())*64)
	cne := ne(shape)
	if vt {
		cne[0], cne[1] = cne[1], cne[0]
	}
	base := C.janis_new(ctx, C.GGML_TYPE_F16, cne[0], cne[1], cne[2], cne[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if buf == nil {
		C.ggml_free(ctx)
		panic("ggml: no room on the device for a key and value pool")
	}
	C.ggml_backend_buffer_clear(buf, 0)
	logAlloc("pool", C.ggml_nbytes(base), shape)
	grp := &batch{ctx: ctx, buf: buf, base: base, cap: shape[2], vt: vt}
	out := &tensor{shape: append([]int(nil), shape...), dt: engine.F16, grp: grp, cacheLen: shape[2], pin: true}
	memMu.Lock()
	grp.refs++
	memMu.Unlock()
	b.mu.Lock()
	out.t = b.cacheView(grp, shape)
	b.mu.Unlock()
	b.track(out)
	return out
}

// reservedSpare is the spare room, in tokens, of a cache buffer that holds
// need tokens at perToken bytes each when the request reserved room for
// reserved tokens: the reservation, unless it is more than a 256th of the
// device's free memory, then what that holds, and never less than
// cacheSpare. A model has many caches (Inkling 2 Mini: 96, 196 KB a token
// between them) and each reserved the whole request: one with max_tokens
// 120,000 asked for 23 GB and was refused ("no room on the device for the
// cache", RTX 5000 Ada 32 GB, 2026-11-27) before it had written a token. A
// cache past its room moves to a larger buffer, as one with no reservation.
func reservedSpare(need, reserved, perToken int, free int64) int {
	spare := cacheSpare(need)
	want := reserved - need
	if want <= spare {
		return spare
	}
	if perToken > 0 && free > 0 {
		if fits := int(free / 256 / int64(perToken)); fits < need+want {
			want = fits - need
		}
	}
	return max(spare, want)
}

// cacheGrow is the least spare room a cache buffer is made with, in tokens.
const cacheGrow = 256

// ReserveCache makes the caches made from now on hold tokens positions from
// the start, the way llama.cpp and Ollama size theirs for the context: the
// prompt is read without a cache being copied. Pages a cache never writes
// are not touched on the CPU.
func (b *Backend) ReserveCache(tokens int) {
	b.mu.Lock()
	b.reserved = tokens
	b.mu.Unlock()
}

// AppendValues is Concat along the token axis for an attention value cache
// (engine.ValueCache). Without flash attention the cache is kept transposed,
// tokens innermost, as llama.cpp keeps it: v·scores reads it in place
// instead of copying the whole cache transposed in every layer.
func (b *Backend) AppendValues(cache, x engine.Tensor) engine.Tensor {
	ta, tb := as(cache), as(x)
	if b.flash || ta.dt != engine.F16 || tb.dt != engine.F16 || len(ta.shape) != 4 {
		return b.Concat(cache, x, 2)
	}
	return b.appendCache(ta, tb, 2, true)
}

// flashOnCPU is whether the CPU backend uses flash attention:
// JANIS_GGML_FLASH=1 or 0, else on x86 (prefill 14.9 → 19.1 tok/s on four
// Xeon cores) and off on ARM, where llama.cpp's prompt reading is faster
// without it (223.6 vs 174.8 tok/s on an M4's ten cores).
func flashOnCPU() bool {
	switch os.Getenv("JANIS_GGML_FLASH") {
	case "1":
		return true
	case "0":
		return false
	}
	return runtime.GOARCH != "arm64"
}

// cacheSpare is the room a cache buffer of need tokens is made with: half
// again, so a long prompt copies its cache a few times in all rather than
// once every other chunk (a fixed 256 copied 7.7 GB over an 8K-token
// prefill of a 1.7B).
func cacheSpare(need int) int {
	return max(cacheGrow, need/2)
}

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
	return cachePrefix(grp.ctx, grp, shape)
}

// cachePrefix is a view, made in ctx, of the first shape[nd-2] tokens of a
// cache buffer, in the cache's logical [B, H, L, D] order.
func cachePrefix(ctx *C.struct_ggml_context, grp *batch, shape []int) *C.struct_ggml_tensor {
	m := ne(shape)
	base := grp.base
	if grp.vt {
		v := C.ggml_view_4d(ctx, base, m[1], m[0], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], 0)
		C.ggml_backend_view_init(v)
		p := C.ggml_permute(ctx, v, 1, 0, 2, 3)
		C.ggml_backend_view_init(p)
		return p
	}
	v := C.ggml_view_4d(ctx, base, m[0], m[1], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], 0)
	C.ggml_backend_view_init(v)
	return v
}

// kvPad is the token multiple a prompt chunk reads the caches padded to.
const kvPad = 32

// paddedLen is the length a cache view can be read padded to, within its
// buffer (whose room past the cache is zeros), or 0 for a tensor that is
// not a cache view.
func (t *tensor) paddedLen() int {
	n := t.cacheLen
	if t.grp == nil || t.grp.base == nil || n == 0 || len(t.shape) != 4 || t.shape[2] != n {
		return 0
	}
	p := (n + kvPad - 1) / kvPad * kvPad
	if p > t.grp.cap {
		return 0
	}
	return p
}

// padTokens is shape with its token axis (nd-2) set to n.
func padTokens(shape []int, n int) []int {
	s := append([]int(nil), shape...)
	s[len(s)-2] = n
	return s
}

// writeCache records the copy of x into the cache buffer at token offset
// off along the token axis (nd-2, ggml's ne1), to run first in the next
// flush.
func (b *Backend) writeCache(grp *batch, x *tensor, off int) {
	m := ne(x.shape)
	base := grp.base
	var dst *C.struct_ggml_tensor
	if grp.vt {
		d := C.ggml_view_4d(b.rec, base, m[1], m[0], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], C.size_t(off)*C.size_t(base.nb[0]))
		C.ggml_backend_view_init(d)
		dst = C.ggml_permute(b.rec, d, 1, 0, 2, 3)
	} else {
		dst = C.ggml_view_4d(b.rec, base, m[0], m[1], m[2], m[3], base.nb[1], base.nb[2], base.nb[3], C.size_t(off)*C.size_t(base.nb[1]))
	}
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
		if C.ggml_is_transposed(t) { // a value cache view (AppendValues)
			t = C.ggml_cont(ctx, t)
		}
		offset := C.size_t(start) * C.size_t(t.nb[g])
		v := C.ggml_view_4d(ctx, t, m[0], m[1], m[2], m[3], t.nb[1], t.nb[2], t.nb[3], offset)
		// A slice of what this graph computes stays a view when its values
		// lie together (q, k and v cut from one projection). A slice of a
		// leaf is copied: a cache is written in place later in the step.
		// The reshape gives the view a fresh tensor's strides: a view keeps
		// its source's on the axes of one element, and a pinned state is
		// copied tensor to tensor, which wants the same layout.
		if t.op != C.GGML_OP_NONE && bool(C.ggml_is_contiguous(v)) {
			return C.ggml_reshape_4d(ctx, v, m[0], m[1], m[2], m[3])
		}
		return C.ggml_cont(ctx, v)
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
	var gather *gatherRef
	if ti.host != nil && ta.grp == nil && ta.t != nil && ta.t.data != nil && len(ta.shape) == 2 {
		gather = &gatherRef{src: ta, ids: ti}
	}
	out := b.op(ta.dt, shape, []*tensor{ta, ti}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
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
	out.gather = gather
	return out
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

// f32Weight gives a small bf16 weight its f32 copy, converted once and kept
// with it (f32 reads it): each was cast again in every step, a copy node
// for each of the 27B's 305 (the norms' weights, the scan's). Only bf16:
// a checkpoint's weights are, and no state or cache is, so what is copied
// never changes.
func (b *Backend) f32Weight(w *tensor) {
	if w.dt != engine.BF16 || w.f32w != nil || w.buf == nil || w.t == nil || numel(w.shape) > 1<<16 {
		return
	}
	n := numel(w.shape)
	raw := make([]uint16, n)
	C.ggml_backend_tensor_get(w.t, unsafe.Pointer(&raw[0]), 0, C.size_t(n*2))
	vals := make([]float32, n)
	for i, h := range raw {
		vals[i] = math.Float32frombits(uint32(h) << 16)
	}
	w.f32w = b.newFrom(engine.F32, w.shape, unsafe.Pointer(&vals[0]), n*4)
	w.f32w.pin = true
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
	if wT := b.transposedWeight(tb); wT != nil {
		// A weight: its transpose was made once, not copied every step.
		return b.op(engine.F32, shape, []*tensor{ta, tb, wT}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
			return C.ggml_mul_mat(ctx, wT.t, f32(ctx, ta))
		})
	}
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
	var tpos *tensor
	b.mu.Lock()
	if rs := b.rec0; rs != nil {
		if L == 1 && offset == rs.offset {
			tpos = rs.pos
		} else {
			rs.spoil(fmt.Sprintf("RoPE of %d at %d", L, offset))
		}
	}
	b.mu.Unlock()
	if tpos == nil {
		tpos = b.positionsFor(offset, L)
	}
	return b.ropeAt(tx, dims, traditional, base, scale, tpos, freqs)
}

// RoPEPositions is RoPE with token i of x [B, H, L, D] at pos[i]
// (engine.RowAttention): a batched step's rows, each at its conversation's
// position.
func (b *Backend) RoPEPositions(x engine.Tensor, dims int, traditional bool, base, scale float32, pos []int) engine.Tensor {
	p := make([]int32, len(pos))
	for i, v := range pos {
		p[i] = int32(v)
	}
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	if rs != nil {
		if rs.offs != nil && len(pos) == len(rs.offs) {
			return b.ropeAt(as(x), dims, traditional, base, scale, rs.posB, nil)
		}
		rs.spoil("RoPE at several positions")
	}
	return b.ropeAt(as(x), dims, traditional, base, scale, as(b.FromInt32(p, len(p))), nil)
}

func (b *Backend) ropeAt(tx *tensor, dims int, traditional bool, base, scale float32, tpos, freqs *tensor) engine.Tensor {
	mode := C.int(C.GGML_ROPE_TYPE_NEOX)
	if traditional {
		mode = C.GGML_ROPE_TYPE_NORMAL
	}
	return b.op(engine.F32, tx.shape, []*tensor{tx, tpos, freqs}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		x := f32(ctx, tx)
		// janis [B,H,L,D] is ne (D,L,H,B); rope wants (D,H,L,B). With one
		// token or one head the two are the same memory: a reshape each
		// way, where the permuted copies were four nodes a layer of a
		// decode step.
		swap := func(t *C.struct_ggml_tensor) *C.struct_ggml_tensor {
			if t.ne[1] == 1 || t.ne[2] == 1 {
				return C.ggml_reshape_4d(ctx, contig(ctx, t), t.ne[0], t.ne[2], t.ne[1], t.ne[3])
			}
			return C.ggml_cont(ctx, C.ggml_permute(ctx, t, 0, 2, 1, 3))
		}
		var fr *C.struct_ggml_tensor
		if freqs != nil {
			fr = f32(ctx, freqs)
		}
		r := C.ggml_rope_ext(ctx, swap(x), tpos.t, fr, C.int(dims), mode, 0, C.float(base), C.float(scale), 0, 1, 32, 1)
		return swap(r)
	})
}

// SDPA over [B, H, L, D] with GQA and the causal mask offset by Lk-Lq. On a
// GPU, and a CPU without flash attention: the classic three ops, scores =
// k·q, softmax with the mask and scale, then v·scores, with ggml's head
// broadcasting doing the GQA share. With flash attention (flashOnCPU): ggml's
// fused kernel, which does not hold the Lq×Lk scores of every head.
func (b *Backend) SDPA(q, k, v engine.Tensor, scale float32, causal bool) engine.Tensor {
	tq, tk, tv := as(q), as(k), as(v)
	B, H, Lq := tq.shape[0], tq.shape[1], tq.shape[2]
	Lk := tk.shape[2]
	Dv := tv.shape[3]
	// A prompt chunk reads the caches padded to kvPad tokens, the padding
	// masked: ggml's f16 GEMM (tinyBLAS) takes only aligned sizes, and an
	// exact Lk fell back to one dot product a row (40% of a prefill).
	Lkp := Lk
	if causal && Lq >= 8 && !b.flash {
		if p := tk.paddedLen(); p > Lk && tv.paddedLen() == p {
			Lkp = p
		}
	}
	var tmask *tensor
	if Lq == 1 && !causal && !tk.replay && b.padDecode() {
		// Read as a recorded step reads: padded to its bucket, masked.
		if p := tk.replayPad(tv); p > Lk {
			Lkp = p
			tmask = b.maskFor(1, Lk, Lkp)
		}
	}
	switch {
	case tk.replay && tv.replay && Lq == 1:
		tmask = b.replayMask(Lk)
	case causal:
		tmask = b.maskFor(Lq, Lk, Lkp)
	}
	shape := []int{B, H, Lq, Dv}
	if b.flash {
		return b.op(engine.F32, shape, []*tensor{tq, tk, tv, tmask}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
			kk, vv := tk.t, tv.t
			if tk.dt != engine.F32 && tk.dt != engine.F16 {
				kk = f32(ctx, tk)
			}
			if tv.dt != engine.F32 && tv.dt != engine.F16 {
				vv = f32(ctx, tv)
			}
			var mask *C.struct_ggml_tensor
			if tmask != nil {
				mask = C.ggml_cast(ctx, tmask.t, C.GGML_TYPE_F16)
			}
			r := C.ggml_flash_attn_ext(ctx, f32(ctx, tq), kk, vv, mask, C.float(scale), 0, 0) // ne (Dv, H, Lq, B)
			return C.ggml_cont(ctx, C.ggml_permute(ctx, r, 0, 2, 1, 3))                       // ne (Dv, Lq, H, B)
		})
	}
	return b.op(engine.F32, shape, []*tensor{tq, tk, tv, tmask}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		// Keys and values may be f16 (the cache): mul_mat takes an f16
		// first operand against f32 as it is, no cast over the whole cache.
		qq := f32(ctx, tq)
		kk, vv := tk.t, tv.t
		if Lkp > Lk {
			kk = cachePrefix(ctx, tk.grp, padTokens(tk.shape, Lkp))
			vv = cachePrefix(ctx, tv.grp, padTokens(tv.shape, Lkp))
		}
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
		// A value cache kept transposed (AppendValues) is read as it is;
		// any other V is copied transposed.
		vT := C.ggml_transpose(ctx, vv) // ne (Lk, Dv, Hk, B)
		if !C.ggml_is_transposed(vv) {
			vT = C.ggml_cont(ctx, vT)
		}
		return C.ggml_mul_mat(ctx, vT, p) // ne (Dv, Lq, H, B)
	})
}

// SDPARows is one decode step's attention for B conversations in one op
// (engine.RowAttention): row j of q [B, H, 1, D] over its own cache's keys
// and values ks[j], vs[j] [1, Hk, Lj, D], as SDPA's three ops read them
// (k·q, softmax with the scale, then v·scores, the value cache read
// transposed in place) -> [B, H, 1, Dv].
func (b *Backend) SDPARows(q engine.Tensor, ks, vs []engine.Tensor, scale float32) engine.Tensor {
	tq := as(q)
	B, H, D := tq.shape[0], tq.shape[1], tq.shape[3]
	Dv := as(vs[0]).shape[3]
	inputs := []*tensor{tq}
	for j := range ks {
		inputs = append(inputs, as(ks[j]), as(vs[j]))
	}
	// A recorded batched step reads each row's caches padded to its bucket,
	// the tokens past the row's position masked by an input.
	masks := make([]*tensor, B)
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	if rs != nil {
		if rs.offs == nil || len(rs.offs) != B {
			rs.spoil("attention over several caches")
		} else {
			for j := range ks {
				if tk, tv := as(ks[j]), as(vs[j]); tk.replay && tv.replay {
					masks[j] = b.replayMaskRow(j, tk.shape[2])
					inputs = append(inputs, masks[j])
				}
			}
		}
	}
	return b.op(engine.F32, []int{B, H, 1, Dv}, inputs, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		qq := C.ggml_cont(ctx, f32(ctx, tq)) // ne (D, 1, H, B)
		var out *C.struct_ggml_tensor
		for j := 0; j < B; j++ {
			qj := C.ggml_view_4d(ctx, qq, C.int64_t(D), 1, C.int64_t(H), 1, qq.nb[1], qq.nb[2], qq.nb[3], C.size_t(j)*qq.nb[3])
			tk, tv := as(ks[j]), as(vs[j])
			kk, vv := tk.t, tv.t
			if tk.dt != engine.F32 && tk.dt != engine.F16 {
				kk = f32(ctx, tk)
			}
			if tv.dt != engine.F32 && tv.dt != engine.F16 {
				vv = f32(ctx, tv)
			}
			kq := C.ggml_mul_mat(ctx, kk, qj) // ne (Lj, 1, H, 1)
			var mask *C.struct_ggml_tensor
			if masks[j] != nil {
				mask = masks[j].t
			}
			p := C.ggml_soft_max_ext(ctx, kq, mask, C.float(scale), 0)
			vT := C.ggml_transpose(ctx, vv)
			if !C.ggml_is_transposed(vv) {
				vT = C.ggml_cont(ctx, vT)
			}
			o := C.ggml_mul_mat(ctx, vT, p) // ne (Dv, 1, H, 1)
			if out == nil {
				out = o
			} else {
				out = C.ggml_concat(ctx, out, o, 3)
			}
		}
		return out // ne (Dv, 1, H, B)
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
		b.flushHeld()
		for _, old := range b.positions {
			old.free()
		}
		b.positions = map[[2]int]*tensor{}
	}
	b.positions[[2]int{offset, L}] = t
	b.mu.Unlock()
	return t
}

// maskFor is the pinned causal mask [Lq, Lkp] (0 kept, -inf above the
// diagonal offset by Lk-Lq, and on the padding past Lk), made once per shape.
func (b *Backend) maskFor(Lq, Lk, Lkp int) *tensor {
	key := [3]int{Lq, Lk, Lkp}
	b.mu.Lock()
	t, ok := b.masks[key]
	b.mu.Unlock()
	if ok {
		return t
	}
	m := make([]float32, Lkp*Lq)
	for i := 0; i < Lq; i++ {
		for j := 0; j < Lkp; j++ {
			if j > i+(Lk-Lq) || j >= Lk {
				m[i*Lkp+j] = float32(math.Inf(-1))
			}
		}
	}
	t = as(b.FromFloats(m, Lq, Lkp))
	t.pin = true
	b.mu.Lock()
	if len(b.masks) > 64 {
		b.flushHeld()
		for _, old := range b.masks {
			old.free()
		}
		b.masks = map[[3]int]*tensor{}
	}
	b.masks[key] = t
	b.mu.Unlock()
	return t
}

// ---- affine 4-bit weights ----

// Dequantize reconstructs the float weight from MLX's (packed words,
// scales, biases): value = q*scale + bias per group. Host-side, as the CPU
// backend does; the matmul path never needs it.
func (b *Backend) Dequantize(w, scales, biases engine.Tensor, groupSize, bits int) engine.Tensor {
	tw := as(w)
	if gw, gs, gb := tw.gather, as(scales).gather, as(biases).gather; gw != nil && gs != nil && gb != nil && gw.ids == gs.ids && gs.ids == gb.ids {
		// Rows of the embedding at known ids: read from the weight, with
		// no readback of the graph (a flush in every decode step).
		shape := append([]int(nil), tw.shape...)
		shape[len(shape)-1] = tw.shape[len(shape)-1] * 32 / bits
		out := as(b.FromFloats(dequantRows(gw.src, gs.src, gb.src, gw.ids.host, groupSize, bits), shape...))
		b.recordEmbed(out, gw, gs, gb, groupSize, bits)
		return out
	}
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
// again into Q4_K (packQ4KRow: sub-blocks of 32 with their own scale and
// min, the same affine form), so the result is within Q4_K's rounding of
// the source. nil when this weight cannot go there: not on the
// CPU, not 4-bit, not 2D, rows not a multiple of 8 or a row not of whole
// 256-value blocks, or the check and trace modes, which read weights back.
func (b *Backend) q4K(tw, ts, tb *tensor, groupSize, bits int) *tensor {
	if b.repack == nil || bits != 4 || len(tw.shape) != 2 || check || trace {
		return nil
	}
	P := tw.shape[1]
	in := P * 8
	rows := tw.shape[0]
	if in%256 != 0 || rows%8 != 0 || groupSize <= 0 || groupSize%32 != 0 || in%groupSize != 0 {
		return nil
	}
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	G := ts.shape[len(ts.shape)-1]
	rowBytes := in / 256 * q4KBlock
	raw := make([]byte, rows*rowBytes)
	var wg sync.WaitGroup
	var next atomic.Int64
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				r := int(next.Add(1)) - 1
				if r >= rows {
					return
				}
				packQ4KRow(words[r*P:(r+1)*P], sc[r*G:(r+1)*G], bi[r*G:(r+1)*G], groupSize, raw[r*rowBytes:(r+1)*rowBytes])
			}
		}()
	}
	wg.Wait()
	shape := []int{rows, in}
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q4_K, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors_from_buft(ctx, b.repack)
	if buf != nil && t.extra == nil {
		// No kernel for this weight on this CPU (an ARM build without
		// dotprod): ggml's set would dereference the missing one.
		C.ggml_backend_buffer_free(buf)
		buf = nil
	}
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

// q4KBlock is sizeof(block_q4_K): d and dmin (f16), twelve bytes of 6-bit
// scales and mins, 128 bytes of 4-bit codes for 256 values.
const q4KBlock = 144

// packQ4KRow writes one row of an MLX affine 4-bit weight (eight codes a
// word; a scale and a bias per group of groupSize, a multiple of 32) as
// block_q4_K, one per 256 values. A sub-block of 32 lies in one group, so
// it holds at most the sixteen values s·q + b: Q4_K's scale and min for it
// are that grid's (the min clamped to at most zero, as Q4_K stores it:
// y = d·sc·q − dmin·m with m ≥ 0), rounded to 6 bits against the block's
// largest, and each old code maps to its new one through a table of
// sixteen: no arithmetic per value. ggml's reference quantizer searches
// twenty scales per sub-block from the dequantized floats instead (281 s
// for an 8B on a GitHub runner's four cores).
func packQ4KRow(words []int32, sc, bi []float32, groupSize int, out []byte) {
	for k := 0; k < len(words)*8/256; k++ {
		blk := out[k*q4KBlock : (k+1)*q4KBlock]
		var grid [8][2]float32 // per sub-block: s and b of its group
		var scales, mins [8]float32
		var maxScale, maxMin float32
		for j := 0; j < 8; j++ {
			g := (k*256 + 32*j) / groupSize
			s, b := sc[g], bi[g]
			grid[j] = [2]float32{s, b}
			lo, hi := min(b, b+15*s), max(b, b+15*s)
			lo = min(lo, 0)
			scales[j], mins[j] = (hi-lo)/15, -lo
			maxScale, maxMin = max(maxScale, scales[j]), max(maxMin, mins[j])
		}
		var invScale, invMin float32
		if maxScale > 0 {
			invScale = 63 / maxScale
		}
		if maxMin > 0 {
			invMin = 63 / maxMin
		}
		sb := blk[4:16]
		clear(sb)
		var ls, lm [8]uint8
		for j := 0; j < 8; j++ {
			ls[j] = uint8(min(63, int(invScale*scales[j]+0.5)))
			lm[j] = uint8(min(63, int(invMin*mins[j]+0.5)))
			if j < 4 {
				sb[j], sb[j+4] = ls[j], lm[j]
			} else {
				sb[j+4] = ls[j]&0xF | (lm[j]&0xF)<<4
				sb[j-4] |= (ls[j] >> 4) << 6
				sb[j] |= (lm[j] >> 4) << 6
			}
		}
		dh, mh := f32tof16(maxScale/63), f32tof16(maxMin/63)
		binary.LittleEndian.PutUint16(blk[0:], dh)
		binary.LittleEndian.PutUint16(blk[2:], mh)
		d, dmin := f16tof32(dh), f16tof32(mh)
		var L [256]uint8
		for j := 0; j < 8; j++ {
			dj := d * float32(ls[j])
			if dj == 0 {
				continue
			}
			mj := dmin * float32(lm[j])
			var lut [16]uint8
			for q := range lut {
				v := (grid[j][0]*float32(q) + grid[j][1] + mj) / dj
				lut[q] = uint8(max(0, min(15, int(v+0.5))))
			}
			w := words[(k*256+32*j)/8:]
			for i := 0; i < 32; i++ {
				L[32*j+i] = lut[(uint32(w[i/8])>>uint((i%8)*4))&0xF]
			}
		}
		qs := blk[16 : 16+128]
		for c := 0; c < 4; c++ {
			for l := 0; l < 32; l++ {
				qs[32*c+l] = L[64*c+l] | L[64*c+l+32]<<4
			}
		}
	}
}

// unpackQ4K is ggml's own dequantization of blocks packQ4K wrote (for the
// tests: cgo is not allowed in a test file).
func unpackQ4K(raw []byte, n int) []float32 {
	out := make([]float32, n)
	C.janis_dequant_q4k(unsafe.Pointer(&raw[0]), (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(n))
	return out
}

// affine8On keeps an 8-bit weight as 8 bits on the device (affine8).
// JANIS_GGML_AFFINE8=0 keeps it as f16, as before.
var affine8On = os.Getenv("JANIS_GGML_AFFINE8") != "0"

// affine8 is the device form of an 8-bit affine weight [out, in]: one ggml
// Q8_0 matrix (blocks of 32: f16 scale d, 32 int8) of its codes and, after
// them, its offsets. A value is code*scale + bias = scale*(code-128) +
// (bias + 128*scale). The first term is a Q8_0 block with d the group's scale
// and the code moved to int8, exact in the codes. The second is one number a
// group, times the sum of the input over that group: it is written in
// affine8Digits columns of the same matrix, as int8 digits in base 128 of the
// row's offset scale, and those columns multiply the group sums, the sums
// over 128 and the sums over 128^2, which follow the input (affine8Input).
// An offset is held to 2^-21 of its row's largest. One multiply a weight:
// kept f32 beside the codes, the offsets were a multiply and an add more a
// weight, a fifth of the 27B's decode step. 8.5 bits a weight and 0.4 of
// offsets, where the f16 copy was 16: Qwen3.8-27B-8bit holds 54 GB as f16.
// Built once on every core, pinned; the words go.
const affine8Digits = 3

// affine8Cols is the offset columns of a weight of G groups: whole blocks.
func affine8Cols(G int) int { return (affine8Digits*G + 31) / 32 * 32 }

func (b *Backend) affine8(tw, ts, tb *tensor, groupSize int) *tensor {
	if tw.q4 != nil {
		return tw.q4
	}
	const bits, valsPerWord, blk = 8, 4, 34 // sizeof(block_q8_0): 2+32
	words := b.Ints(tw)
	sc, bi := b.Floats(ts), b.Floats(tb)
	P := tw.shape[len(tw.shape)-1]
	in := P * valsPerWord
	G := ts.shape[len(ts.shape)-1]
	rows := numel(tw.shape) / P
	extra := affine8Cols(G)
	cols := in + extra
	raw := make([]byte, rows*(cols/32)*blk)
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			offs := make([]float64, G)
			digits := make([]int8, extra)
			for r := w; r < rows; r += workers {
				row := raw[r*(cols/32)*blk:]
				for blkIdx := 0; blkIdx < in/32; blkIdx++ {
					o := blkIdx * blk
					binary.LittleEndian.PutUint16(row[o:], f32tof16(sc[r*G+blkIdx*32/groupSize]))
					for j := 0; j < 32; j++ {
						e := blkIdx*32 + j
						code := (uint32(words[r*P+e/valsPerWord]) >> uint((e%valsPerWord)*bits)) & 0xff
						row[o+2+j] = byte(int8(int(code) - 128))
					}
				}
				// The block's scale is the group's, as f16: the offset is taken
				// against that f16 value, so code 128 is the bias exactly.
				var most float64
				for g := 0; g < G; g++ {
					offs[g] = float64(bi[r*G+g]) + 128*float64(f16tof32(f32tof16(sc[r*G+g])))
					most = math.Max(most, math.Abs(offs[g]))
				}
				d16 := offsetScale(most)
				d := float64(f16tof32(d16))
				clear(digits)
				for g := 0; g < G && d > 0; g++ {
					for j, c := range offsetDigits(offs[g] / d) {
						digits[j*G+g] = c
					}
				}
				for e := 0; e < extra; e += 32 {
					o := (in + e) / 32 * blk
					binary.LittleEndian.PutUint16(row[o:], d16)
					for j := 0; j < 32; j++ {
						row[o+2+j] = byte(digits[e+j])
					}
				}
			}
		}(w)
	}
	wg.Wait()
	shape := append([]int(nil), tw.shape...)
	shape[len(shape)-1] = cols
	ctx := C.janis_ctx(ctxMem)
	n := ne(shape)
	t := C.janis_new(ctx, C.GGML_TYPE_Q8_0, n[0], n[1], n[2], n[3])
	buf := C.ggml_backend_alloc_ctx_tensors(ctx, b.be)
	if buf == nil {
		C.ggml_free(ctx)
		panic(fmt.Sprintf("ggml: no room on the device for an 8-bit weight of %v", shape))
	}
	logAlloc("Q8_0 weight (affine8)", C.ggml_nbytes(t), shape)
	C.ggml_backend_tensor_set(t, unsafe.Pointer(&raw[0]), 0, C.size_t(len(raw)))
	q := &tensor{t: t, ctx: ctx, buf: buf, shape: shape, dt: engine.F16, pin: true, offCols: extra}
	tw.q4 = q
	if !tw.gathered && tw.buf != nil {
		C.ggml_backend_buffer_free(tw.buf)
		C.ggml_free(tw.ctx)
		tw.buf, tw.ctx, tw.t, tw.dropped = nil, nil, nil, true
	}
	return q
}

// offsetDigits is v, within ±127, as affine8Digits int8 digits in base 128,
// the first the units: v is within 128^-2/2 of their sum.
func offsetDigits(v float64) (digits [affine8Digits]int8) {
	for j := range digits {
		c := math.Max(-127, math.Min(127, math.Round(v)))
		digits[j] = int8(c)
		v = (v - c) * 128
	}
	return digits
}

// offsetScale is the f16 scale of a row's offset digits: the least whose
// 127 reaches most, the row's largest offset.
func offsetScale(most float64) uint16 {
	d := f32tof16(float32(most / 127))
	for float64(f16tof32(d))*127 < most {
		d++
	}
	return d
}

// affine8Input is x followed by what an 8-bit weight's offset columns
// multiply: x's sums over each group of gs values, times 1, 1/128 and
// 1/128^2, and zeros to a whole block. One input's is made once.
func (b *Backend) affine8Input(x *tensor, gs int) *tensor {
	if s := x.sums; s != nil && !s.freed && x.sumsGS == gs {
		return s
	}
	in := x.shape[len(x.shape)-1]
	G := in / gs
	extra := affine8Cols(G)
	b.mu.Lock()
	spread := b.spreads[G]
	b.mu.Unlock()
	if spread == nil {
		m := make([]float32, extra*G)
		for j, f := 0, float32(1); j < affine8Digits; j, f = j+1, f/128 {
			for g := 0; g < G; g++ {
				m[(j*G+g)*G+g] = f
			}
		}
		spread = b.newFrom(engine.F32, []int{extra, G}, unsafe.Pointer(&m[0]), len(m)*4)
		spread.pin = true
		b.mu.Lock()
		if b.spreads == nil {
			b.spreads = map[int]*tensor{}
		}
		b.spreads[G] = spread
		b.mu.Unlock()
	}
	shape := append([]int(nil), x.shape...)
	shape[len(shape)-1] = in + extra
	s := b.op(engine.F32, shape, []*tensor{x, spread}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.janis_affine8_input(ctx, f32(ctx, x), spread.t, C.int64_t(gs))
	})
	x.sums, x.sumsGS = s, gs
	return s
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
	var q *tensor
	if tw := as(w); affine8On && bits == 8 && groupSize%32 == 0 && len(tw.shape) == 2 {
		q = b.affine8(tw, as(scales), as(biases), groupSize)
	} else {
		q = b.packed(tx, tw, as(scales), as(biases), groupSize, bits)
	}
	nx := len(tx.shape)
	shape := append(append([]int(nil), tx.shape[:nx-1]...), q.shape[0])
	inputs := []*tensor{tx, q, as(w), as(scales), as(biases)}
	in := tx
	if q.offCols > 0 {
		in = b.affine8Input(tx, groupSize)
		inputs = append(inputs, in)
	}
	out := b.op(engine.F32, shape, inputs, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		return C.ggml_mul_mat(ctx, q.t, f32(ctx, in)) // ne (out, M, …)
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
		xs := C.ggml_reshape_3d(ctx, contig(ctx, f32(ctx, tx)), C.int64_t(Cn), C.int64_t(L), C.int64_t(B))
		sx := C.ggml_cont(ctx, C.ggml_transpose(ctx, xs))
		c := C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tw)), C.int64_t(K), C.int64_t(Cn))
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
			x := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, t)), C.int64_t(Dk), 1, C.int64_t(Hk), C.int64_t(T*B))
			if rep > 1 {
				x = C.ggml_repeat_4d(ctx, x, C.int64_t(Dk), C.int64_t(rep), C.int64_t(Hk), C.int64_t(T*B))
			}
			return C.ggml_reshape_4d(ctx, x, C.int64_t(Dk), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		}
		qh := C.ggml_scale(ctx, heads(tq), C.float(math.Sqrt(float64(Dv))))
		kh := heads(tk)
		vh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, tv)), C.int64_t(Dv), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		gh := C.ggml_reshape_4d(ctx, C.ggml_log(ctx, contig(ctx, f32(ctx, tg))), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		bh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, tb)), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		sh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, ts)), C.int64_t(Dk), C.int64_t(Dv), C.int64_t(Hv), C.int64_t(B))
		return C.ggml_gated_delta_net(ctx, qh, kh, vh, gh, bh, sh)
	})
	y := b.Reshape(b.Slice(res, 0, 0, T*B), B, T, Hv, Dv)
	st := b.Reshape(b.Slice(res, 0, T*B, rows), B, Hv, Dv, Dk)
	return y, st
}

// GatedDeltaConvScan is GatedDeltaScan read from the conv's output
// (engine.DeltaConvScanner). q and k are normalized, scaled and repeated to Hv
// heads as one tensor: both end up times D^-1/2 (q's D^-1 from the reference
// times the D^1/2 that undoes ggml's output scale), and the gate arrives as
// its log, which is what ggml takes. Eight nodes fewer a layer than the
// separate ops; on a decode step q, k and v are views of the conv's output.
func (b *Backend) GatedDeltaConvScan(conv, logG, beta, state engine.Tensor, Hk, Hv, D int) (engine.Tensor, engine.Tensor) {
	tc, tg, tb, ts := as(conv), as(logG), as(beta), as(state)
	B, T, Cn := tc.shape[0], tc.shape[1], tc.shape[2]
	keys := 2 * Hk * D
	if Cn != keys+Hv*D || Hv%Hk != 0 {
		panic(fmt.Sprintf("ggml: GatedDeltaConvScan: %d channels for %d key heads and %d value heads of %d", Cn, Hk, Hv, D))
	}
	rep := Hv / Hk
	n := C.int64_t(T * B)
	rows := T*B + D*B
	res := b.op(engine.F32, []int{rows, Hv * D}, []*tensor{tc, tg, tb, ts}, func(ctx *C.struct_ggml_context) *C.struct_ggml_tensor {
		x := C.ggml_reshape_2d(ctx, contig(ctx, f32(ctx, tc)), C.int64_t(Cn), n)
		qk := contig(ctx, C.ggml_view_2d(ctx, x, C.int64_t(keys), n, x.nb[1], 0))
		qk = C.ggml_reshape_4d(ctx, qk, C.int64_t(D), 1, C.int64_t(2*Hk), n)
		qk = C.ggml_scale(ctx, C.ggml_rms_norm(ctx, qk, 1e-6), C.float(1/math.Sqrt(float64(D))))
		if rep > 1 {
			qk = C.ggml_repeat_4d(ctx, qk, C.int64_t(D), C.int64_t(rep), C.int64_t(2*Hk), n)
		}
		qk = C.ggml_reshape_3d(ctx, qk, C.int64_t(D), C.int64_t(2*Hv), n)
		half := func(off int) *C.struct_ggml_tensor {
			h := contig(ctx, C.ggml_view_3d(ctx, qk, C.int64_t(D), C.int64_t(Hv), n, qk.nb[1], qk.nb[2], C.size_t(off)*C.size_t(qk.nb[1])))
			return C.ggml_reshape_4d(ctx, h, C.int64_t(D), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		}
		v := contig(ctx, C.ggml_view_2d(ctx, x, C.int64_t(Hv*D), n, x.nb[1], C.size_t(keys)*C.size_t(x.nb[0])))
		vh := C.ggml_reshape_4d(ctx, v, C.int64_t(D), C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		gh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, tg)), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		bh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, tb)), 1, C.int64_t(Hv), C.int64_t(T), C.int64_t(B))
		sh := C.ggml_reshape_4d(ctx, contig(ctx, f32(ctx, ts)), C.int64_t(D), C.int64_t(D), C.int64_t(Hv), C.int64_t(B))
		return C.ggml_gated_delta_net(ctx, half(0), half(Hv), vh, gh, bh, sh)
	})
	y := b.Reshape(b.Slice(res, 0, 0, T*B), B, T, Hv, D)
	st := b.Reshape(b.Slice(res, 0, T*B, rows), B, Hv, D, D)
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
