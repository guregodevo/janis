//go:build ggml

package ggml

/*
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/guregodevo/janis/engine"
)

// A decode step built fresh every token costs more than its compute: on
// an H100 the 27B's step was 34 ms of GPU work in a graph rebuilt from
// 5,464 nodes and about 32 ms of recording, building and laying it out,
// while the same graph run again was 14.9 ms, a CUDA graph replayed
// (2026-11-22). ggml replays a graph only when its nodes, their addresses,
// shapes and parameters are those of the last run, so a recorded step
// takes everything that moves from token to token as an input it writes
// before each run:
//   - the token's embedding row, dequantized from the weight's row;
//   - the RoPE position, a [1] tensor;
//   - where the new keys and values go: set_rows at an index input, the
//     value cache kept transposed written element by element, as
//     llama.cpp does;
//   - how much of the caches attention reads: a view of the caches padded
//     to a bucket of replayBucket tokens, the tokens past the step's
//     position masked by an input;
//   - a recurrent layer's state: copied once into a buffer the step
//     overwrites at its end.

// replayBucket is the token multiple a recorded step reads the caches
// padded to: the step is recorded again each time a cache fills one.
const replayBucket = 256

// replayLog (JANIS_GGML_REPLAYLOG=1) says why a step is not replayed.
var replayLog = os.Getenv("JANIS_GGML_REPLAYLOG") == "1"

// spoil marks the step as one that cannot be replayed.
func (rs *recording) spoil(why string) {
	if rs.ok && replayLog {
		fmt.Fprintf(os.Stderr, "ggml replay: step at %d not replayed: %s\n", rs.offset, why)
	}
	rs.ok = false
}

// callers names the functions that called into a flush, for replayLog.
func callers() string {
	if !replayLog {
		return ""
	}
	pc := make([]uintptr, 8)
	n := runtime.Callers(3, pc)
	fs := runtime.CallersFrames(pc[:n])
	var out []string
	for {
		f, more := fs.Next()
		out = append(out, f.Function[strings.LastIndex(f.Function, "/")+1:])
		if !more {
			break
		}
	}
	return strings.Join(out, " < ")
}

// replayOnCPU lets the CPU backend record steps too, as JANIS_GGML_REPLAY=cpu
// does: for tests, the CPU being deterministic where Metal is not.
var replayOnCPU = false

// replayTest makes BeginStep record whatever the environment says (tests).
var replayTest = false

// padDecode has a plain decode step read the caches the way a recorded one
// does, padded to the bucket and masked: the two then compute the same
// sums in the same order, so a reply is the same whichever of its steps
// were replayed. On where steps are recorded (replayOn), unless
// JANIS_GGML_PADDECODE=0; JANIS_GGML_PADDECODE=1 turns it on anywhere.
// Off, a padded read and an exact one part by rounding.
func (b *Backend) padDecode() bool {
	switch os.Getenv("JANIS_GGML_PADDECODE") {
	case "1":
		return true
	case "0":
		return false
	}
	return b.replayOn()
}

// replayPad is the bucket a recorded step reads key cache k and value
// cache v at, or 0 when they are not cache views.
func (k *tensor) replayPad(v *tensor) int {
	n := k.shape[len(k.shape)-2]
	if k.grp == nil || k.grp.base == nil || v.grp == nil || v.grp.base == nil || k.cacheLen != n || v.cacheLen != n {
		return 0
	}
	return min(k.grp.cap, v.grp.cap, (n-1+replayBucket)/replayBucket*replayBucket)
}

// replayOn is whether this backend records decode steps, read at each
// recording: on CUDA by default (the 27B on an RTX 5090 decodes at 60.6
// tok/s with it, 38.5 without, 2026-11-25), JANIS_GGML_REPLAY=0 turns it
// off; on another GPU (Metal, not measured) JANIS_GGML_REPLAY=1 turns it
// on; on the CPU backend JANIS_GGML_REPLAY=cpu.
func (b *Backend) replayOn() bool {
	switch os.Getenv("JANIS_GGML_REPLAY") {
	case "0":
		return false
	case "1":
		return !b.cpu
	case "cpu":
		return true
	}
	return strings.HasPrefix(b.name, "CUDA")
}

func replayOnCPUNow() bool { return replayOnCPU || os.Getenv("JANIS_GGML_REPLAY") == "cpu" }

// recording is the state of a step being recorded.
type recording struct {
	offset int
	ok     bool // false once the step cannot be replayed
	pos    *tensor
	row    *tensor            // I32 [1]: the token's row in a key cache
	cols   map[[2]int]*tensor // I32 [m] per (capacity, m): the token's elements in a transposed value cache
	masks  map[int]*tensor    // F32 [bucket] per bucket
	limit  int                // the least room of the caches the step writes
	carry  []*tensor
	leaves []*tensor // made while recording: the token input
	embeds []embedRecipe

	// A batched step (BeginBatchStep): row j at offs[j], offset is -1.
	offs   []int
	posB   *tensor            // I32 [B]: each row's RoPE position
	rowsB  []*tensor          // I32 [1] per row: its token's row in a key cache
	colsB  map[[3]int]*tensor // per (row, capacity, m): its elements in a transposed value cache
	masksB map[[2]int]*tensor // F32 [bucket] per (row, bucket)
	limits []int              // per row: the least room of the caches it writes
	cur    int                // the row the cache appends are for (AppendRow)
	// A pool's inputs: the value elements per (room, m), the mask per length.
	poolCols  map[[2]int]*tensor
	poolMasks map[int]*tensor
	// The rows on and off ([B, 1, 1, 1], 1 and 0) of a step with rows
	// held (RowsOn), nil until asked for.
	onB, offB *tensor
	on        []bool
}

// at is the offset the cache appends being recorded write at.
func (rs *recording) at() int {
	if rs.offs != nil {
		return rs.offs[rs.cur]
	}
	return rs.offset
}

// embedRecipe is an embedding row the step reads as a leaf: each run
// dequantizes the row of its token into it.
type embedRecipe struct {
	out             *tensor
	w, s, b         *tensor
	groupSize, bits int
}

// recordEmbed keeps an embedding gather of a step being recorded.
func (b *Backend) recordEmbed(out *tensor, gw, gs, gb *gatherRef, groupSize, bits int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if rs := b.rec0; rs != nil {
		if n := len(gw.ids.host); n != 1 && (rs.offs == nil || n != len(rs.offs)) {
			rs.spoil("an embedding of more than one token")
			return
		}
		rs.embeds = append(rs.embeds, embedRecipe{out: out, w: gw.src, s: gs.src, b: gb.src, groupSize: groupSize, bits: bits})
	}
}

// replayInput is a pinned leaf the step writes before each run.
func (b *Backend) replayInput(dt engine.DType, shape ...int) *tensor {
	t := b.newFrom(dt, shape, nil, 0)
	t.pin = true
	return t
}

func setInt32(t *tensor, v []int32) {
	C.ggml_backend_tensor_set(t.t, unsafe.Pointer(&v[0]), 0, C.size_t(4*len(v)))
}

func setFloat32(t *tensor, v []float32) {
	C.ggml_backend_tensor_set(t.t, unsafe.Pointer(&v[0]), 0, C.size_t(4*len(v)))
}

// BeginStep starts recording a decode step at offset (engine.StepReplayer).
func (b *Backend) BeginStep(offset int, carry []engine.Tensor) []engine.Tensor {
	if !(b.replayOn() || replayTest) || (b.cpu && !replayOnCPUNow()) {
		return nil
	}
	b.flush()
	rs := &recording{offset: offset, ok: true, cols: map[[2]int]*tensor{}, masks: map[int]*tensor{}, limit: math.MaxInt}
	rs.pos = b.replayInput(engine.I32, 1)
	rs.row = b.replayInput(engine.I32, 1)
	moved := make([]engine.Tensor, len(carry))
	for i, c := range carry {
		x := as(c)
		t := b.replayInput(x.dt, x.shape...)
		C.ggml_backend_tensor_copy(x.t, t.t)
		rs.carry = append(rs.carry, t)
		moved[i] = t
	}
	rs.fill(offset)
	b.mu.Lock()
	b.rec0 = rs
	b.mu.Unlock()
	return moved
}

// fill writes the inputs for a step at offset.
func (rs *recording) fill(offset int) {
	setInt32(rs.pos, []int32{int32(offset)})
	setInt32(rs.row, []int32{int32(offset)})
	for k, t := range rs.cols {
		capacity, m := k[0], k[1]
		v := make([]int32, m)
		for i := range v {
			v[i] = int32(offset + capacity*i)
		}
		setInt32(t, v)
	}
	for n, t := range rs.masks {
		v := make([]float32, n)
		for j := offset + 1; j < n; j++ {
			v[j] = float32(math.Inf(-1))
		}
		setFloat32(t, v)
	}
}

// replayCols is the element index input of a transposed value cache of
// capacity tokens written m elements a token. Called with b.mu held.
func (b *Backend) replayCols(capacity, m int) *tensor {
	rs := b.rec0
	if rs.offs != nil {
		return b.replayColsRow(rs.cur, capacity, m)
	}
	k := [2]int{capacity, m}
	if t, ok := rs.cols[k]; ok {
		return t
	}
	b.mu.Unlock()
	t := b.replayInput(engine.I32, m)
	b.mu.Lock()
	rs.cols[k] = t
	setInt32(t, colsAt(rs.offset, capacity, m))
	return t
}

// replayColsRow is replayCols for row j of a batched step. Called with
// b.mu held.
func (b *Backend) replayColsRow(j, capacity, m int) *tensor {
	rs := b.rec0
	k := [3]int{j, capacity, m}
	if t, ok := rs.colsB[k]; ok {
		return t
	}
	b.mu.Unlock()
	t := b.replayInput(engine.I32, m)
	b.mu.Lock()
	rs.colsB[k] = t
	setInt32(t, colsAt(rs.offs[j], capacity, m))
	return t
}

// colsAt is the elements of the token at offset in a transposed value
// cache of capacity tokens, m elements a token.
func colsAt(offset, capacity, m int) []int32 {
	v := make([]int32, m)
	for i := range v {
		v[i] = int32(offset + capacity*i)
	}
	return v
}

// maskAt is a [n] mask: 0 up to offset, -inf past it.
func maskAt(offset, n int) []float32 {
	v := make([]float32, n)
	for j := offset + 1; j < n; j++ {
		v[j] = float32(math.Inf(-1))
	}
	return v
}

// replayMaskRow is replayMask for row j of a batched step. Called without
// b.mu.
func (b *Backend) replayMaskRow(j, n int) *tensor {
	b.mu.Lock()
	rs := b.rec0
	t, ok := rs.masksB[[2]int{j, n}]
	b.mu.Unlock()
	if ok {
		return t
	}
	t = as(b.FromFloats(make([]float32, n), 1, n))
	t.pin = true
	setFloat32(t, maskAt(rs.offs[j], n))
	b.mu.Lock()
	rs.masksB[[2]int{j, n}] = t
	b.mu.Unlock()
	return t
}

// BeginBatchStep starts recording a decode step of len(offsets)
// conversations (engine.BatchStepReplayer).
func (b *Backend) BeginBatchStep(offsets []int, carry []engine.Tensor) []engine.Tensor {
	if !(b.replayOn() || replayTest) || (b.cpu && !replayOnCPUNow()) {
		return nil
	}
	b.flush()
	B := len(offsets)
	rs := &recording{offset: -1, ok: true, cols: map[[2]int]*tensor{}, masks: map[int]*tensor{}, limit: math.MaxInt,
		offs: append([]int(nil), offsets...), colsB: map[[3]int]*tensor{}, masksB: map[[2]int]*tensor{}, limits: make([]int, B),
		poolCols: map[[2]int]*tensor{}, poolMasks: map[int]*tensor{}}
	rs.pos = b.replayInput(engine.I32, 1)
	rs.row = b.replayInput(engine.I32, 1)
	rs.posB = b.replayInput(engine.I32, B)
	for j := range offsets {
		rs.rowsB = append(rs.rowsB, b.replayInput(engine.I32, 1))
		rs.limits[j] = math.MaxInt
	}
	moved := make([]engine.Tensor, len(carry))
	for i, c := range carry {
		x := as(c)
		t := b.replayInput(x.dt, x.shape...)
		C.ggml_backend_tensor_copy(x.t, t.t)
		rs.carry = append(rs.carry, t)
		moved[i] = t
	}
	rs.fillRows(offsets)
	b.mu.Lock()
	b.rec0 = rs
	b.mu.Unlock()
	return moved
}

// AppendRow names the row the next cache appends are for
// (engine.BatchStepReplayer).
func (b *Backend) AppendRow(j int) {
	b.mu.Lock()
	if rs := b.rec0; rs != nil && rs.offs != nil {
		rs.cur = j
	}
	b.mu.Unlock()
}

// fillRows writes a batched step's inputs, row j at offs[j].
func (rs *recording) fillRows(offs []int) {
	copy(rs.offs, offs)
	p := make([]int32, len(offs))
	for j, o := range offs {
		p[j] = int32(o)
		setInt32(rs.rowsB[j], []int32{int32(o)})
	}
	setInt32(rs.posB, p)
	for k, t := range rs.colsB {
		setInt32(t, colsAt(offs[k[0]], k[1], k[2]))
	}
	for k, t := range rs.masksB {
		setFloat32(t, maskAt(offs[k[0]], k[1]))
	}
	for k, t := range rs.poolCols {
		setInt32(t, poolColsAt(offs, k[0], k[1]))
	}
	lens := make([]int, len(offs))
	for j, o := range offs {
		lens[j] = o + 1
	}
	for n, t := range rs.poolMasks {
		setFloat32(t, poolMaskAt(lens, n))
	}
	if rs.onB != nil {
		on, off := rowsOnAt(rs.on)
		setFloat32(rs.onB, on)
		setFloat32(rs.offB, off)
	}
}

// rowsOnAt is 1 for each row on and 0 for each held, and its complement.
func rowsOnAt(rows []bool) (on, off []float32) {
	on, off = make([]float32, len(rows)), make([]float32, len(rows))
	for j, r := range rows {
		if r {
			on[j] = 1
		} else {
			off[j] = 1
		}
	}
	return on, off
}

// RowsOn is the rows on and off of a batched step (engine.RowHolder): the
// step's inputs while it is recorded, else tensors of their own.
func (b *Backend) RowsOn(rows []bool) (engine.Tensor, engine.Tensor) {
	B := len(rows)
	b.mu.Lock()
	rs := b.rec0
	b.mu.Unlock()
	if rs == nil || rs.offs == nil || len(rs.offs) != B {
		on, off := rowsOnAt(rows)
		return b.FromFloats(on, B, 1, 1, 1), b.FromFloats(off, B, 1, 1, 1)
	}
	if rs.onB == nil {
		rs.onB = b.replayInput(engine.F32, B, 1, 1, 1)
		rs.offB = b.replayInput(engine.F32, B, 1, 1, 1)
	}
	rs.on = append(rs.on[:0], rows...)
	on, off := rowsOnAt(rows)
	setFloat32(rs.onB, on)
	setFloat32(rs.offB, off)
	return rs.onB, rs.offB
}

// replayPoolCols is a pool's value element input for a batched step being
// recorded. Called without b.mu.
func (b *Backend) replayPoolCols(room, m int) *tensor {
	b.mu.Lock()
	rs := b.rec0
	t, ok := rs.poolCols[[2]int{room, m}]
	b.mu.Unlock()
	if ok {
		return t
	}
	t = b.replayInput(engine.I32, len(rs.offs)*m)
	setInt32(t, poolColsAt(rs.offs, room, m))
	b.mu.Lock()
	rs.poolCols[[2]int{room, m}] = t
	b.mu.Unlock()
	return t
}

// replayPoolMask is a pool's [B, n] mask input for a batched step being
// recorded. Called without b.mu.
func (b *Backend) replayPoolMask(n int) *tensor {
	b.mu.Lock()
	rs := b.rec0
	t, ok := rs.poolMasks[n]
	b.mu.Unlock()
	if ok {
		return t
	}
	B := len(rs.offs)
	t = b.replayInput(engine.F32, B, 1, 1, n)
	lens := make([]int, B)
	for j, o := range rs.offs {
		lens[j] = o + 1
	}
	setFloat32(t, poolMaskAt(lens, n))
	b.mu.Lock()
	rs.poolMasks[n] = t
	b.mu.Unlock()
	return t
}

// replayMask is the [bucket] mask input: 0 up to the step's position,
// -inf past it. Called without b.mu.
func (b *Backend) replayMask(n int) *tensor {
	b.mu.Lock()
	rs := b.rec0
	t, ok := rs.masks[n]
	b.mu.Unlock()
	if ok {
		return t
	}
	t = as(b.FromFloats(make([]float32, n), 1, n))
	t.pin = true
	v := make([]float32, n)
	for j := rs.offset + 1; j < n; j++ {
		v[j] = float32(math.Inf(-1))
	}
	setFloat32(t, v)
	b.mu.Lock()
	rs.masks[n] = t
	b.mu.Unlock()
	return t
}

// recordAppend is appendCache for one token while a step is recorded:
// the token is written with set_rows at the row input, and the cache is
// read as a view padded to the bucket. ok is false when the cache has no
// room for the step, which is then not replayed. Called with b.mu held.
func (b *Backend) recordAppend(grp *batch, tb *tensor, shape []int, axis int) (*tensor, bool) {
	rs := b.rec0
	n := shape[axis] - 1
	bucket := min(grp.cap, (n+replayBucket)/replayBucket*replayBucket)
	if tb.shape[axis] != 1 || n != rs.at() || bucket <= n {
		return nil, false
	}
	if rs.offs != nil {
		rs.limits[rs.cur] = min(rs.limits[rs.cur], bucket)
	} else {
		rs.limit = min(rs.limit, bucket)
	}
	base := grp.base
	src := tb.t
	var dst, idx *C.struct_ggml_tensor
	if grp.vt {
		// Tokens innermost: element (t, d, h, b) is t + capacity*(d + D*(h + H*b)),
		// and the token's elements in order are its D*H*B values.
		m := numel(tb.shape)
		dst = C.ggml_view_2d(b.rec, base, 1, C.int64_t(numel(shapeOf(base))), base.nb[0], 0)
		C.ggml_backend_view_init(dst)
		src = C.ggml_reshape_2d(b.rec, C.ggml_cont(b.rec, src), 1, C.int64_t(m))
		idx = b.replayCols(grp.cap, m).t
	} else {
		dst = base
		idx = b.rec0.row.t
		if rs.offs != nil {
			idx = rs.rowsB[rs.cur].t
		}
	}
	if tb.dt != engine.F32 {
		src = C.ggml_cast(b.rec, src, C.GGML_TYPE_F32)
	}
	b.extra = append(b.extra, C.ggml_set_rows(b.rec, dst, src, idx))
	b.extraDeps = append(b.extraDeps, tb)
	padded := append([]int(nil), shape...)
	padded[axis] = bucket
	out := &tensor{shape: padded, dt: engine.F16, grp: grp, cacheLen: bucket, replay: true}
	memMu.Lock()
	grp.refs++
	memMu.Unlock()
	out.t = b.cacheView(grp, padded)
	return out, true
}

func shapeOf(t *C.struct_ggml_tensor) []int {
	return []int{int(t.ne[3]), int(t.ne[2]), int(t.ne[1]), int(t.ne[0])}
}

// step is a recorded decode step (engine.Step).
type step struct {
	b      *Backend
	rs     *recording
	ctx    *C.struct_ggml_context
	g      *C.struct_ggml_cgraph
	galloc C.ggml_gallocr_t
	ids    *tensor
	logits *tensor
	vocab  int
	held   []*tensor
}

// EndStep ends the recording and computes the step (engine.StepReplayer).
// AbortStep drops a step being recorded whose recording failed partway
// (out of device memory): the ops that follow are computed as usual.
func (b *Backend) AbortStep() {
	b.mu.Lock()
	rs := b.rec0
	b.rec0 = nil
	b.mu.Unlock()
	if rs != nil {
		rs.free()
	}
}

func (b *Backend) EndStep(ids, logits engine.Tensor, next []engine.Tensor) (engine.Step, []float32) {
	b.mu.Lock()
	rs := b.rec0
	b.rec0 = nil
	if rs == nil || !rs.ok || len(next) != len(rs.carry) || b.rec == nil {
		b.mu.Unlock()
		floats := b.Floats(logits) // computed as recorded, reading the inputs
		if rs != nil {
			rs.free()
		}
		return nil, floats
	}
	ctx, pending, extra, extraDeps, nodes := b.rec, b.pending, b.extra, b.extraDeps, b.recCap
	b.rec, b.pending, b.extra, b.extraDeps = nil, nil, nil, nil
	b.mu.Unlock()

	tl := as(logits)
	// Every result the model still holds is an output, as a flush makes
	// them: a GPU backend fusing two ops skips writing the first one's
	// buffer unless it is flagged one (flushLocked).
	live := []*tensor{tl}
	for _, p := range pending {
		t := p.strong
		if t == nil {
			t = p.wp.Value()
		}
		if t != nil && t != tl {
			live = append(live, t)
		}
	}
	g := C.ggml_new_graph_custom(ctx, C.size_t(nodes), false)
	for _, e := range extra {
		C.ggml_build_forward_expand(g, e)
	}
	for _, t := range live {
		keep(t.t)
		C.ggml_build_forward_expand(g, t.t)
	}
	for i, nx := range next {
		x := as(nx)
		src := x.t
		if x.dt != rs.carry[i].dt {
			src = C.ggml_cast(ctx, src, ggmlType(rs.carry[i].dt))
		}
		C.ggml_build_forward_expand(g, C.ggml_cpy(ctx, src, rs.carry[i].t))
	}
	galloc := C.ggml_gallocr_new(b.buft)
	if !C.ggml_gallocr_alloc_graph(galloc, g) {
		C.ggml_gallocr_free(galloc)
		C.ggml_free(ctx)
		panic("ggml: no room on the device for a recorded decode step")
	}
	s := &step{b: b, rs: rs, ctx: ctx, g: g, galloc: galloc, ids: as(ids), logits: tl, vocab: numel(tl.shape)}
	s.ids.pin = true
	for _, l := range rs.leaves {
		l.pin = true
	}
	// The recorded results stay nodes of the step's graph: no later flush
	// may take them, and the inputs they read stay with the step.
	s.held = append(extraDeps, rs.leaves...)
	return s, s.compute()
}

func (s *step) compute() []float32 {
	t0 := time.Now()
	if st := C.ggml_backend_graph_compute(s.b.be, s.g); st != C.GGML_STATUS_SUCCESS {
		panic("ggml: a recorded decode step failed")
	}
	out := make([]float32, s.vocab)
	t1 := time.Now()
	C.ggml_backend_tensor_get(s.logits.t, unsafe.Pointer(&out[0]), 0, C.size_t(4*s.vocab))
	if timing {
		// JANIS_GGML_TIMING: a replayed step's graph call and its read of
		// the logits, averaged over a hundred steps, with the time between
		// steps (everything outside the two).
		now := time.Now()
		if !replayT.last.IsZero() {
			replayT.gap += t0.Sub(replayT.last)
		}
		replayT.compute, replayT.read, replayT.n, replayT.last = replayT.compute+t1.Sub(t0), replayT.read+now.Sub(t1), replayT.n+1, now
		if replayT.n == 100 {
			fmt.Fprintf(os.Stderr, "ggml replay: 100 steps of %d nodes: graph call %.2f ms, logits read %.2f ms, between steps %.2f ms\n",
				int(C.ggml_graph_n_nodes(s.g)), replayT.compute.Seconds()*10, replayT.read.Seconds()*10, replayT.gap.Seconds()*10)
			if n := int(C.ggml_graph_n_nodes(s.g)); !histSeen[-n] {
				histSeen[-n] = true
				fmt.Fprintf(os.Stderr, "ggml replay: %d nodes by op: %s\n", n, opHist(s.g))
			}
			replayT = replayTiming{}
		}
	}
	return out
}

type replayTiming struct {
	compute, read, gap time.Duration
	n                  int
	last               time.Time
}

var replayT replayTiming

func (s *step) Fits(offset int) bool { return offset < s.rs.limit }

// EndBatchStep ends the recording of a batched step
// (engine.BatchStepReplayer).
func (b *Backend) EndBatchStep(ids, logits engine.Tensor, next []engine.Tensor) (engine.BatchStep, []float32) {
	s, floats := b.EndStep(ids, logits, next)
	if s == nil {
		return nil, floats
	}
	return s.(*step), floats
}

// FitsRows: every row has room in the caches it writes (engine.BatchStep).
func (s *step) FitsRows(offs []int) bool {
	for j, o := range offs {
		if o >= s.rs.limits[j] {
			return false
		}
	}
	return true
}

// RunRowsOn is RunRows with the rows on (engine.HeldStep); a step recorded
// without RowsOn reads none.
func (s *step) RunRowsOn(toks []int32, offs []int, on []bool) []float32 {
	if s.rs.onB != nil {
		s.rs.on = append(s.rs.on[:0], on...)
	}
	return s.RunRows(toks, offs)
}

// RunRows computes the batched step for toks[j] at offs[j] (engine.BatchStep).
func (s *step) RunRows(toks []int32, offs []int) []float32 {
	setInt32(s.ids, toks)
	s.ids.host = append([]int32(nil), toks...)
	for _, e := range s.rs.embeds {
		setFloat32(e.out, dequantRows(e.w, e.s, e.b, toks, e.groupSize, e.bits))
	}
	s.rs.fillRows(offs)
	return s.compute()
}

func (s *step) Run(tok int32, offset int) []float32 {
	setInt32(s.ids, []int32{tok})
	s.ids.host = []int32{tok}
	for _, e := range s.rs.embeds {
		setFloat32(e.out, dequantRows(e.w, e.s, e.b, []int32{tok}, e.groupSize, e.bits))
	}
	s.rs.fill(offset)
	return s.compute()
}

func (s *step) Close() {
	C.ggml_gallocr_free(s.galloc)
	C.ggml_free(s.ctx)
	s.ids.pin = false
	for _, l := range s.rs.leaves {
		l.pin = false
	}
	s.rs.free()
}

// free releases the inputs; the carried states stay with their caches.
func (rs *recording) free() {
	rs.pos.free()
	rs.row.free()
	for _, t := range rs.cols {
		t.free()
	}
	for _, t := range rs.masks {
		t.free()
	}
	if rs.offs != nil {
		rs.posB.free()
		for _, t := range rs.rowsB {
			t.free()
		}
		for _, t := range rs.colsB {
			t.free()
		}
		for _, t := range rs.masksB {
			t.free()
		}
		for _, t := range rs.poolCols {
			t.free()
		}
		for _, t := range rs.poolMasks {
			t.free()
		}
		if rs.onB != nil {
			rs.onB.free()
			rs.offB.free()
		}
	}
}

// CacheView is a cache the recorded step read, viewed at n tokens
// (engine.StepReplayer).
func (b *Backend) CacheView(cache engine.Tensor, n int) engine.Tensor {
	x := as(cache)
	if !x.replay || x.grp == nil || n > x.grp.cap {
		return cache
	}
	shape := padTokens(x.shape, n)
	out := &tensor{shape: shape, dt: x.dt, grp: x.grp, cacheLen: n}
	memMu.Lock()
	x.grp.refs++
	memMu.Unlock()
	b.mu.Lock()
	out.t = b.cacheView(x.grp, shape)
	b.mu.Unlock()
	b.track(out)
	return out
}
