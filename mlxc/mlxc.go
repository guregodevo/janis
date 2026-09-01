//go:build darwin && arm64 && cgo

// Package mlxc implements engine.Backend on top of MLX's C API (mlx-c),
// linking libmlxc/libmlx from Homebrew. It is the only package that calls
// into C — everything above it speaks the engine interface. Apple-Silicon only;
// on other platforms the pure-Go llm/cpu backend is used instead.
package mlxc

/*
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -lmlxc -lmlx -lc++ -Wl,-rpath,/opt/homebrew/lib -framework Foundation -framework Metal -framework Accelerate
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"os"
	"strconv"
	"unsafe"

	"memdoor/llm/engine"
)

// tensor wraps an mlx_array handle. shape is tracked Go-side; pinned protects it
// from Sweep; backing pins host bytes for arrays built from Go memory.
type tensor struct {
	arr     C.mlx_array
	shape   []int
	pinned  bool
	backing []byte
}

func (t *tensor) Shape() []int { return t.shape }

// Backend holds the default GPU stream and tracks created arrays for sweeping.
type Backend struct {
	stream  C.mlx_stream
	tracked []*tensor
}

// New constructs an MLX-C backend bound to the default GPU (Metal) stream.
func New() *Backend {
	b := &Backend{stream: C.mlx_default_gpu_stream_new()}
	// Bound MLX's buffer-reuse cache. By default it's UNBOUNDED: freed Metal
	// buffers are retained for reuse keyed by size, so a workload with varying
	// tensor sizes (different prompt / page lengths) accumulates a buffer per
	// size and never returns it to the OS. Measured: the embedder alone grew the
	// cache to ~10GB, which on a 16GB unified-memory Mac exhausts RAM and gets
	// the process jetsam-killed. A modest limit keeps reuse for the hot working
	// set while letting the long tail of one-off sizes go back to the OS.
	// Override with MLX_CACHE_LIMIT_MB (0 disables caching entirely).
	limit := uint64(512) // MB
	if v := os.Getenv("MLX_CACHE_LIMIT_MB"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			limit = n
		}
	}
	b.SetCacheLimit(limit * 1024 * 1024)
	return b
}

// newT wraps and tracks a freshly produced array.
func (b *Backend) newT(arr C.mlx_array, shape []int) *tensor {
	t := &tensor{arr: arr, shape: shape}
	b.tracked = append(b.tracked, t)
	return t
}

// --- lifecycle (engine.Sweeper) ---

func (b *Backend) PinAll() {
	for _, t := range b.tracked {
		t.pinned = true
	}
}

func (b *Backend) Pin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			t.(*tensor).pinned = true
		}
	}
}

func (b *Backend) Unpin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			t.(*tensor).pinned = false
		}
	}
}

// Free immediately releases specific tensors (engine.Freer) — used to drop a
// MoE layer's expert stacks mid-forward. Pinned tensors are left alone; freed
// arrays are nulled so the next Sweep skips them.
func (b *Backend) Free(ts ...engine.Tensor) {
	for _, t := range ts {
		if t == nil {
			continue
		}
		tt := t.(*tensor)
		if !tt.pinned && tt.arr.ctx != nil {
			C.mlx_array_free(tt.arr)
			tt.arr.ctx = nil
		}
	}
}

func (b *Backend) Sweep() {
	kept := b.tracked[:0]
	for _, t := range b.tracked {
		if t.pinned {
			kept = append(kept, t)
		} else if t.arr.ctx != nil {
			C.mlx_array_free(t.arr)
			t.arr.ctx = nil
		}
	}
	b.tracked = kept
}

// --- helpers ---

func mlxDType(dt engine.DType) C.mlx_dtype {
	switch dt {
	case engine.F32:
		return C.MLX_FLOAT32
	case engine.F16:
		return C.MLX_FLOAT16
	case engine.BF16:
		return C.MLX_BFLOAT16
	case engine.I32:
		return C.MLX_INT32
	case engine.U32:
		return C.MLX_UINT32
	default:
		panic("mlxc: unsupported dtype")
	}
}

func cShape(shape []int) ([]C.int, int) {
	cs := make([]C.int, len(shape))
	n := 1
	for i, s := range shape {
		cs[i] = C.int(s)
		n *= s
	}
	return cs, n
}

func cloneShape(shape []int) []int { return append([]int(nil), shape...) }

// matmulShape computes the result shape of a batched matmul a@b with NumPy
// broadcasting on the leading (batch) dims.
func matmulShape(a, b []int) []int {
	m := a[len(a)-2]
	n := b[len(b)-1]
	lead := broadcastShapes(a[:len(a)-2], b[:len(b)-2])
	return append(append(lead, m), n)
}

func broadcastShapes(x, y []int) []int {
	n := len(x)
	if len(y) > n {
		n = len(y)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		xi, yi := 1, 1
		if i < len(x) {
			xi = x[len(x)-1-i]
		}
		if i < len(y) {
			yi = y[len(y)-1-i]
		}
		d := xi
		if yi > d {
			d = yi
		}
		out[n-1-i] = d
	}
	return out
}

func optInt(v int) C.mlx_optional_int {
	o := C.mlx_optional_int{value: C.int(v)}
	o.has_value = true
	return o
}

// --- info ---

func (b *Backend) Version() string {
	s := C.mlx_string_new()
	defer C.mlx_string_free(s)
	C.mlx_version(&s)
	return C.GoString(C.mlx_string_data(s))
}

func (b *Backend) Close() { C.mlx_stream_free(b.stream) }

// PeakMemoryMB reports MLX's peak memory in MB (diagnostic).
func (b *Backend) PeakMemoryMB() float64 {
	var n C.size_t
	C.mlx_get_peak_memory(&n)
	return float64(n) / (1024 * 1024)
}

// ActiveMemoryMB is memory backing live arrays (not the reuse cache).
func (b *Backend) ActiveMemoryMB() float64 {
	var n C.size_t
	C.mlx_get_active_memory(&n)
	return float64(n) / (1024 * 1024)
}

// CacheMemoryMB is memory MLX retains in its buffer-reuse cache after arrays are
// freed. Unbounded by default — it does NOT return to the OS on mlx_array_free,
// so it shows up as resident growth.
func (b *Backend) CacheMemoryMB() float64 {
	var n C.size_t
	C.mlx_get_cache_memory(&n)
	return float64(n) / (1024 * 1024)
}

// ClearCache returns MLX's buffer-reuse cache to the OS.
func (b *Backend) ClearCache() { C.mlx_clear_cache() }

// MemoryStatsMB reports MLX's process-wide active / cache / peak memory in MB.
// These are global (not per-backend), so any caller gets the whole-process view.
func MemoryStatsMB() (active, cache, peak float64) {
	var a, c, p C.size_t
	C.mlx_get_active_memory(&a)
	C.mlx_get_cache_memory(&c)
	C.mlx_get_peak_memory(&p)
	return float64(a) / (1 << 20), float64(c) / (1 << 20), float64(p) / (1 << 20)
}

// SetCacheLimit bounds MLX's buffer-reuse cache (bytes); above it, freed buffers
// go back to the OS instead of being retained. 0 disables caching entirely.
func (b *Backend) SetCacheLimit(bytes uint64) {
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(bytes))
}

// --- constructors ---

func (b *Backend) FromFloats(data []float32, shape ...int) engine.Tensor {
	cs, n := cShape(shape)
	if n != len(data) {
		panic("mlxc.FromFloats: shape does not match data length")
	}
	arr := C.mlx_array_new_data(unsafe.Pointer(&data[0]), &cs[0], C.int(len(cs)), C.MLX_FLOAT32)
	return b.newT(arr, cloneShape(shape))
}

func (b *Backend) FromInt32(data []int32, shape ...int) engine.Tensor {
	cs, n := cShape(shape)
	if n != len(data) {
		panic("mlxc.FromInt32: shape does not match data length")
	}
	arr := C.mlx_array_new_data(unsafe.Pointer(&data[0]), &cs[0], C.int(len(cs)), C.MLX_INT32)
	return b.newT(arr, cloneShape(shape))
}

func (b *Backend) FromRaw(dt engine.DType, raw []byte, shape ...int) engine.Tensor {
	cs, _ := cShape(shape)
	arr := C.mlx_array_new_data(unsafe.Pointer(&raw[0]), &cs[0], C.int(len(cs)), mlxDType(dt))
	t := b.newT(arr, cloneShape(shape))
	t.backing = raw
	return t
}

// --- ops ---

func (b *Backend) Add(x, y engine.Tensor) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	out := C.mlx_array_new()
	C.mlx_add(&out, tx.arr, ty.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Mul(x, y engine.Tensor) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	out := C.mlx_array_new()
	C.mlx_multiply(&out, tx.arr, ty.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) SiLU(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	sig := C.mlx_array_new()
	C.mlx_sigmoid(&sig, tx.arr, b.stream)
	sigT := b.newT(sig, cloneShape(tx.shape))
	out := C.mlx_array_new()
	C.mlx_multiply(&out, tx.arr, sigT.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) RMSNorm(x, weight engine.Tensor, eps float32) engine.Tensor {
	tx, tw := x.(*tensor), weight.(*tensor)
	out := C.mlx_array_new()
	C.mlx_fast_rms_norm(&out, tx.arr, tw.arr, C.float(eps), b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) LayerNorm(x, weight, bias engine.Tensor, eps float32) engine.Tensor {
	tx, tw, tb := x.(*tensor), weight.(*tensor), bias.(*tensor)
	out := C.mlx_array_new()
	C.mlx_fast_layer_norm(&out, tx.arr, tw.arr, tb.arr, C.float(eps), b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Erf(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_erf(&out, tx.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Exp(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_exp(&out, tx.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Sigmoid(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_sigmoid(&out, tx.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

// Softplus is log(1+exp(x)), computed as logaddexp(x, 0) — the stable form
// (mlx.nn.softplus does the same).
func (b *Backend) Softplus(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	zero := b.scalarLike(0, tx.arr)
	out := C.mlx_array_new()
	C.mlx_logaddexp(&out, tx.arr, zero, b.stream)
	C.mlx_array_free(zero)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Conv1dDepthwise(x, w engine.Tensor) engine.Tensor {
	tx, tw := x.(*tensor), w.(*tensor)
	channels := tw.shape[0]
	kernel := tw.shape[1]
	out := C.mlx_array_new()
	C.mlx_conv1d(&out, tx.arr, tw.arr, C.int(1), C.int(0), C.int(1), C.int(channels), b.stream)
	shape := cloneShape(tx.shape)
	shape[1] -= kernel - 1
	return b.newT(out, shape)
}

func (b *Backend) Mean(x engine.Tensor, axis int) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_mean_axis(&out, tx.arr, C.int(axis), C.bool(false), b.stream)
	shape := append(cloneShape(tx.shape[:axis]), tx.shape[axis+1:]...)
	return b.newT(out, shape)
}

func (b *Backend) RoPE(x engine.Tensor, dims int, traditional bool, base, scale float32, offset int) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	baseOpt := C.mlx_optional_float{value: C.float(base)}
	baseOpt.has_value = true
	var nullFreqs C.mlx_array
	C.mlx_fast_rope(&out, tx.arr, C.int(dims), C.bool(traditional), baseOpt, C.float(scale), C.int(offset), nullFreqs, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

// RoPEFreqs applies RoPE with explicit per-dim frequencies (base ignored).
func (b *Backend) RoPEFreqs(x engine.Tensor, dims int, traditional bool, scale float32, offset int, freqs engine.Tensor) engine.Tensor {
	tx, tf := x.(*tensor), freqs.(*tensor)
	out := C.mlx_array_new()
	var noBase C.mlx_optional_float // unset -> use freqs
	C.mlx_fast_rope(&out, tx.arr, C.int(dims), C.bool(traditional), noBase, C.float(scale), C.int(offset), tf.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Reshape(x engine.Tensor, shape ...int) engine.Tensor {
	tx := x.(*tensor)
	cs, _ := cShape(shape)
	out := C.mlx_array_new()
	C.mlx_reshape(&out, tx.arr, &cs[0], C.size_t(len(cs)), b.stream)
	return b.newT(out, cloneShape(shape))
}

// ScatterRows replaces dst's rows at the given leading-axis indices with the
// rows of updates ([len(indices), ...dst.shape[1:]]) in one mlx_scatter_single.
// MLX scatter semantics want updates shaped indices.shape + dst.shape with the
// scattered axis collapsed to 1, so updates reshapes to [M, 1, ...].
func (b *Backend) ScatterRows(dst engine.Tensor, indices []int32, updates engine.Tensor) engine.Tensor {
	td, tu := dst.(*tensor), updates.(*tensor)
	idxShape := []int{len(indices)}
	cs, _ := cShape(idxShape)
	idxArr := C.mlx_array_new_data(unsafe.Pointer(&indices[0]), &cs[0], C.int(len(cs)), C.MLX_INT32)
	defer C.mlx_array_free(idxArr)

	upShape := append([]int{len(indices), 1}, td.shape[1:]...)
	ucs, _ := cShape(upShape)
	reshaped := C.mlx_array_new()
	C.mlx_reshape(&reshaped, tu.arr, &ucs[0], C.size_t(len(ucs)), b.stream)
	defer C.mlx_array_free(reshaped)

	out := C.mlx_array_new()
	C.mlx_scatter_single(&out, td.arr, idxArr, reshaped, 0, b.stream)
	return b.newT(out, cloneShape(td.shape))
}

func (b *Backend) Transpose(x engine.Tensor, axes ...int) engine.Tensor {
	tx := x.(*tensor)
	ca := make([]C.int, len(axes))
	newShape := make([]int, len(axes))
	for i, a := range axes {
		ca[i] = C.int(a)
		newShape[i] = tx.shape[a]
	}
	out := C.mlx_array_new()
	C.mlx_transpose_axes(&out, tx.arr, &ca[0], C.size_t(len(ca)), b.stream)
	return b.newT(out, newShape)
}

func (b *Backend) Concat(x, y engine.Tensor, axis int) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	C.mlx_vector_array_append_value(vec, tx.arr)
	C.mlx_vector_array_append_value(vec, ty.arr)
	out := C.mlx_array_new()
	C.mlx_concatenate_axis(&out, vec, C.int(axis), b.stream)
	shape := cloneShape(tx.shape)
	shape[axis] += ty.shape[axis]
	return b.newT(out, shape)
}

// Slice keeps [start:end] along axis, full extent on the other dims.
func (b *Backend) Slice(x engine.Tensor, axis, start, end int) engine.Tensor {
	tx := x.(*tensor)
	nd := len(tx.shape)
	cstart := make([]C.int, nd)
	stop := make([]C.int, nd)
	strides := make([]C.int, nd)
	newShape := cloneShape(tx.shape)
	for i := 0; i < nd; i++ {
		stop[i] = C.int(tx.shape[i])
		strides[i] = 1
	}
	cstart[axis] = C.int(start)
	stop[axis] = C.int(end)
	newShape[axis] = end - start
	out := C.mlx_array_new()
	C.mlx_slice(&out, tx.arr, &cstart[0], C.size_t(nd), &stop[0], C.size_t(nd), &strides[0], C.size_t(nd), b.stream)
	return b.newT(out, newShape)
}

func (b *Backend) SDPA(q, k, v engine.Tensor, scale float32, causal bool) engine.Tensor {
	tq, tk, tv := q.(*tensor), k.(*tensor), v.(*tensor)
	out := C.mlx_array_new()
	modeStr := ""
	if causal {
		modeStr = "causal"
	}
	mode := C.CString(modeStr)
	defer C.free(unsafe.Pointer(mode))
	var nullMask, nullSinks C.mlx_array
	C.mlx_fast_scaled_dot_product_attention(&out, tq.arr, tk.arr, tv.arr, C.float(scale), mode, nullMask, nullSinks, b.stream)
	return b.newT(out, cloneShape(tq.shape))
}

func (b *Backend) MatMul(x, y engine.Tensor) engine.Tensor {
	tx, ty := x.(*tensor), y.(*tensor)
	out := C.mlx_array_new()
	C.mlx_matmul(&out, tx.arr, ty.arr, b.stream)
	return b.newT(out, matmulShape(tx.shape, ty.shape))
}

func (b *Backend) Cast(x engine.Tensor, dt engine.DType) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_astype(&out, tx.arr, mlxDType(dt), b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Tanh(x engine.Tensor) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_tanh(&out, tx.arr, b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) Softmax(x engine.Tensor, axis int) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_softmax_axis(&out, tx.arr, C.int(axis), C.bool(true), b.stream)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) ExpandDims(x engine.Tensor, axis int) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_expand_dims(&out, tx.arr, C.int(axis), b.stream)
	shape := cloneShape(tx.shape)
	shape = append(shape, 0)
	copy(shape[axis+1:], shape[axis:])
	shape[axis] = 1
	return b.newT(out, shape)
}

// scalarLike builds a scalar array cast to like's dtype, matching MLX's
// weak-scalar promotion (so f16 math stays f16 rather than upcasting to f32).
func (b *Backend) scalarLike(v float32, like C.mlx_array) C.mlx_array {
	s := C.mlx_array_new_float32(C.float(v))
	out := C.mlx_array_new()
	C.mlx_astype(&out, s, C.mlx_array_dtype(like), b.stream)
	C.mlx_array_free(s)
	return out
}

func (b *Backend) ScalarMul(x engine.Tensor, v float32) engine.Tensor {
	tx := x.(*tensor)
	s := b.scalarLike(v, tx.arr)
	out := C.mlx_array_new()
	C.mlx_multiply(&out, tx.arr, s, b.stream)
	C.mlx_array_free(s)
	return b.newT(out, cloneShape(tx.shape))
}

func (b *Backend) ScalarAdd(x engine.Tensor, v float32) engine.Tensor {
	tx := x.(*tensor)
	s := b.scalarLike(v, tx.arr)
	out := C.mlx_array_new()
	C.mlx_add(&out, tx.arr, s, b.stream)
	C.mlx_array_free(s)
	return b.newT(out, cloneShape(tx.shape))
}

// Gelu composes the tanh-approx GELU from primitives:
// 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
func (b *Backend) Gelu(x engine.Tensor) engine.Tensor {
	const c = 0.7978845608028654 // sqrt(2/pi)
	x3 := b.Mul(b.Mul(x, x), x)
	inner := b.Add(x, b.ScalarMul(x3, 0.044715))
	t := b.Tanh(b.ScalarMul(inner, c))
	return b.ScalarMul(b.Mul(x, b.ScalarAdd(t, 1.0)), 0.5)
}

func (b *Backend) TakeAxis(a, indices engine.Tensor, axis int) engine.Tensor {
	ta, ti := a.(*tensor), indices.(*tensor)
	out := C.mlx_array_new()
	C.mlx_take_axis(&out, ta.arr, ti.arr, C.int(axis), b.stream)
	shape := append([]int(nil), ta.shape[:axis]...)
	shape = append(shape, ti.shape...)
	shape = append(shape, ta.shape[axis+1:]...)
	return b.newT(out, shape)
}

func (b *Backend) Dequantize(w, scales, biases engine.Tensor, groupSize, bits int) engine.Tensor {
	tw, ts, tb := w.(*tensor), scales.(*tensor), biases.(*tensor)
	out := C.mlx_array_new()
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	var nullScale C.mlx_array
	// Output in the scales' dtype (the model's native compute dtype: f16 or
	// bf16) rather than MLX's f32 default, so the forward pass runs at the
	// reference precision (critical for bf16 models like Qwen3).
	dtype := C.mlx_optional_dtype{value: C.mlx_array_dtype(ts.arr)}
	dtype.has_value = true
	C.mlx_dequantize(&out, tw.arr, ts.arr, tb.arr, optInt(groupSize), optInt(bits), mode, nullScale, dtype, b.stream)
	shape := cloneShape(tw.shape)
	shape[len(shape)-1] = shape[len(shape)-1] * 32 / bits
	return b.newT(out, shape)
}

func (b *Backend) QuantMatmul(x, w, scales, biases engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	tx, tw, ts, tb := x.(*tensor), w.(*tensor), scales.(*tensor), biases.(*tensor)
	out := C.mlx_array_new()
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	C.mlx_quantized_matmul(&out, tx.arr, tw.arr, ts.arr, tb.arr, C.bool(transpose), optInt(groupSize), optInt(bits), mode, b.stream)
	n := len(tx.shape)
	shape := append(cloneShape(tx.shape[:n-1]), tw.shape[0])
	return b.newT(out, shape)
}

func (b *Backend) GatherQuantMatmul(x, w, scales, biases, rhsIndices engine.Tensor, transpose bool, groupSize, bits int) engine.Tensor {
	tx, tw, ts, tb := x.(*tensor), w.(*tensor), scales.(*tensor), biases.(*tensor)
	tr := rhsIndices.(*tensor)
	out := C.mlx_array_new()
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	var nullArr C.mlx_array // lhs_indices = null (no left gather)
	C.mlx_gather_qmm(&out, tx.arr, tw.arr, ts.arr, tb.arr, nullArr, tr.arr,
		C.bool(transpose), optInt(groupSize), optInt(bits), mode, C.bool(false), b.stream)
	// x is [..lead.., 1, 1, in] (mlx-lm's double expand_dims); the result keeps
	// an M=1 axis -> shape = rhsIndices.shape + [1, w.out]. w is [E, out, in_p].
	shape := append(cloneShape(tr.shape), 1, tw.shape[1])
	return b.newT(out, shape)
}

func (b *Backend) Argmax(x engine.Tensor, axis int) engine.Tensor {
	tx := x.(*tensor)
	out := C.mlx_array_new()
	C.mlx_argmax_axis(&out, tx.arr, C.int(axis), C.bool(false), b.stream)
	shape := append(cloneShape(tx.shape[:axis]), tx.shape[axis+1:]...)
	return b.newT(out, shape)
}

// --- evaluation / readback ---

func (b *Backend) Eval(ts ...engine.Tensor) {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range ts {
		if t != nil {
			C.mlx_vector_array_append_value(vec, t.(*tensor).arr)
		}
	}
	C.mlx_eval(vec)
}

// AsyncEval schedules evaluation without blocking (mlx provides backpressure
// when the async queue gets deep).
func (b *Backend) AsyncEval(ts ...engine.Tensor) {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range ts {
		if t != nil {
			C.mlx_vector_array_append_value(vec, t.(*tensor).arr)
		}
	}
	C.mlx_async_eval(vec)
}

func (b *Backend) Floats(t engine.Tensor) []float32 {
	tt := t.(*tensor)
	f32 := C.mlx_array_new()
	C.mlx_astype(&f32, tt.arr, C.MLX_FLOAT32, b.stream)
	defer C.mlx_array_free(f32)
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	C.mlx_vector_array_append_value(vec, f32)
	C.mlx_eval(vec)
	n := int(C.mlx_array_size(f32))
	ptr := C.mlx_array_data_float32(f32)
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n))
	return out
}

// DTypeOf reports the tensor's element type (engine.RawReader).
func (b *Backend) DTypeOf(t engine.Tensor) engine.DType {
	switch C.mlx_array_dtype(t.(*tensor).arr) {
	case C.MLX_FLOAT32:
		return engine.F32
	case C.MLX_FLOAT16:
		return engine.F16
	case C.MLX_BFLOAT16:
		return engine.BF16
	case C.MLX_INT32:
		return engine.I32
	case C.MLX_UINT32:
		return engine.U32
	default:
		panic("mlxc.DTypeOf: unsupported dtype")
	}
}

// Bytes downloads the tensor's raw bytes at native precision (engine.RawReader).
// The array is forced contiguous first — mlx_array_data_uint8 reads the buffer
// linearly, and a Concat/Slice result (a KV cache) may be a strided view.
func (b *Backend) Bytes(t engine.Tensor) []byte {
	tt := t.(*tensor)
	cont := C.mlx_array_new()
	C.mlx_contiguous(&cont, tt.arr, false, b.stream)
	defer C.mlx_array_free(cont)
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	C.mlx_vector_array_append_value(vec, cont)
	C.mlx_eval(vec)
	n := int(C.mlx_array_nbytes(cont))
	ptr := C.mlx_array_data_uint8(cont)
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n))
	return out
}

func (b *Backend) Ints(t engine.Tensor) []int32 {
	tt := t.(*tensor)
	i32 := C.mlx_array_new()
	C.mlx_astype(&i32, tt.arr, C.MLX_INT32, b.stream)
	defer C.mlx_array_free(i32)
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	C.mlx_vector_array_append_value(vec, i32)
	C.mlx_eval(vec)
	n := int(C.mlx_array_size(i32))
	ptr := C.mlx_array_data_int32(i32)
	out := make([]int32, n)
	copy(out, unsafe.Slice((*int32)(unsafe.Pointer(ptr)), n))
	return out
}

// compile-time assertions.
var (
	_ engine.Backend = (*Backend)(nil)
	_ engine.Sweeper = (*Backend)(nil)
)
