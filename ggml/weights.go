//go:build ggml

package ggml

/*
#include "ggml.h"
#include "ggml-backend.h"
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/guregodevo/janis/engine"
)

// isWeight reports whether t holds values that stay: pinned and computed,
// in a buffer nothing writes again. A cache is appended to in place, and a
// recorded step writes its inputs and the states it carries before and
// during each run; a state that is replaced, not written, counts.
func isWeight(t *tensor) bool {
	return t != nil && t.pin && !t.input && !t.replay && (t.grp == nil || t.grp.base == nil) && t.t != nil && t.t.data != nil &&
		bool(C.ggml_is_contiguous(t.t))
}

// weightFloats reads a float weight from the device without a flush (a
// flush ends a step being recorded). Nil for a dtype it does not read.
func weightFloats(w *tensor) []float32 {
	n := numel(w.shape)
	out := make([]float32, n)
	switch w.dt {
	case engine.F32:
		C.ggml_backend_tensor_get(w.t, unsafe.Pointer(&out[0]), 0, C.size_t(n*4))
	case engine.BF16:
		raw := make([]uint16, n)
		C.ggml_backend_tensor_get(w.t, unsafe.Pointer(&raw[0]), 0, C.size_t(n*2))
		for i, h := range raw {
			out[i] = math.Float32frombits(uint32(h) << 16)
		}
	case engine.F16:
		raw := make([]uint16, n)
		C.ggml_backend_tensor_get(w.t, unsafe.Pointer(&raw[0]), 0, C.size_t(n*2))
		C.ggml_fp16_to_fp32_row((*C.ggml_fp16_t)(unsafe.Pointer(&raw[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(n))
	default:
		return nil
	}
	return out
}

// madeOf is what build makes of weight w under key, made once and kept
// with the weight. Nil when w is not a weight or build gives nothing.
func (b *Backend) madeOf(w *tensor, key string, build func(vals []float32) ([]float32, []int)) *tensor {
	if !isWeight(w) {
		return nil
	}
	b.mu.Lock()
	t := w.made[key]
	b.mu.Unlock()
	if t != nil {
		return t
	}
	vals := weightFloats(w)
	if vals == nil {
		return nil
	}
	out, shape := build(vals)
	if out == nil {
		return nil
	}
	t = b.kept(b.newFrom(engine.F32, shape, unsafe.Pointer(&out[0]), len(out)*4))
	b.mu.Lock()
	if w.made == nil {
		w.made = map[string]*tensor{}
	}
	w.made[key] = t
	b.mu.Unlock()
	return t
}

// transposedWeight is a matrix weight [K, N] as [N, K] in float32, the
// first operand ggml_mul_mat takes: MatMul made that copy on every call.
func (b *Backend) transposedWeight(w *tensor) *tensor {
	if len(w.shape) != 2 {
		return nil
	}
	K, N := w.shape[0], w.shape[1]
	return b.madeOf(w, "T", func(v []float32) ([]float32, []int) {
		out := make([]float32, len(v))
		for k := 0; k < K; k++ {
			for n := 0; n < N; n++ {
				out[n*K+k] = v[k*N+n]
			}
		}
		return out, []int{N, K}
	})
}

// kept pins a leaf for the backend's life. A leaf made while a step is
// recorded is the step's, unpinned when the step is closed: this one is
// taken out of the step's.
func (b *Backend) kept(t *tensor) *tensor {
	t.pin = true
	b.mu.Lock()
	if rs := b.rec0; rs != nil {
		for i, l := range rs.leaves {
			if l == t {
				rs.leaves = append(rs.leaves[:i], rs.leaves[i+1:]...)
				break
			}
		}
	}
	b.mu.Unlock()
	return t
}
