//go:build darwin && arm64 && cgo

package mlxc

/*
#cgo darwin CFLAGS: -I/opt/homebrew/include
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"sync"
	"unsafe"

	"memdoor/llm/engine"
)

// gatedDeltaSource is mlx-lm's gated_delta_step metal kernel (models/
// gated_delta.py, scalar-gating non-masked variant), ported verbatim: one
// threadgroup simd-scans a value head's state rows across all T steps, so the
// whole recurrence is a single graph node instead of ~8 ops per token per
// layer. Verified against the pure-Go cpu.GatedDeltaScan in the diff test.
const gatedDeltaSource = `
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    // g: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      // Increment data pointers to next time step
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
`

var (
	gatedDeltaOnce   sync.Once
	gatedDeltaKernel C.mlx_fast_metal_kernel
)

func initGatedDeltaKernel() {
	name := C.CString("gated_delta_step")
	defer C.free(unsafe.Pointer(name))
	src := C.CString(gatedDeltaSource)
	defer C.free(unsafe.Pointer(src))
	hdr := C.CString("")
	defer C.free(unsafe.Pointer(hdr))

	inputs := C.mlx_vector_string_new()
	defer C.mlx_vector_string_free(inputs)
	for _, s := range []string{"q", "k", "v", "g", "beta", "state_in", "T"} {
		cs := C.CString(s)
		C.mlx_vector_string_append_value(inputs, cs)
		C.free(unsafe.Pointer(cs))
	}
	outputs := C.mlx_vector_string_new()
	defer C.mlx_vector_string_free(outputs)
	for _, s := range []string{"y", "state_out"} {
		cs := C.CString(s)
		C.mlx_vector_string_append_value(outputs, cs)
		C.free(unsafe.Pointer(cs))
	}
	gatedDeltaKernel = C.mlx_fast_metal_kernel_new(name, inputs, outputs, src, hdr, C.bool(true), C.bool(false))
}

// GatedDeltaScan runs the fused recurrence (see engine.Backend). All inputs
// are float32; one kernel dispatch covers every step of the pass.
func (b *Backend) GatedDeltaScan(q, k, v, g, beta, state engine.Tensor) (engine.Tensor, engine.Tensor) {
	gatedDeltaOnce.Do(initGatedDeltaKernel)

	tq, tk, tv := q.(*tensor), k.(*tensor), v.(*tensor)
	tg, tb, ts := g.(*tensor), beta.(*tensor), state.(*tensor)
	B, T := tq.shape[0], tq.shape[1]
	Hk, Dk := tq.shape[2], tq.shape[3]
	Hv, Dv := tv.shape[2], tv.shape[3]

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	for name, dt := range map[string]C.mlx_dtype{"InT": C.MLX_FLOAT32, "StT": C.MLX_FLOAT32} {
		cs := C.CString(name)
		C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, cs, dt)
		C.free(unsafe.Pointer(cs))
	}
	for _, ta := range []struct {
		name string
		v    int
	}{{"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}} {
		cs := C.CString(ta.name)
		C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cs, C.int(ta.v))
		C.free(unsafe.Pointer(cs))
	}
	yShape := []C.int{C.int(B), C.int(T), C.int(Hv), C.int(Dv)}
	stShape := []C.int{C.int(B), C.int(Hv), C.int(Dv), C.int(Dk)}
	C.mlx_fast_metal_kernel_config_add_output_arg(cfg, &yShape[0], C.size_t(len(yShape)), C.MLX_FLOAT32)
	C.mlx_fast_metal_kernel_config_add_output_arg(cfg, &stShape[0], C.size_t(len(stShape)), C.MLX_FLOAT32)
	C.mlx_fast_metal_kernel_config_set_grid(cfg, 32, C.int(Dv), C.int(B*Hv))
	C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1)

	tScalar := C.mlx_array_new_int(C.int(T))
	defer C.mlx_array_free(tScalar)
	ins := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(ins)
	for _, t := range []C.mlx_array{tq.arr, tk.arr, tv.arr, tg.arr, tb.arr, ts.arr, tScalar} {
		C.mlx_vector_array_append_value(ins, t)
	}

	outs := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outs)
	C.mlx_fast_metal_kernel_apply(&outs, gatedDeltaKernel, ins, cfg, b.stream)

	yArr := C.mlx_array_new()
	stArr := C.mlx_array_new()
	C.mlx_vector_array_get(&yArr, outs, 0)
	C.mlx_vector_array_get(&stArr, outs, 1)
	yT := b.newT(yArr, []int{B, T, Hv, Dv})
	stT := b.newT(stArr, []int{B, Hv, Dv, Dk})
	return yT, stT
}
