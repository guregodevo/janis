// compat.h — janis against more than one mlx-c.
//
// Homebrew's mlx-c moves under janis: 0.8.0 added force_fused to
// mlx_fast_scaled_dot_product_attention and global_scale to mlx_gather_qmm,
// both after the arguments janis passes, and a developer's Mac on 0.6.0 and
// the release runner on 0.8.0 must build the same source. The headers carry
// no version macro, so each shim asks the compiler which signature the
// function has (_Generic on its address) and calls it through that type; the
// branch not taken is dead code behind a constant. A signature neither
// typedef matches is a compile error here, not a wrong call at run time.
#ifndef JANIS_MLXC_COMPAT_H
#define JANIS_MLXC_COMPAT_H

#include <dlfcn.h>
#include <stdbool.h>
#include "mlx/c/mlx.h"

typedef mlx_stream (*janis_stream_new_thread_unsafe)(mlx_device);

// janis_gpu_stream is the backend's one stream on the GPU. From mlx-c 0.8.0
// (MLX 0.32) a stream is thread affine: its command encoder is registered on
// the thread that made it, and evaluating from another thread fails with
// "There is no Stream(gpu, 0) in current thread". Go moves goroutines between
// threads, so janis asks for the stream "that can be used from any thread"
// (mlx_stream_new_thread_unsafe) when the library has it, looked up by name
// so the same source builds against 0.6.0, and the default GPU stream
// otherwise. MLX applies no synchronization to such a stream; janis's gpuMu
// serializes every call on it.
static inline mlx_stream janis_gpu_stream(void) {
	janis_stream_new_thread_unsafe any_thread =
	    (janis_stream_new_thread_unsafe)dlsym(RTLD_DEFAULT, "mlx_stream_new_thread_unsafe");
	if (any_thread == NULL)
		return mlx_default_gpu_stream_new();
	mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
	mlx_stream s = any_thread(gpu);
	mlx_device_free(gpu);
	return s;
}

typedef int (*janis_sdpa_v06)(mlx_array*, mlx_array, mlx_array, mlx_array, float, const char*, mlx_array, mlx_array, mlx_stream);
typedef int (*janis_sdpa_v08)(mlx_array*, mlx_array, mlx_array, mlx_array, float, const char*, mlx_array, mlx_array, bool, mlx_stream);

// janis_sdpa is mlx_fast_scaled_dot_product_attention with force_fused off
// (0.8.0's default, what 0.6.0 did).
static inline int janis_sdpa(mlx_array* res, mlx_array q, mlx_array k, mlx_array v, float scale,
                             const char* mask_mode, mlx_array mask, mlx_array sinks, mlx_stream s) {
	if (_Generic(&mlx_fast_scaled_dot_product_attention, janis_sdpa_v08: 1, janis_sdpa_v06: 0))
		return ((janis_sdpa_v08)&mlx_fast_scaled_dot_product_attention)(res, q, k, v, scale, mask_mode, mask, sinks, false, s);
	return ((janis_sdpa_v06)&mlx_fast_scaled_dot_product_attention)(res, q, k, v, scale, mask_mode, mask, sinks, s);
}

typedef int (*janis_gqmm_v06)(mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, bool,
                              mlx_optional_int, mlx_optional_int, const char*, bool, mlx_stream);
typedef int (*janis_gqmm_v08)(mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, bool,
                              mlx_optional_int, mlx_optional_int, const char*, mlx_array, bool, mlx_stream);

// janis_gather_qmm is mlx_gather_qmm with no global_scale (null, as 0.8.0 allows).
static inline int janis_gather_qmm(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                                   mlx_array lhs_indices, mlx_array rhs_indices, bool transpose,
                                   mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
                                   bool sorted_indices, mlx_stream s) {
	mlx_array no_global_scale = {0};
	if (_Generic(&mlx_gather_qmm, janis_gqmm_v08: 1, janis_gqmm_v06: 0))
		return ((janis_gqmm_v08)&mlx_gather_qmm)(res, x, w, scales, biases, lhs_indices, rhs_indices, transpose,
		                                         group_size, bits, mode, no_global_scale, sorted_indices, s);
	return ((janis_gqmm_v06)&mlx_gather_qmm)(res, x, w, scales, biases, lhs_indices, rhs_indices, transpose,
	                                         group_size, bits, mode, sorted_indices, s);
}

#endif
