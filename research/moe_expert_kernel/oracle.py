"""
Ground-truth oracle for the fused sparse-gather + dequant + GEMV kernel,
matching MLX's affine 4-bit layout so it validates against MLX QuantMatmul
(the feature-flag baseline) and against llm/cpu's QuantMatmul.

Sparsity note: `active_idx` is whatever subset of rows you touch. For our
models the real, free sparsity is the MoE router's top-k experts
(Qwen3-30B-A3B: 8 of 128) -- NOT predicted SwiGLU neurons (those aren't
naturally sparse). Same kernel either way; only the source of active_idx differs.
"""
import numpy as np

BITS = 4
VALS_PER_WORD = 8          # 8 nibbles per uint32, LSB-first (MLX layout)
QMAX = (1 << BITS) - 1     # 15


# ---- MLX-style affine quantizer: w = scale*q + bias, q in [0,15] -----------
def quantize_rows(W: np.ndarray, group_size: int):
    """W:[N,H] f32 -> packed:[N,H//8] uint32, scales/biases:[N,H//group_size] f32."""
    N, H = W.shape
    assert H % group_size == 0 and H % VALS_PER_WORD == 0
    Wg = W.reshape(N, H // group_size, group_size)
    gmin = Wg.min(axis=2)
    gmax = Wg.max(axis=2)
    scales = ((gmax - gmin) / QMAX)
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    biases = gmin.astype(np.float32)
    q = np.round((Wg - biases[:, :, None]) / scales[:, :, None])
    q = np.clip(q, 0, QMAX).astype(np.uint32).reshape(N, H)
    # pack 8 nibbles per uint32, element i -> bits [4i,4i+3]
    qv = q.reshape(N, H // VALS_PER_WORD, VALS_PER_WORD)
    shifts = (BITS * np.arange(VALS_PER_WORD)).astype(np.uint32)
    packed = (qv << shifts).sum(axis=2).astype(np.uint32)   # [N, H//8]
    return packed, scales, biases


# ---- dequant a SINGLE row (== one threadgroup's job in the Metal kernel) ----
def dequantize_row(packed_row, scales_row, biases_row, H, group_size):
    shifts = (BITS * np.arange(VALS_PER_WORD)).astype(np.uint32)
    q = ((packed_row[:, None] >> shifts) & QMAX).reshape(H).astype(np.float32)
    s = np.repeat(scales_row, group_size)
    b = np.repeat(biases_row, group_size)
    return (q * s + b).astype(np.float32)


# ---- THE ORACLE: gather active rows, dequant, scale, accumulate ------------
def sparse_gather_dequant_gemv(act_scale, packed, scales, biases,
                               active_idx, H, group_size):
    out = np.zeros(H, dtype=np.float32)
    for k, n in enumerate(active_idx):
        out += act_scale[k] * dequantize_row(packed[n], scales[n], biases[n], H, group_size)
    return out


# ---- self-test: cross-check vs an independent dense-masked path ------------
def _selftest():
    rng = np.random.default_rng(0)
    N, H, group_size, K = 4096, 2048, 64, 1024
    W = (rng.standard_normal((N, H)) * 0.05).astype(np.float32)
    packed, scales, biases = quantize_rows(W, group_size)

    active_idx = np.sort(rng.choice(N, size=K, replace=False))
    act_vals = rng.standard_normal(K).astype(np.float32)
    out = sparse_gather_dequant_gemv(act_vals, packed, scales, biases, active_idx, H, group_size)

    W_dq = np.stack([dequantize_row(packed[n], scales[n], biases[n], H, group_size) for n in range(N)])
    act_full = np.zeros(N, dtype=np.float32)
    act_full[active_idx] = act_vals
    ref = (act_full[None, :] @ W_dq).astype(np.float32).ravel()

    rel = float(np.max(np.abs(out - ref))) / (float(np.max(np.abs(ref))) + 1e-9)
    q_err = float(np.max(np.abs(W_dq - W)))
    print(f"N={N} H={H} K={K}  K/N={K/N:.3f}")
    print(f"oracle vs dense-masked (algebraically identical): max rel diff = {rel:.3e}")
    print(f"quantization error |dequant(W)-W| max abs = {q_err:.3e}  (kernel tolerance band)")
    assert rel < 1e-5, "oracle internal cross-check FAILED"
    print("PASS: oracle internally consistent, MLX-affine layout.")


if __name__ == "__main__":
    _selftest()
