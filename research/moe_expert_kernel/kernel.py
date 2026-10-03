"""
Fused sparse-gather + dequant + GEMV as a hand-written Metal kernel
(mx.fast.metal_kernel) — the kernel BODY is ours; MLX only dispatches/compiles.

Validates against MLX's OWN quantize/dequantize (the feature-flag baseline):
  MLX quantize  ->  { our Metal kernel , MLX dequantize+matmul }  ->  compare
If our kernel reads MLX's affine 4-bit layout correctly, the two agree to
float-accumulation noise (not just quant tolerance — same quantized weights).

Operation:  out[h] = sum over active rows n of  x[k] * (scale*q + bias)[n,h]
Only K of N rows are touched — the sparse win. For Qwen3-30B-A3B the K active
rows are the router's top-k experts (real, free sparsity).
"""
import time
import mlx.core as mx

GROUP_SIZE = 64
BITS = 4


def build_kernel(N, H, K):
    wpr = H // 8                 # uint32 words per row (8 nibbles/word)
    ng = H // GROUP_SIZE         # groups per row
    src = f"""
        uint h = thread_position_in_grid.x;
        if (h >= {H}u) return;
        uint g    = h / {GROUP_SIZE}u;
        uint shift = (h % 8u) * 4u;
        uint wcol = h / 8u;
        float acc = 0.0f;
        for (uint k = 0; k < {K}u; ++k) {{
            uint n = active_idx[k];
            uint word = packed[n * {wpr}u + wcol];
            uint q = (word >> shift) & 0xFu;
            float s = scales[n * {ng}u + g];
            float b = biases[n * {ng}u + g];
            acc += x[k] * (s * (float)q + b);
        }}
        out[h] = acc;
    """
    return mx.fast.metal_kernel(
        name="sparse_gather_dequant_gemv",
        input_names=["x", "packed", "scales", "biases", "active_idx"],
        output_names=["out"],
        source=src,
    )


def run_kernel(kernel, x, packed, scales, biases, active_idx, H):
    (out,) = kernel(
        inputs=[x, packed, scales, biases, active_idx],
        output_shapes=[(H,)],
        output_dtypes=[mx.float32],
        grid=(H, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def main():
    mx.random.seed(0)
    N, H, K = 4096, 2048, 1024          # 25% active -> 4x byte reduction ceiling

    W = (mx.random.normal((N, H)) * 0.05).astype(mx.float32)
    # MLX's OWN quantizer — this is the authoritative layout we validate against.
    w_q, scales, biases = mx.quantize(W, group_size=GROUP_SIZE, bits=BITS)
    scales = scales.astype(mx.float32)
    biases = biases.astype(mx.float32)

    active_idx = mx.array(
        sorted(mx.random.permutation(N)[:K].tolist()), dtype=mx.uint32)
    x = (mx.random.normal((K,))).astype(mx.float32)

    kernel = build_kernel(N, H, K)
    out = run_kernel(kernel, x, w_q, scales, biases, active_idx, H)
    mx.eval(out)

    # Baseline: MLX's own dequantize, then the masked GEMV. Same quantized
    # weights -> any difference is float accumulation order, not layout.
    W_dq = mx.dequantize(w_q, scales, biases, group_size=GROUP_SIZE, bits=BITS)
    x_full = mx.zeros((N,), dtype=mx.float32)
    x_full[active_idx] = x
    ref = (x_full[None, :] @ W_dq)[0]
    mx.eval(ref)

    diff = mx.max(mx.abs(out - ref)).item()
    rel = diff / (mx.max(mx.abs(ref)).item() + 1e-9)
    print(f"N={N} H={H} K={K}  K/N={K/N:.3f}")
    print(f"Metal kernel vs MLX dequantize+matmul: max|Δ|={diff:.3e} rel={rel:.3e}")
    ok = rel < 1e-3
    print("PASS — kernel reads MLX layout correctly" if ok
          else "FAIL — layout/accumulation mismatch")

    bench_moe()


def bench_moe():
    """Realistic Qwen3-30B-A3B down-proj geometry, where compute >> launch floor
    so the true K/N (active-expert) scaling shows.

    down_proj combined across experts:
      rows N = num_experts * moe_inter = 128 * 768 = 98304   (each row = one
        (expert, intermediate-neuron); its [hidden] vector is that neuron's
        down-proj contribution)
      H = hidden = 2048
      routing to E of 128 experts -> K = E * moe_inter active rows.
    """
    NUM_EXPERTS, MOE_INTER, HID = 128, 768, 2048
    N, H = NUM_EXPERTS * MOE_INTER, HID
    print(f"\n=== Realistic MoE down-proj: N={N} (128 experts x 768) H={H} ===")
    W = (mx.random.normal((N, H)) * 0.05).astype(mx.float32)
    w_q, scales, biases = mx.quantize(W, group_size=GROUP_SIZE, bits=BITS)
    scales, biases = scales.astype(mx.float32), biases.astype(mx.float32)

    print(f"  {'experts':>7} {'K':>7} {'K/N':>6}  {'µs/call':>9}  {'vs dense':>8}")
    dense_us = None
    for E in (128, 64, 32, 16, 8, 4, 2):
        K = E * MOE_INTER
        ai = mx.array(list(range(K)), dtype=mx.uint32)
        xx = mx.random.normal((K,)).astype(mx.float32)
        kr = build_kernel(N, H, K)
        o = run_kernel(kr, xx, w_q, scales, biases, ai, H); mx.eval(o)  # warmup
        t0 = time.perf_counter()
        for _ in range(50):
            o = run_kernel(kr, xx, w_q, scales, biases, ai, H)
        mx.eval(o)
        us = (time.perf_counter() - t0) / 50 * 1e6
        if dense_us is None:
            dense_us = us
        print(f"  {E:>7} {K:>7} {K/N:>6.3f}  {us:>8.1f}  {dense_us/us:>7.2f}x")
    print("  (E=8 is the real routing point for Qwen3-30B-A3B)")


if __name__ == "__main__":
    main()
