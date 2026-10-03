"""
Head-to-head: our fused gather+dequant+GEMV Metal kernel vs MLX's OWN
quantized_matmul on the gathered experts -- the honest production baseline a
competent engineer would actually write to exploit MoE sparsity.

Operation (one MoE projection, e.g. up_proj across experts):
  W : [N, IN] quantized along IN (MLX layout), N = num_experts*out_per_expert
  route to E experts -> K = E*out_per_expert active rows
  y[r] = sum_in x[in] * dequant(W)[active_row r, in]      (contract IN)

Three contenders, same quantized weights, same active set:
  A) MLX dense quantized_matmul over ALL N rows, then take active  (no sparsity)
  B) MLX gathered: take active rows, then quantized_matmul          (library sparse)
  C) ours: fused kernel, touches only active rows, no gather buffer (custom)

Beating B is the real claim. A shows what sparsity buys at all.
"""
import time
import mlx.core as mx

GROUP_SIZE, BITS = 64, 4


def build_fused(N, IN, K):
    wpr, ng = IN // 8, IN // GROUP_SIZE
    src = f"""
        uint r = thread_position_in_grid.x;
        if (r >= {K}u) return;
        uint n = active_idx[r];
        uint nrow = n*{wpr}u; uint noff = n*{ng}u;
        float acc = 0.0f;
        for (uint wc=0; wc<{wpr}u; ++wc) {{
            uint word = packed[nrow + wc];
            uint ii = wc*8u; uint g = wc/8u;        // group changes every 8 words
            float s = scales[noff+g]; float b = biases[noff+g];
            for (uint j=0;j<8u;++j) {{
                uint q = (word >> (4u*j)) & 0xFu;
                acc += x[ii+j] * (s*(float)q + b);
            }}
        }}
        out[r] = acc;
    """
    return mx.fast.metal_kernel(name="fused_gemv",
        input_names=["x", "packed", "scales", "biases", "active_idx"],
        output_names=["out"], source=src)


def run_fused(k, x, packed, scales, biases, active_idx, K):
    (out,) = k(inputs=[x, packed, scales, biases, active_idx],
               output_shapes=[(K,)], output_dtypes=[mx.float32],
               grid=(K, 1, 1), threadgroup=(256, 1, 1))
    return out


def t(fn, iters=50):
    o = fn(); mx.eval(o)                       # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
    mx.eval(o)
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    mx.random.seed(0)
    NUM_EXPERTS, OUT_PER, IN = 128, 768, 2048
    N = NUM_EXPERTS * OUT_PER
    W = (mx.random.normal((N, IN)) * 0.05).astype(mx.float32)
    w_q, scales, biases = mx.quantize(W, group_size=GROUP_SIZE, bits=BITS)
    scales, biases = scales.astype(mx.float32), biases.astype(mx.float32)
    x = mx.random.normal((IN,)).astype(mx.float32)

    fused_cache = {}
    def fused_for(K):
        if K not in fused_cache:
            fused_cache[K] = build_fused(N, IN, K)
        return fused_cache[K]

    # correctness at E=8 vs MLX gathered quantized_matmul
    Kc = 8 * OUT_PER
    ai = mx.array(list(range(Kc)), dtype=mx.uint32)
    gw = mx.take(w_q, ai, axis=0)
    gs = mx.take(scales, ai, axis=0)
    gb = mx.take(biases, ai, axis=0)
    y_mlx = mx.quantized_matmul(x[None, :], gw, gs, gb, transpose=True,
                                group_size=GROUP_SIZE, bits=BITS)[0]
    y_ours = run_fused(fused_for(Kc), x, w_q, scales, biases, ai, Kc)
    mx.eval(y_mlx, y_ours)
    rel = (mx.max(mx.abs(y_ours - y_mlx)) / (mx.max(mx.abs(y_mlx)) + 1e-9)).item()
    print(f"correctness E=8: ours vs MLX gathered quantized_matmul  rel={rel:.2e} "
          f"{'PASS' if rel < 5e-2 else 'FAIL'}")

    print(f"\nhead-to-head (N={N} IN={IN}), µs/call:")
    print(f"  {'E':>4} {'K':>6}  {'A:dense':>8} {'B:gathered':>10} {'C:ours':>8}"
          f"  {'ours vs B':>9}")
    for E in (128, 32, 8, 4):
        K = E * OUT_PER
        ai = mx.array(list(range(K)), dtype=mx.uint32)

        def dense():
            return mx.quantized_matmul(x[None, :], w_q, scales, biases,
                                       transpose=True, group_size=GROUP_SIZE, bits=BITS)

        def gathered():
            gw = mx.take(w_q, ai, axis=0); gs = mx.take(scales, ai, axis=0)
            gb = mx.take(biases, ai, axis=0)
            return mx.quantized_matmul(x[None, :], gw, gs, gb, transpose=True,
                                       group_size=GROUP_SIZE, bits=BITS)

        kf = fused_for(K)
        def ours():
            return run_fused(kf, x, w_q, scales, biases, ai, K)

        a, b, c = t(dense), t(gathered), t(ours)
        print(f"  {E:>4} {K:>6}  {a:>8.1f} {b:>10.1f} {c:>8.1f}  {b/c:>8.2f}x")
    print("  (E=8 is the real Qwen3-30B-A3B routing point)")


if __name__ == "__main__":
    main()
