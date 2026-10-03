"""
Kernel optimization: close the theory-vs-achieved gap on the sparse MoE
down-proj GEMV by removing redundant weight reads via thread coarsening.

Baseline (kernel.py): 1 thread per output column. Each thread reads the full
uint32 weight word but consumes only 1 of its 8 nibbles -> every word is read
8x redundantly across 8 adjacent threads.

Coarsened: each thread owns CPT consecutive outputs = CPT/8 words. It reads each
word ONCE and produces all 8 outputs from it, and reuses scale/bias (same group)
and x[k]/active_idx[k] across those outputs. Fewer redundant reads, but fewer
threads -> lower occupancy. Net effect is empirical, so we sweep CPT and measure.
"""
import time
import mlx.core as mx

GROUP_SIZE, BITS = 64, 4


def build_baseline(N, H, K):
    wpr, ng = H // 8, H // GROUP_SIZE
    src = f"""
        uint h = thread_position_in_grid.x;
        if (h >= {H}u) return;
        uint g = h / {GROUP_SIZE}u; uint shift = (h % 8u)*4u; uint wcol = h/8u;
        float acc = 0.0f;
        for (uint k=0;k<{K}u;++k) {{
            uint n = active_idx[k];
            uint word = packed[n*{wpr}u + wcol];
            uint q = (word >> shift) & 0xFu;
            acc += x[k] * (scales[n*{ng}u+g]*(float)q + biases[n*{ng}u+g]);
        }}
        out[h] = acc;
    """
    return mx.fast.metal_kernel(name="gemv_base",
        input_names=["x", "packed", "scales", "biases", "active_idx"],
        output_names=["out"], source=src), (H, 1, 1)


def build_coarsened(N, H, K, CPT):
    """CPT consecutive outputs per thread (CPT a multiple of 8)."""
    wpr, ng, words = H // 8, H // GROUP_SIZE, CPT // 8
    src = f"""
        uint t = thread_position_in_grid.x;
        uint h0 = t*{CPT}u;
        if (h0 >= {H}u) return;
        float acc[{CPT}];
        for (uint a=0;a<{CPT}u;++a) acc[a]=0.0f;
        for (uint k=0;k<{K}u;++k) {{
            uint n = active_idx[k];
            float xk = x[k];
            uint nrow = n*{wpr}u;
            for (uint wo=0; wo<{words}u; ++wo) {{
                uint hbase = h0 + wo*8u;
                uint word = packed[nrow + h0/8u + wo];
                uint g = hbase/{GROUP_SIZE}u;
                float s = scales[n*{ng}u+g]; float b = biases[n*{ng}u+g];
                for (uint j=0;j<8u;++j) {{
                    uint q = (word >> (4u*j)) & 0xFu;
                    acc[wo*8u + j] += xk*(s*(float)q + b);
                }}
            }}
        }}
        for (uint a=0;a<{CPT}u;++a) {{ if (h0+a < {H}u) out[h0+a]=acc[a]; }}
    """
    grid = ((H + CPT - 1) // CPT, 1, 1)
    return mx.fast.metal_kernel(name=f"gemv_c{CPT}",
        input_names=["x", "packed", "scales", "biases", "active_idx"],
        output_names=["out"], source=src), grid


def run(kernel, grid, x, packed, scales, biases, active_idx, H, tg=128):
    (out,) = kernel(inputs=[x, packed, scales, biases, active_idx],
                    output_shapes=[(H,)], output_dtypes=[mx.float32],
                    grid=grid, threadgroup=(min(tg, grid[0]), 1, 1))
    return out


def time_kernel(kernel, grid, args, H, iters=50):
    o = run(kernel, grid, *args, H); mx.eval(o)            # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        o = run(kernel, grid, *args, H)
    mx.eval(o)
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    mx.random.seed(0)
    # correctness at small dims, then bench at realistic MoE down-proj dims
    NUM_EXPERTS, MOE_INTER, HID = 128, 768, 2048
    N, H = NUM_EXPERTS * MOE_INTER, HID
    W = (mx.random.normal((N, H)) * 0.05).astype(mx.float32)
    w_q, scales, biases = mx.quantize(W, group_size=GROUP_SIZE, bits=BITS)
    scales, biases = scales.astype(mx.float32), biases.astype(mx.float32)

    # --- correctness: every coarsened variant must match MLX dequantize+matmul
    Kc = 8 * MOE_INTER
    ai = mx.array(list(range(Kc)), dtype=mx.uint32)
    xc = mx.random.normal((Kc,)).astype(mx.float32)
    W_dq = mx.dequantize(w_q, scales, biases, group_size=GROUP_SIZE, bits=BITS)
    xf = mx.zeros((N,), dtype=mx.float32); xf[ai] = xc
    ref = (xf[None, :] @ W_dq)[0]; mx.eval(ref)
    print("correctness (vs MLX dequantize+matmul), E=8:")
    for name, (k, g) in {
        "baseline":   build_baseline(N, H, Kc),
        "coarsen-8":  build_coarsened(N, H, Kc, 8),
        "coarsen-16": build_coarsened(N, H, Kc, 16),
        "coarsen-32": build_coarsened(N, H, Kc, 32),
    }.items():
        o = run(k, g, xc, w_q, scales, biases, ai, H); mx.eval(o)
        rel = (mx.max(mx.abs(o - ref)) / (mx.max(mx.abs(ref)) + 1e-9)).item()
        print(f"  {name:>11}: rel={rel:.2e}  {'PASS' if rel < 1e-3 else 'FAIL'}")

    # --- benchmark at the two interesting points
    print(f"\nbench (N={N} H={H}), µs/call:")
    print(f"  {'variant':>11} {'E=128':>9} {'E=8':>9}  {'E=8 vs base':>11}")
    variants = {
        "baseline":   build_baseline,
        "coarsen-8":  lambda N, H, K: build_coarsened(N, H, K, 8),
        "coarsen-16": lambda N, H, K: build_coarsened(N, H, K, 16),
        "coarsen-32": lambda N, H, K: build_coarsened(N, H, K, 32),
    }
    base_e8 = None
    for name, builder in variants.items():
        row = {}
        for E in (128, 8):
            K = E * MOE_INTER
            ai = mx.array(list(range(K)), dtype=mx.uint32)
            xx = mx.random.normal((K,)).astype(mx.float32)
            k, g = builder(N, H, K)
            row[E] = time_kernel(k, g, (xx, w_q, scales, biases, ai), H)
        if name == "baseline":
            base_e8 = row[8]
        print(f"  {name:>11} {row[128]:>8.1f} {row[8]:>8.1f}  {base_e8/row[8]:>10.2f}x")


if __name__ == "__main__":
    main()
