# Fused sparse-gather + dequant + GEMV — a hand-written Metal kernel

A custom Apple-GPU kernel (`mx.fast.metal_kernel`) that computes the
**MoE down-projection** directly from MLX's affine 4-bit weights, touching only
the rows of the **routed experts** instead of all of them. The kernel *body* is
hand-authored MSL; MLX only dispatches and compiles it.

This is the compute core of MoE expert-offloading (running a 30B-class MoE on a
16 GB Mac by keeping only active experts hot) **and** a standalone
GPU-kernel-authoring artifact.

## What it computes

```
out[h] = Σ over active rows n of   x[n] · (scale·q + bias)[n, h]
```

Stack every expert's down-proj rows as `W : [num_experts·moe_inter, hidden]`,
where row `n = (expert e, intermediate-neuron j)` carries that neuron's
contribution vector over `hidden`. Routing a token to `E` of `num_experts`
experts means only `E·moe_inter` rows are active — that's the sparsity, and it's
**given for free by the router** (no predictor, unlike dense-SwiGLU contextual
sparsity, which these models don't have).

For Qwen3-30B-A3B: `num_experts=128`, `moe_inter=768`, `hidden=2048`, `E=8`.

## Why a custom kernel (not MLX's ops)

Calling MLX's `quantized_matmul` is *using* a kernel; the hiring/skill signal is
*authoring* one. And the fused gather→dequant→GEMV is a pattern the stock op
doesn't express: it dequantizes 4-bit weights inline and touches only gathered
rows, so the byte movement scales with `K`, not `N`. On a bandwidth-bound decode
path, fewer bytes moved is the whole game.

## Correctness — a three-way validation triangle

`oracle.py` (NumPy) ↔ **MLX `quantize`/`dequantize`** ↔ the Metal kernel.

- `oracle.py` cross-checks itself against an independent dense-masked path:
  `rel 3.4e-7`, and confirms the affine layout (`w = scale·q + bias`, unsigned
  `q∈[0,15]`, 8 nibbles LSB-first per uint32) — exactly MLX's format and
  `llm/cpu`'s `QuantMatmul`.
- `kernel.py` validates the Metal kernel against **MLX's own
  `dequantize`+matmul** on weights MLX itself quantized:

  ```
  Metal kernel vs MLX dequantize+matmul:  max|Δ|=5.2e-6  rel=9.7e-7  → PASS
  ```

  Same quantized weights, so the residual is float-accumulation order, not a
  layout error. This *is* the "feature-flag against MLX" gate: a runtime toggle
  picks custom-kernel vs MLX `quantized_matmul`, and this diff-test guards it
  (mirrors the `quant4SIMD` + `cpu_mlx_diff_test` pattern in the Go engine).

## Performance — and the honest part

**Toy dims (`N=4096, H=2048`): overhead-bound.** A 4× row reduction bought only
~1.75× wall-time and plateaued below K/N=0.25 — a ~15 µs kernel-launch floor
dominated the tiny compute. Reporting *why* (floor > saveable compute at this
scale) matters more than the number.

**An optimization that didn't pan out (and why that's worth reporting).**
Hypothesis: the first kernel read each weight word but used 1 of its 8 nibbles,
so 8 adjacent threads re-read the same word — fix by *thread coarsening* (one
thread reads a word once, emits all 8 outputs). Measured (`optimize.py`): it
**didn't help and got worse** as it grew (coarsen-8 = no change, coarsen-32 =
0.51×). The redundant reads were already cache-absorbed; coarsening just cut
thread count and starved occupancy. At the sparse routing point the kernel is
**occupancy/latency-bound, not redundant-read-bound** — so the real fix was the
opposite: *more* parallelism (one thread per active row), not less.

**Head-to-head vs MLX's own `quantized_matmul` (`head2head.py`) — the honest
production baseline.** Same quantized weights, same active set. `A` = MLX dense
over all experts (no sparsity); `B` = the library sparse path (`mx.take` gather
+ `quantized_matmul`); `C` = our fused kernel. µs/call, `N=98304, IN=2048`:

| E | K | A: dense | B: gathered | C: ours | ours vs B |
|---:|---:|---:|---:|---:|---:|
| 128 | 98304 | 32.8 | 82.5 | 50.9 | 1.62× |
| 32 | 24576 | 30.0 | 27.5 | 16.1 | 1.71× |
| **8** | **6144** | 29.9 | 12.1 | **7.2** | **1.67×** |
| 4 | 3072 | 30.7 | 20.7 | 7.1 | 2.94× |

At the **real E=8 routing point, ours beats MLX's gathered `quantized_matmul`
1.67×** (and dense MLX 4.2×), correct to `rel 1.26e-6`. The win is **fusing the
gather**: the library path materializes the gathered weights with a `mx.take`
copy; the kernel reads only the active rows in place, no copy.

**The honest crossover:** at `E=128` (no sparsity) MLX dense wins (32.8 vs 50.9)
— gathering all experts is pointless, so you'd call dense there. The kernel's win
is specifically the sparse regime (`E ≤ 32`). Reporting where it *loses* is what
makes the where-it-wins credible.

## Honest limits / next

- **Win is in the sparse regime only.** For dense (no routing) MLX's tuned
  `quantized_matmul` is faster — call it there; this kernel is for `E ≪ experts`.
- **Still room to push C.** No threadgroup-memory reuse of `x`, no simdgroup
  matrix ops, no K-tiling — the 1.67× could grow with those.
- **Sparsity is real for MoE** (router top-k), *not* for dense SwiGLU FFNs
  (no natural activation sparsity, would need a trained predictor).
- **Wire it into the Go engine** behind a feature flag (custom-Metal vs MLX
  `quantized_matmul`), gated by the same diff-test (mirrors `quant4SIMD` +
  `cpu_mlx_diff_test`).

## Files

- `oracle.py` — NumPy ground truth, MLX-affine layout, self-validating.
- `kernel.py` — first `mx.fast.metal_kernel` body + MLX-validation + the
  toy/realistic benchmarks (incl. the toy-dims overhead-floor finding).
- `optimize.py` — thread-coarsening sweep (the negative result).
- `head2head.py` — the natural one-row-per-thread kernel vs MLX dense and
  gathered `quantized_matmul` (the result of record).

Run (needs `mlx` + `mlx-lm`, Apple Silicon):

```bash
python oracle.py      # numpy only — proves the layout/algebra
python kernel.py      # Metal kernel: validate vs MLX, then both benchmarks
```
