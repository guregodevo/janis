# Local Accelerator Abstraction — the `engine.Backend` seam (design note)

**Status:** implemented and shipping. memdoor runs LLM inference in-process on
two interchangeable backends behind one Go interface, so the same model code runs
on Apple Silicon GPUs and on plain Linux/Intel CPUs.

> History: an earlier version of this note designed a llamafit/llama.cpp
> GPU-offload abstraction (`autoGPULayers`, VRAM probes, a CUDA box). That whole
> stack — llamafit, llama.cpp, llama-server, GGUF — was removed in 2026-06. The
> abstraction below is the one that replaced it.

---

## The seam: `engine.Backend`

`llm/engine/backend.go` defines a single interface — a tensor handle plus the op
set a transformer forward pass needs:

- creation / readback: `FromFloats`, `FromInt32`, `FromRaw` (f16/bf16/u32),
  `Floats`, `Ints`
- math: `MatMul` (batched + broadcast), `Add`/`Mul`, `Softmax`, `RMSNorm`,
  `LayerNorm`, activations (`SiLU`/`Gelu`/`Erf`/`Tanh`), `Mean`
- attention: `RoPE` / `RoPEFreqs`, `SDPA` (GQA + causal)
- shape: `Reshape`, `Transpose`, `ExpandDims`, `Concat`, `Slice`, `TakeAxis`
- 4-bit affine quant: `Dequantize`, `QuantMatmul`
- misc: `Cast`, `Argmax`, `ScalarMul/Add`, `Close`
- optional: `engine.Sweeper` (manual buffer lifetime — type-asserted, no-op if absent)

The **model code depends only on this interface** — `llm/qwen` (Qwen2/Qwen3/
Llama chat), `llm/gemma`, `llm/bge` (bge-m3 embeddings), `llm/reranker` (bge
cross-encoder). It never imports a specific backend and never calls C. New
hardware = a new `engine.Backend` impl; the model code doesn't change.

## Two implementations

| Backend | Package | Where it runs | How |
|---|---|---|---|
| **MLX-native** | `llm/mlxc` | Apple Silicon | cgo → MLX-C → Metal GPU + unified memory; manual buffer lifetime via `engine.Sweeper` (`mlx_clear_cache`, cache-limit, memory stats) |
| **Pure-Go CPU** | `llm/cpu` | Linux, Intel | eager, no cgo; Go's GC manages memory (no `Sweeper`); parallel matmul/GEMM across `GOMAXPROCS` |

Selection is build-tagged so non-Apple builds never link the MLX cgo:

- `llm/backend_mlx.go` (`//go:build darwin && arm64`) → `mlxc.New()`
- `llm/backend_cpu.go` (everything else) → `cpu.New()`
- `newBackend()` is the only constructor the high-level `llm` package calls.
- `JANIS_BACKEND=cpu` forces the CPU backend on a Mac (validation + fallback).
- `Accelerated()` reports GPU vs CPU (true on MLX, false on CPU) — used to size
  the model (below).

On Linux, `go list -deps ./llm` shows **0** `mlxc` deps and **1** `cpu` dep.

## CPU backend: SIMD kernels + runtime detection (no env var)

The pure-Go CPU backend isn't only plain Go: the hot path (the 4-bit quantized
decode GEMV, ~65-80% of decode CPU) has hand-written SIMD kernels, selected by
**runtime CPU-feature detection** (`golang.org/x/sys/cpu`) with no env var:

| Arch | Kernel | Gate |
|---|---|---|
| amd64 | **AVX2** (`qmv_amd64.s`) — 256-bit, 8 nibbles/word/vector | `cpu.X86.HasAVX2 && HasFMA` (Haswell 2013+) |
| arm64 | **NEON** (`qmv_arm64.s`) — 128-bit | `cpu.ARM64.HasASIMD` (ARMv8 baseline) |
| anything else / no feature | portable blocked-Go (4-channel register blocking) | — |

`init()` in `qmv_<arch>.go` flips `quant4SIMD`/`simdName`; `qChannel4Accel` only
runs the kernel when the feature is present (older x86 → safe fallback, never a
SIGILL). The kernel `.s` files are tagged by *arch only*, so they apply across
OSes (a future windows/amd64 would inherit AVX2 for free). `memdoor llm status`
prints the selected path, e.g. `Engine: CPU backend, SIMD: AVX2 (linux/amd64)`.

Speed ladder (this is where the latency wins came from, each token-exact vs MLX):
fused decode (no per-element div/mod, no scratch buffer) → 8-nibble unroll with
dual accumulators → 4-channel register blocking (x reused, shared ΣX) → arch SIMD.
On a 1B-4bit model that took the pure-Go backend ~0.1 → ~7.6 tok/s, +~14% more
from NEON on arm64 Linux; AVX2 is CI-validated for correctness on real x86 (the
arm64 dev box can't run AVX2, so GitHub's amd64 runners are the test rig).

## Correctness: MLX is the oracle

The CPU backend must reproduce MLX bit-for-bit, or a model emits fluent garbage.
`llm/cpu/cpu_mlx_diff_test.go` (`//go:build darwin && arm64`) runs identical
inputs through both backends and asserts they match — MatMul (0), Softmax/RMSNorm/
LayerNorm/RoPE/RoPEFreqs (~1e-7), SDPA (~6e-8), Dequantize (0), QuantMatmul
(~1.5e-5). A Linux-runnable hand-computed `TestDequantize4bit` covers the quant
path where the MLX harness can't run.

This caught a real bug end-to-end: every unit op matched MLX, yet a llama3-scaled
model produced gibberish — `RoPEFreqs` must **divide** by the supplied freqs (MLX
semantics), not multiply. Lesson: unit-exact ops don't guarantee an exact forward;
run a real model.

## 4-bit on CPU

`Dequantize`/`QuantMatmul` reproduce MLX affine quantization (LSB-first packing,
`scale*q + bias` per group), so a `Qwen3-8B-4bit` MLX checkpoint runs unchanged on
the CPU backend — no separate f16 model. `QuantMatmul` is a fused parallel GEMM
(dequantize each weight row once, dot against x, parallel over output channels),
which took the CPU backend ~18× over the naive first cut.

## Backend-aware model selection

On a GPU the limit is RAM; on CPU it's compute. `memdoor llm auto` and the
gateway's `resolveMLXModelDir` both branch on `llm.Accelerated()`:

- **GPU (MLX):** scale with RAM — Qwen3 4B → 8B → 14B → 32B.
- **CPU:** cap at a small **Qwen3-4B** regardless of RAM (an 8B fits but decodes
  too slowly). The resolver also caps the largest *cached* model it will load.

`hostRAMGiB` reads `/proc/meminfo` on Linux, `sysctl hw.memsize` on macOS.

## Why a duck-typed seam (the principle)

The consumer (model code) declares the narrow behavior it needs (`engine.Backend`);
concrete backends just *have* the methods. No `runtime.GOOS` branches in the model,
no env-var matrix, no registration. A future backend — ROCm, CUDA, WebGPU, a remote
worker — slots in by satisfying the interface, and every model + the diff-test
oracle works unchanged. This is the project's duck-typing rule (#5) applied to the
hottest seam in the system.
