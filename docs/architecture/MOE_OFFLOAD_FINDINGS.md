# MoE expert-offloading on a 16 GB Mac — engine findings

**Status:** working, bounded, slow. The native Go engine runs `Qwen3-30B-A3B-4bit`
on a 16 GB M4 — a model MLX OOMs on and llama.cpp can't run interactively. These
are the measured findings from building and profiling it (branch
`feat/moe-engine`).

## What works
- Sharded safetensors loading (`StackedRow` reads one expert without the stack).
- `ExpertStore`: byte-range expert materialization (`StackRaw`), bounded.
- Qwen3-MoE forward (router → top-k → per-expert SwiGLU → score-weighted sum),
  token-exact in behavior (output matches mlx-lm / the Python prototype).
- Fused `GatherQuantMatmul` (binds `mlx_gather_qmm`) — one call per projection.
- Runs end-to-end with **no OOM**, ~3–4 GB resident, where MLX OOMs.

## Performance (Qwen3-30B-A3B-4bit, base M4, 16 GB)
| phase | before | after | note |
|---|---|---|---|
| decode | ~2 tok/s | ~2 tok/s | bounded; not yet interactive |
| prefill (short prompt) | ~25–35 s | **~10 s** | fixed: per-token stream-free prefill |
| peak memory | ~3–4 GB | ~3–4 GB | bounded across a 200-token run |

For reference: 8B-4bit (fits resident) = 22 tok/s; the **hardware ceiling** for
8B on this chip (~120 GB/s) is ~25 tok/s. The Python offload prototype hit
~3.74 tok/s decode.

## The decisive measurements (what's true vs what was assumed)

**1. There is NO memory leak.** Sampled RSS every 12 s over a 200-token run:
stayed in a 2.9–4.2 GB band, peak (4.17 GB) hit at t=48 s and never exceeded for
the next 72 s, ended at 3.3 GB, completed cleanly. A leak climbs monotonically to
OOM; this is flat/sawtooth (Sweep + GC reclaiming each step). Earlier OOMs were
the *cache/gather experiments* (design flaws, reverted), not leaks; the ~16 GB of
"used" memory people saw was orphaned **swap** from those killed runs plus OS
**page cache** (both reclaimable; `sudo purge` clears the swap).

**2. On 16 GB there is no free RAM to cache experts.** The model's experts total
~16 GB; the OS page cache can't hold them alongside the process, and an in-RAM
expert cache OOMs. So every token reads its active experts cold from disk —
that's inherent, not a bug.

**3. Caching experts made it SLOWER, not faster** (0.60 → 0.37 tok/s, reverted).
This is the key disproof: **the bottleneck is NOT expert data movement.** Caching
added pinning + a per-token stack copy and lost; if data movement were the
bottleneck, caching would have won. The real cost is elsewhere — per-layer router
host-syncs + cgo per-op overhead + MLX memory pressure.

**4. `mlx_stack` (single multi-input stack) vs pairwise `Concat`:** pairwise
Concat blows memory ~4.4× (growing intermediates) and OOMs under stream-free
decode. The single-op `mlx_stack` avoids it — but since caching didn't help, the
shipped path is `StackRaw` (one `FromRaw` per projection, no cache, no Concat).

**5. Prefill was the real problem, and it's fixable.** Prefill ran one batched
per-layer-eval pass that re-materialized ~all the prompt's experts at once (48
forced GPU syncs + ~120 experts/layer). Fix: prefill **one token at a time**
through the stream-free decode path (`Session.PrefillChunk = 1` for MoE; reuses
the existing chunked-prefill Pin/Eval/Sweep loop). Measured prefill-dominated
(MAXTOK=4): 36.1 s → 12.7 s (**~3×**). Now automatic for MoE models.

## Honest verdict
Decode at ~2 tok/s is **bounded and correct but not interactive** (the hardware
caps even a resident 8B at ~25 tok/s, and a streamed-from-disk 30B is far below
that). Reaching the Python prototype's 3.74 — let alone interactive — needs work
on the *actual* bottleneck, which the data says is **per-layer host syncs + cgo
per-op overhead**, not expert I/O. That means:
- On-device top-k so there's no host sync per layer (hard: offload needs host
  indices to fetch experts; only clean if experts are resident, which reopens the
  memory wall).
- Fewer cgo crossings (batch ops across layers); async eval.
Profile first — the one surprise here (caching hurt) means guessed fixes are
risky; measure where each decode token's time actually goes before optimizing.

## Bottom line
A 30B MoE runs on a 16 GB Mac, no OOM, no leak, prefill ~3× faster than the first
cut. It's a proven capability and a clean systems artifact — but at ~2 tok/s it's
a "works, not usable for chat" result. Whether to push decode further is a
goal-dependent call (see LARGE_MODELS_ON_SMALL_RAM.md): for the product, an 8B at
22 tok/s already covers the job; this matters mainly as a "30B on a laptop"
capability/credibility moment.
