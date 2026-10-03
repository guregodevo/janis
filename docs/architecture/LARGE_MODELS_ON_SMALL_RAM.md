# Running a 32B-class model on 16 GB — design note

**Status:** research / design. No code yet — this maps the problem, kills the
approaches that physics rules out, and proposes a build order. Written to decide
*whether and how* memdoor should run a model bigger than RAM on a 16 GB Mac
(today we cap at `Qwen3-8B-4bit`; see `cmd/cli/cmd/llm.go` `pickModel` and
`gateway/providers/mlx.go` `resolveMLXModelDir`).

The question: *"Can we run a 32 GB model in 16 GB? Split the file into sub-files +
lazy-load, or use MoE?"* Short answer: **the two intuitions are both right, and
they're the same answer.** A dense 32B streamed off SSD is non-interactive by
physics. An MoE model (we already cache `Qwen3-30B-A3B-4bit`) makes lazy loading
*fast* because only a small, locality-rich slice of weights is touched per token.
The work is: sharded loading → MoE architecture → expert offloading.

---

## 1. Why this is hard: decode is memory-bandwidth-bound

Two phases, opposite characters:

- **Prefill** (process the prompt) is *compute*-bound: big matmuls over many
  tokens at once, weights reused across the batch. Streaming weights from disk
  here is fine — the I/O amortizes over the work.
- **Decode** (generate, one token at a time) is *memory-bandwidth*-bound: each
  new token's forward pass reads **every weight the path touches, once**, and
  does only a thin vector·matrix against it. There's no batch to amortize over.

So the governing number for "can I generate at a usable speed" is **bytes of
weight touched per token ÷ the bandwidth of wherever those bytes live.** If the
weights fit in unified memory, that's ~100–400 GB/s (fast). If they spill to
SSD, it's ~3–6 GB/s (≈50–100× slower). That cliff is the whole problem.

This matches our own past lesson (`feedback_measure_call_path`): the bottleneck
isn't "the model is big," it's *which memory the per-token reads hit*. Measure the
call path, not the file size.

## 2. The dispositive math — dense 32B streamed on 16 GB

`Qwen3-32B-4bit` ≈ 16–18 GB of weights. On a 16 GB Mac it does **not** fit
resident alongside the KV cache, activations, and the OS — so some fraction is
re-read from SSD every token.

```
weights touched / token ≈ 16 GB (dense: the whole model, every token)
Apple NVMe read         ≈ 5 GB/s  (optimistic; M1 ~2.5–3, M3 ~4–6)
→  ~3.2 s / token  ≈  0.3 tok/s
```

Even if half the model stays resident and you only stream 8 GB/token, that's
~1.6 s/token. **Dense streaming is non-interactive — physics says no, before we
write a line of code.** `mmap` (which our reader already does conceptually, see
§5) doesn't change this: it just lets the OS page-cache thrash on your behalf.
The bytes still cross the SSD bus once per token. This is the same wall
`llama.cpp` partial offload hits — past a point, more offload = linearly slower.

**Conclusion:** "fit a *dense* 32B in 16 GB and decode usefully" is not
achievable. The two ways out are (a) make the bytes-per-token smaller (MoE), or
(b) make the bytes smaller still (more aggressive quant). Only one of those is
robust.

## 3. The three levers

| Lever | What it changes | Fits 16 GB? | Verdict |
|---|---|---|---|
| **Aggressive low-bit quant** (2–3 bit) | bytes/weight | 32B@~2.5bit ≈ 10–12 GB | **Fragile.** Uniform 2–3 bit *breaks MoE/large models* — NaNs / near-random output (the research is consistent on this). Needs mixed precision (attention high-bit, FFN low-bit). A complementary lever, not a standalone answer. |
| **Dense weight streaming** (sub-files + lazy load) | where weights live | "fits" but thrashes | **No for decode** (§2). Fine for a one-shot prefill/embedding pass; useless for interactive generation. |
| **MoE + expert offloading** | bytes *touched* / token | yes, with caching | **The answer.** 30B total, ~3.3B active/token, and active experts have temporal locality → small resident working set. |

The trap is treating these as alternatives. The winning design is **MoE as the
architecture + lazy expert loading as the mechanism**, with optional mixed-bit
quant as a multiplier.

## 4. Why MoE turns "lazy loading" from slow into viable

`Qwen3-30B-A3B`: 30.5B total params, **3.3B active per token**, 128 experts per
layer, **8 fire per token**, 48 layers, GQA. (We already have
`Qwen3-30B-A3B-4bit` in the HF cache.)

The forward pass per token splits into:

- **Always-on** (~small): embeddings, attention, layernorms, the router/gate.
  Keep these *resident*. A few GB at 4-bit.
- **Experts** (~most of the 30B): 128 FFNs per layer, only 8 selected per token.

So the bytes *touched* per token are roughly the active experts: ~3.3B params
minus attention ≈ **~1.2–1.5 GB at 4-bit**, not the full 15 GB. Cold-cache worst
case at 5 GB/s ≈ 0.3 s/token (~3 tok/s) — already borderline usable. But experts
have **locality**: consecutive tokens reuse many of the same experts (the
literature reports high hit rates with simple caches). So with an LRU/LFU expert
cache sized to most of the 16 GB, the *actual* cold reads per token drop to a
fraction of 1.5 GB → comfortably interactive.

This is the asymmetry that makes MoE the answer: dense streaming re-reads 16 GB
every token forever; MoE re-reads only the *cold* active experts — a fraction of
that ~1.5 GB, set by the cache hit rate (measured in §4½). Same SSD, completely
different outcome.

State of the art to borrow from (all 2025–26): expert **caching** (LRU/LFU),
**speculative prefetch** (predict next layer's experts from the current layer's
hidden state and load them while you compute), and **domain/prefill-aware
preloading**. See references.

## 4½. Measured: a routing-locality spike (the dispositive number)

Before building anything, we measured the number §4 hand-waved — *how localized
is expert selection?* — because the whole bet depends on it. Harness:
[`research/moe_routing_spike/`](../../research/moe_routing_spike/). It instruments
the router on **OLMoE-1B-7B (top-8 of 64)** — a proxy for `Qwen3-30B-A3B`
(top-8 of 128); the routing *shape* matches and locality is a property of the
routing, not the param count — over memdoor-style grounded-Q&A prompts (532
tokens), then simulates an LRU expert cache.

```
token-to-token expert reuse:   mean 38.5%   (only ~38% carry over per token)
distinct experts used / layer: 63.8 of 64   (≈ ALL experts get used over a run)

LRU hit-rate vs resident experts/layer:
   K=8  (12%) → 28%      K=32 (50%) → 77%      K=64 (100%) → 98.5%
   K=16 (25%) → 51%      K=48 (75%) → 91%
```

**This is a yellow light, not a green one — and it's the honest result.** Expert
locality is *modest*, not high: training's load-balancing loss spreads usage
across the whole pool, so a small cache can't win — you need ~75% of experts
resident to clear ~90% hit-rate. The naive hope ("tiny hot set, stream the
rest") is wrong.

**What saves the specific case is arithmetic.** `Qwen3-30B-A3B-4bit` ≈ **16.8 GB**
— only ~5% over 16 GB. So keep the non-expert weights + ~60–75% of experts
resident (~11–12 GB) and stream only the **cold expert tail**. At ~83–90% hit,
cold reads ≈ **~100–150 MB/token**, not 16 GB/token:

```
cold I/O / token ≈ 150 MB ÷ ~5 GB/s ≈ ~30 ms     (AirLLM dense: ~16 GB → ~3 s)
compute (3.3B active)               ≈ ~30–60 ms
→ projected ~8–15 tok/s, COMPUTE-bound — I/O is not the wall for this size
```

That's ~30–50× faster than dense per-layer streaming (which re-reads the whole
model every token). **Prior art:** AirLLM (`github.com/lyogavin/airllm`) does
exactly that dense per-layer load→run→free, *with* prefetch and compression, and
has **no** expert/MoE awareness — which is why it's slow interactively and why
expert-tail offloading (not "streaming") is the part worth building.

**The boundary, stated plainly:** this works *because 30B-A3B barely exceeds RAM.*
At 38% locality, a model **≫ RAM** (e.g. 70B) needs a small resident fraction,
the hit-rate collapses, and you fall back to the AirLLM I/O wall. The honest
claim is narrow: *run a 30B-class MoE that's ~1.05–1.3× your RAM, interactively*
— not "run any huge model in any small RAM." Caveat on the proxy: 128 experts
(vs OLMoE's 64) at the same top-8 may spread *thinner* → possibly lower hit-rate
at a given %-resident; but since 30B-A3B only slightly exceeds RAM we can afford
a high resident %, landing in the safe part of the curve. Replace the proxy with
the real 30B router curve before building PR #3.

## 5. What our stack has today (the real seams + the gaps)

Reading the code, three concrete prerequisites surface:

1. **Single-file loader.** `llm/engine.go` opens exactly
   `model.safetensors`. `llm/safetensors/safetensors.go` is a nice lazy
   *byte-range* reader (it indexes without reading data, `Get` reads a range on
   demand) — but it's one file. A 30B model ships **sharded**
   (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`). This is
   exactly the "split into sub-files" idea — it already exists as the HF sharding
   convention; **we just can't read it yet.** Prerequisite #1.

2. **Eager pin.** `qwen.LoadModel` pulls every tensor via `Get`, copies it into a
   backend tensor (`FromRaw`), and `pinAll(e.bk)` pins them resident for the
   process lifetime. So despite the "lazy" reader, **the whole model is
   materialized and pinned at `Open()`.** Lazy *expert* loading means changing
   this: keep the safetensors `File`(s) mapped, and fetch expert weights on
   demand into an LRU-managed pool instead of pinning all of them. Prerequisite
   #3 lives here.

3. **No MoE.** `llm/qwen/` is dense (`attention.go`, `mlp.go`, `block.go`, …) —
   no router, no experts, no top-k gate. `Qwen3-30B-A3B` won't load until that
   exists. Prerequisite #2.

Note on MLX (`llm/mlxc`): MLX *natively* mmaps safetensors and is lazy, but we
deliberately load through our own reader + `FromRaw`, so we don't get MLX's lazy
mmap for free — the offload policy will be ours to implement on either backend
(it belongs above `engine.Backend`, so MLX and CPU both inherit it). Watch the
MLX cgo buffer lifetime (`engine.Sweeper`) when experts are evicted.

## 6. Proposed build order (one PR each)

1. **Sharded safetensors** — read `*.index.json` + N shards behind the existing
   `safetensors.File` API. Unlocks loading *any* large model (not just MoE).
   Standalone, testable, low risk. Validates against a known sharded model
   byte-for-byte.
2. **MoE architecture** — router + top-k gate + per-expert FFN in `llm/qwen`,
   diff-tested token-exact vs MLX on `Qwen3-30B-A3B` (our existing
   `cpu_mlx_diff_test.go` oracle pattern). At this point the model *runs* but
   still wants ~15 GB resident — i.e. not yet on 16 GB, but correct.
3. **Expert offloading** — stop pinning experts; LRU pool + on-demand fetch from
   the mapped shards; optional speculative prefetch. This is the PR that actually
   lands 30B on 16 GB.

Each step is independently useful and independently verifiable — no big-bang.

## 7. Measure before building (empirical discipline)

Per `feedback_empirical_discipline` / `feedback_measure_call_path`, get
dispositive numbers *before* committing to a caching policy:

- **SSD read bandwidth** on the target Mac (don't trust the 5 GB/s assumption —
  measure it; M1 vs M3 differ 2×).
- **Expert hit rate / locality**: ✅ done (§4½, OLMoE proxy). Result: locality is
  modest (38% reuse), so offloading only pays for a model *slightly* over RAM;
  for 30B-A3B that projects to interactive (~8–15 tok/s, compute-bound). Still
  **to do before PR #3**: re-run the harness on the *real* `Qwen3-30B-A3B` router
  (128 experts) to replace the proxy curve — needs a machine that can load it.
- **Prefill vs decode split** for memdoor's actual ask path (short grounded
  answers): if our outputs are short, decode cost dominates and the cache must be
  warm fast; if prefill dominates, streaming is cheaper than it looks.

## 8. Open questions / risks

- **2–3 bit on MoE → NaN/garbage** unless mixed-precision. If we want 30B *and*
  headroom, we likely need attention@4–6bit + experts@2–3bit. Quality must be
  benchmarked (we have a benchmark harness — signal, not target).
- **MLX cgo eviction**: paging experts in/out must respect MLX buffer lifetime;
  the CPU (GC) backend is simpler to prototype on first.
- **Prefetch overlap**: hiding expert I/O behind compute needs async loads; if
  the engine is synchronous per layer, gains are capped at "don't re-read cached
  experts."
- **Cold start**: first tokens are all cache misses. Acceptable for a chat turn,
  worse for one-shot CLI calls — may want a warm pool.

## References

- Qwen3-30B-A3B specs — [apxml](https://apxml.com/models/qwen3-30b-a3b),
  [Jetson AI Lab](https://www.jetson-ai-lab.com/models/qwen3-30b-a3b/)
- MoE offloading / caching / prefetch —
  [DALI (workload-aware offloading, arXiv)](https://arxiv.org/html/2602.03495v1),
  [SpecMD (speculative expert prefetch, arXiv)](https://arxiv.org/pdf/2602.03921),
  [OD-MoE (on-demand loading, arXiv)](https://arxiv.org/html/2512.03927v1),
  [Expert offloading to CPU/NVMe (apxml course)](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading)
- MLX larger-than-RAM / lazy mmap / low-bit limits —
  [mlx-lm #876 (chunked conversion)](https://github.com/ml-explore/mlx-lm/issues/876),
  [Explore LLMs on Apple silicon with MLX (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/298/)

Related internal: [`LOCAL_ACCELERATOR_ABSTRACTION.md`](LOCAL_ACCELERATOR_ABSTRACTION.md)
(the `engine.Backend` seam the offload policy sits above).
