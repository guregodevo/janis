"""MoE routing-locality spike.

Measures the ONE number that decides whether expert offloading beats the I/O
wall: how often a token's active experts are already cached (the LRU hit-rate vs
resident-budget curve), plus raw token-to-token expert reuse.

Runs on OLMoE-1B-7B (top-8 of 64) as a PROXY for Qwen3-30B-A3B (top-8 of 128) —
the routing *shape* (top-8 sparse) matches; absolute numbers are directional and
projected to 30B via config arithmetic in the report. We can't run the real 30B
on 16 GB, and locality is a property of the routing, not the param count.
"""
import os, sys, collections
import mlx.core as mx
import mlx_lm
from mlx_lm.models import olmoe

REPO = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"

# --- instrument the router: record the experts each layer selects per token ---
traces = collections.defaultdict(list)   # layer_id -> [frozenset(experts) per token]
_orig = olmoe.OlmoeSparseMoeBlock.__call__
_layer_ids = {}

def patched(self, x):
    lid = _layer_ids.setdefault(id(self), len(_layer_ids))
    x_flat = x.reshape(-1, x.shape[-1])
    logits = self.gate(x_flat)
    k = self.top_k
    sel = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    mx.eval(sel)
    for row in sel.tolist():            # one row per token in this call
        traces[lid].append(frozenset(row))
    return _orig(self, x)

olmoe.OlmoeSparseMoeBlock.__call__ = patched

# --- run representative (memdoor-style grounded Q&A) prompts ---
PROMPTS = [
    "What is the breakout volume rule and how is position size computed?",
    "Summarize the risk-management rules for entering a trade.",
    "Explain how the cited answer is grounded in the source pages.",
    "What changed in the deployment between the two most recent versions?",
    "Describe the steps to onboard a new user from install to first answer.",
]

model, tok = mlx_lm.load(REPO)
NTOK = int(os.environ.get("NTOK", "80"))
for p in PROMPTS:
    msgs = [{"role": "user", "content": p}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    # generate; the patched router logs every layer/token selection as a side effect
    for _ in mlx_lm.stream_generate(model, tok, prompt, max_tokens=NTOK):
        pass

# --- analysis ---------------------------------------------------------------
def lru_hitrate(seq, K):
    """Hit rate over all (token -> its experts) requests for an LRU cache of K
    experts. A hit = the needed expert was already resident."""
    cache = collections.OrderedDict()
    hits = reqs = 0
    for experts in seq:
        for e in experts:
            reqs += 1
            if e in cache:
                hits += 1
                cache.move_to_end(e)
            else:
                cache[e] = True
                if len(cache) > K:
                    cache.popitem(last=False)
    return hits / reqs if reqs else 0.0

def reuse(seq):
    """Avg fraction of a token's experts that were also active the previous token."""
    if len(seq) < 2:
        return 0.0
    tot = ov = 0
    for a, b in zip(seq, seq[1:]):
        ov += len(a & b)
        tot += len(b)
    return ov / tot if tot else 0.0

layers = sorted(traces)
n_experts = max(max(e) for s in traces.values() for e in s) + 1
top_k = len(next(iter(traces[layers[0]])))
tokens = len(traces[layers[0]])
print(f"\n=== MoE routing spike — proxy {REPO} ===")
print(f"layers={len(layers)} experts/layer={n_experts} top_k={top_k} tokens_logged={tokens}")

# raw locality (token-to-token reuse), averaged over layers
reuses = [reuse(traces[l]) for l in layers]
print(f"\ntoken-to-token expert reuse: mean={sum(reuses)/len(reuses):.1%} "
      f"min={min(reuses):.1%} max={max(reuses):.1%}")

# distinct experts actually used per layer over the run (the working set)
ws = [len(set().union(*traces[l])) for l in layers]
print(f"distinct experts used / layer (of {n_experts}): "
      f"mean={sum(ws)/len(ws):.1f} min={min(ws)} max={max(ws)}")

# LRU hit-rate vs per-layer cache budget K (averaged over layers)
print(f"\nLRU hit-rate vs resident experts per layer (top_k={top_k} active/token):")
print(f"  {'K':>4}  {'%of experts':>11}  {'hit-rate':>9}")
for K in sorted(set([top_k, top_k*2, n_experts//4, n_experts//2, int(n_experts*0.75), n_experts])):
    if K < top_k: continue
    hr = sum(lru_hitrate(traces[l], K) for l in layers) / len(layers)
    print(f"  {K:>4}  {K/n_experts:>10.0%}  {hr:>8.1%}")
