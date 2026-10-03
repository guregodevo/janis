# MoE routing-locality spike

Measures the one number that decides whether **expert offloading** beats the
SSD I/O wall (see [`../../docs/architecture/LARGE_MODELS_ON_SMALL_RAM.md`](../../docs/architecture/LARGE_MODELS_ON_SMALL_RAM.md)):
how often a token's active experts are already cached — the **LRU hit-rate vs
resident-budget curve** — plus raw token-to-token expert reuse.

It instruments the router (`OlmoeSparseMoeBlock.__call__`) to log the experts
each layer selects per token over memdoor-style grounded-Q&A prompts, then
simulates an LRU expert cache at several budgets.

## Proxy, not the target

Runs on **OLMoE-1B-7B (top-8 of 64)** as a stand-in for **Qwen3-30B-A3B
(top-8 of 128)**: the routing *shape* (top-8 sparse) matches, and locality is a
property of the routing, not the parameter count. We can't *run* the real 30B on
16 GB. Absolute numbers are directional; the design note projects them to 30B via
config arithmetic. To replace the proxy with the real curve, point `REPO` at the
30B model on a machine that can load it.

## Run

```bash
python3.14 -m venv venv && ./venv/bin/pip install mlx mlx-lm
NTOK=80 ./venv/bin/python measure_routing.py
```

## Result (2026-06-23, OLMoE proxy, 532 tokens)

```
token-to-token expert reuse:   mean 38.5%
distinct experts used / layer: 63.8 of 64        (≈ all experts used over a run)
LRU hit-rate:  K=8 →28%  K=16 →51%  K=32 →77%  K=48 →91%  K=64 →98.5%
```

Read: locality is **modest** (load balancing spreads usage across all experts),
so offloading only wins when a **large** fraction of experts stays resident —
which is feasible for a model only slightly over RAM (30B-A3B ≈ 16.8 GB), and
not for a model ≫ RAM. See the design note's Feasibility section for the full
tok/s projection and the boundary.
