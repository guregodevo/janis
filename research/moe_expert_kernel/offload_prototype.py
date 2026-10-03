"""
MoE expert-offloading prototype: run Qwen3-30B-A3B-4bit on a 16 GB Mac that
MLX OOMs on, by keeping the expert stacks OFF-device (mmap'd) and materializing
only the router's active experts per token (LRU-cached) before gather_qmm.

Non-expert weights (embed, attn, norms, router gate, lm_head) load resident.
The 128-expert stacks (switch_mlp.*) are never made resident; we byte-slice the
active ~8 per layer from a numpy.memmap and mx.array just those.

The measurement: does it generate at all (MLX can't), and at what tok/s.
"""
import json, os, sys, glob, time, collections
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from mlx_lm.models import qwen3_moe, switch_layers
from mlx_lm.utils import load_tokenizer

DT = {"U32": np.uint32, "I32": np.int32, "F16": np.float16, "F32": np.float32,
      "U16": np.uint16, "BF16": "bf16", "U8": np.uint8, "I8": np.int8}
DTSZ = {"U32": 4, "I32": 4, "F16": 2, "F32": 4, "U16": 2, "BF16": 2, "U8": 1, "I8": 1}


def to_mx(buf, dt, shape):
    if dt == "BF16":
        a = mx.array(np.frombuffer(buf, np.uint16).copy())
        return a.view(mx.bfloat16).reshape(shape)
    return mx.array(np.frombuffer(buf, DT[dt]).copy().reshape(shape))


class ExpertStore:
    """Byte-range gather of active experts from memmap'd shards, with an LRU
    cache of materialized mx.arrays keyed (layer, proj, kind, expert)."""
    def __init__(self, model_dir, budget_gb=6.0):
        self.idx = {}      # tensor name -> (mm, dtype, shape, abs_start)
        self.mms = {}
        for sf in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
            with open(sf, "rb") as f:
                hlen = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(hlen))
            mm = np.memmap(sf, dtype=np.uint8, mode="r")
            data_start = 8 + hlen
            for name, e in hdr.items():
                if name == "__metadata__":
                    continue
                self.idx[name] = (mm, e["dtype"], tuple(e["shape"]),
                                  data_start + e["data_offsets"][0])
        self.keys = set(self.idx.keys())
        self.cache = collections.OrderedDict()
        self.bytes = 0
        self.budget = int(budget_gb * 1e9)
        self.hits = self.misses = 0

    def _expert(self, name, e):
        # direct memmap view → one copy into MLX (no bytes()/frombuffer churn)
        mm, dt, shape, start = self.idx[name]
        per = 1
        for s in shape[1:]:
            per *= s
        nb = per * DTSZ[dt]
        off = start + e * nb
        sl = mm[off:off + nb]
        if dt == "BF16":
            a = mx.array(sl.view(np.uint16).copy()).view(mx.bfloat16)
        else:
            a = mx.array(sl.view(DT[dt]).copy())
        return a.reshape(shape[1:])

    def gather(self, layer, proj, experts):
        """Return stacked (weight, scales, biases) for the given expert ids."""
        out = {"weight": [], "scales": [], "biases": []}
        new = []
        for e in experts:
            for kind in out:
                key = (layer, proj, kind, e)
                t = self.cache.get(key)
                if t is None:
                    self.misses += 1
                    name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{kind}"
                    t = self._expert(name, e)
                    self.cache[key] = t
                    self.bytes += t.nbytes
                    new.append(t)
                    while self.bytes > self.budget and len(self.cache) > 3:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old.nbytes
                else:
                    self.hits += 1
                    self.cache.move_to_end(key)
                out[kind].append(t)
        if new:
            mx.eval(new)        # one batched eval per gather, not per expert
        return (mx.stack(out["weight"]), mx.stack(out["scales"]), mx.stack(out["biases"]))


STORE = None


def patched_call(self, x, indices, sorted_indices=False):
    layer, proj = self._eid
    inds_np = np.array(indices.tolist(), dtype=np.int64)
    uniq = np.unique(inds_np)
    idmap = np.zeros(STORE.num_experts, dtype=np.int32)
    idmap[uniq] = np.arange(len(uniq), dtype=np.int32)   # fast remap, no vectorize
    w, s, b = STORE.gather(layer, proj, uniq.tolist())
    remap = mx.array(idmap[inds_np])
    return mx.gather_qmm(x, w, s, b, rhs_indices=remap, transpose=True,
                         group_size=self.group_size, bits=self.bits, mode=self.mode,
                         sorted_indices=False)


def main():
    global STORE
    model_dir = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots/*/"))[0]
    config = json.load(open(os.path.join(model_dir, "config.json")))
    args = qwen3_moe.ModelArgs.from_dict(config)
    print(f"building Qwen3-30B-A3B: {args.num_hidden_layers} layers, "
          f"{args.num_experts} experts, top-{args.num_experts_per_tok}")

    STORE = ExpertStore(model_dir, budget_gb=12.5)  # parses headers + memmaps
    STORE.num_experts = args.num_experts
    model = qwen3_moe.Model(args)        # expert params stay LAZY (never eval'd)

    # Quantize the structure to match the saved 4-bit format. This also turns the
    # experts into QuantizedSwitchLinear (which our patch targets). Mirror mlx-lm:
    # quantize a module iff its .scales are present on disk.
    cfgq = config["quantization"]
    def class_predicate(p, m):
        if p in cfgq:
            return cfgq[p]
        return hasattr(m, "to_quantized") and f"{p}.scales" in STORE.keys
    nn.quantize(model, group_size=cfgq["group_size"], bits=cfgq["bits"],
                mode=cfgq.get("mode", "affine"), class_predicate=class_predicate)

    # load only non-expert weights resident (expert stacks stay lazy & unused)
    nonexpert = {}
    for sf in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
        for k, v in mx.load(sf).items():
            if "switch_mlp" not in k:
                nonexpert[k] = v
    model.load_weights(list(nonexpert.items()), strict=False)
    mx.eval(list(nonexpert.values()))          # force non-expert resident
    print(f"non-expert weights loaded; active mem = {mx.get_active_memory()/1e9:.2f} GB")

    # tag each switch linear with its (layer, proj) identity + patch
    for i, layer in enumerate(model.model.layers):
        smlp = getattr(layer.mlp, "switch_mlp", None)
        if smlp is None:
            continue
        for proj in ("gate_proj", "up_proj", "down_proj"):
            getattr(smlp, proj)._eid = (i, proj)
    switch_layers.QuantizedSwitchLinear.__call__ = patched_call

    tok = load_tokenizer(Path(model_dir))
    from mlx_lm import stream_generate
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is a mixture-of-experts model? One sentence."}],
        add_generation_prompt=True)

    print("generating (first tokens are cold-cache)...")
    n, t0, text = 0, None, ""
    for r in stream_generate(model, tok, prompt, max_tokens=40):
        if t0 is None:
            t0 = time.perf_counter()    # start timing after first token (excl. prefill)
        text += r.text
        n += 1
    dt = time.perf_counter() - t0
    print("\n--- output ---\n" + text)
    print(f"\n{n} tokens in {dt:.1f}s -> {(n-1)/dt:.2f} tok/s (decode)")
    print(f"expert cache: {STORE.hits} hits / {STORE.misses} misses "
          f"({STORE.hits/(STORE.hits+STORE.misses+1e-9):.0%} hit) "
          f"resident {STORE.bytes/1e9:.1f} GB")
    print(f"peak mem = {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
