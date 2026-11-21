"""Builds the tiny Inkling 2 models janis's port is checked against.

Run with the mlx-lm that has models/inkling2: ml-explore/mlx-lm pull
request 2086, here at commit 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 with
mlx 0.32.4:

    pip install "git+https://github.com/ml-explore/mlx-lm.git@0c4c99192230f1943dd468d297f0b8a64ae7b7c2"
    python gen_fixture.py

Writes, for each of f32 and bf16, a model directory (config.json,
model.safetensors, in the layout of the published MLX conversions: affine
4-bit projections, nvfp4 experts with a global scale, mxfp8 experts in the
last layer) and expected.safetensors: a token row, the logits of one full
forward pass, the output of every layer visit, and a greedy continuation
of the first tokens.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models import inkling2
from mlx_lm.utils import quantize_model

TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "vocab_size": 128,
    "unpadded_vocab_size": 120,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 16,
    "sliding_window_size": 8,
    "local_layer_ids": [0, 1, 2, 3, 4, 6],
    "d_rel": 4,
    "rel_extent": 8,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "qk_l2_clip_threshold": 3.0,
    "rel_l2_clip_threshold": 2.0,
    "sconv_kernel_size": 4,
    "n_layers_per_attn_res_block": 3,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "intermediate_size": 64,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 2.0,
    "final_logit_softcapping": 30.0,
    "loop_config": {
        "core_start": 3,
        "core_end": 6,
        "num_passes": 3,
        "gate_rank": 8,
        "is_varloop": False,
    },
}
CONFIG = {
    "model_type": "inkling2_loop",
    "eos_token_id": 119,
    "image_token_id": 127,
    "text_config": TEXT,
}
PROMPT, GREEDY = 8, 16
TOKENS = [5, 17, 3, 99, 42, 7, 7, 64, 110, 1, 28, 56, 91, 12, 0, 33, 75, 118, 2, 60]


def build(dtype) -> inkling2.Model:
    mx.random.seed(0)
    model = inkling2.Model(inkling2.ModelArgs.from_dict(CONFIG))
    cast = model.cast_predicate
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in**-0.5
        weights.append((k, v.astype(dtype) if cast(k) else v))
    model.update(tree_unflatten(weights))

    def experts(path, module):
        if ".mlp.experts." not in path:
            return False
        if ".layers.7." in path:
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}

    quantize_model(model, {}, None, None, quant_predicate=experts)
    base = model.quant_predicate

    def rest(path, module):
        return ".mlp.experts." not in path and base(path, module)

    quantize_model(model, {}, 64, 4, quant_predicate=rest)
    mx.eval(model.parameters())
    return model


VISION = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "native_patch_size": 2,
    "adapter_merge_size": 2,
    "posemb_grid_size": 4,
    "projection_hidden_dim": 48,
}


def vision(here: Path) -> None:
    """tiny-vision: the f32 model with a vision tower, an image of 8 by 12
    pixels (2 by 3 image tokens), its rows and the logits of a prompt that
    holds it."""
    global CONFIG
    text_only, CONFIG = CONFIG, {**CONFIG, "vision_config": VISION}
    model = build(mx.float32)
    config, CONFIG = CONFIG, text_only
    out = here / "tiny-vision"
    out.mkdir(exist_ok=True)
    config = {**config, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())))

    mx.random.seed(1)
    image = mx.random.uniform(-1, 1, shape=(3, 8, 12))
    rows = model.encode_image(image)
    token = config["image_token_id"]
    tokens = mx.array([[5, 17] + [token] * rows.shape[0] + [3, 99, 42]])
    expected = {
        "pixels": image,
        "image_rows": rows.astype(mx.float32),
        "tokens": tokens[0].astype(mx.int32),
        "logits": model(tokens, pixel_values=[image])[0],
    }
    mx.save_safetensors(str(out / "expected.safetensors"), expected)
    print("vision", rows.shape, expected["logits"].shape)


def alt_layout(here: Path) -> None:
    """tiny-alt: the f32 model in the layout MLX-VLM reads (the netprojects
    conversions): the tensors renamed, the experts requantized affine at 2
    and 3 bits with gate and up apart, sink_proj and weight_proj as floats.
    The expected logits are the reference's with the requantized experts
    put back as floats."""
    import re

    from mlx_lm.layers.moe import SwitchLinear

    model = build(mx.float32)
    out = {}
    renames = [
        (r"^model\.language_model\.", "language_model.model."),
        (r"^lm_head\.", "language_model.lm_head."),
        (r"\.q_l2_clip\.scale$", ".q_scale"),
        (r"\.rel_l2_clip\.scale$", ".rel_scale"),
        (r"\.rel_logits_proj\.proj$", ".rel_proj"),
        (r"\.conv1d\.weight$", ".conv.weight"),
        (r"\.post_attention_convolution\.", ".attn_sconv."),
        (r"\.post_mlp_convolution\.", ".mlp_sconv."),
        (r"\.mlp\.gate\.weight$", ".mlp.gate_weight"),
        (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.e_score_correction_bias"),
        (r"\.q_message_gate\.", ".controllers.attn."),
        (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
        (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
        (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
        (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
        (r"\.message_input\.query$", ".message_queries"),
    ]
    weights = dict(tree_flatten(model.parameters()))
    floats = ("sink_proj", "weight_proj")
    for k, v in weights.items():
        if ".mlp.experts." in k:
            continue
        name = k
        for pattern, to in renames:
            name = re.sub(pattern, to, name)
        m = re.match(r"(.+\.(?:%s))\.(weight|scales|biases)$" % "|".join(floats), k)
        if m:
            if m.group(2) == "weight":
                p = m.group(1)
                dense = mx.dequantize(v, weights[p + ".scales"], weights[p + ".biases"], 64, 4)
                out[name] = dense
            continue
        out[name] = v

    layers = model.model.language_model.layers
    for i, layer in enumerate(layers):
        bits = 3 if i in (0, len(layers) - 1) else 2
        base = f"language_model.model.layers.{i}.mlp.switch_mlp."
        experts = layer.mlp.experts
        put_back = {}
        for name in ("gate_up_proj", "down_proj"):
            q = getattr(experts, name)
            scale = q.get("global_scale")
            dense = mx.stack(
                [
                    mx.dequantize(
                        q.weight[e], q.scales[e], None, q.group_size, q.bits,
                        mode=q.mode, global_scale=None if scale is None else scale[e],
                    )
                    for e in range(q.weight.shape[0])
                ]
            ).astype(mx.float32)
            if name == "gate_up_proj":
                n, two, d = dense.shape
                pairs = dense.reshape(n, two // 2, 2, d)
                parts = {"gate_proj": pairs[:, :, 0], "up_proj": pairs[:, :, 1]}
            else:
                parts = {"down_proj": dense}
            again = {}
            for part, w in parts.items():
                qw, qs, qb = mx.quantize(w, 64, bits)
                out[base + part + ".weight"], out[base + part + ".scales"], out[base + part + ".biases"] = qw, qs, qb
                again[part] = mx.dequantize(qw, qs, qb, 64, bits)
            if name == "gate_up_proj":
                n, half, d = again["gate_proj"].shape
                w = mx.stack([again["gate_proj"], again["up_proj"]], axis=2).reshape(n, 2 * half, d)
            else:
                w = again["down_proj"]
            lin = SwitchLinear(w.shape[2], w.shape[1], w.shape[0], bias=False)
            lin.weight = w
            put_back[name] = lin
        experts.gate_up_proj, experts.down_proj = put_back["gate_up_proj"], put_back["down_proj"]

    dest = here / "tiny-alt"
    dest.mkdir(exist_ok=True)
    config = {**CONFIG, "quantization": {"group_size": 64, "bits": 2, "mode": "affine"}}
    (dest / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    mx.eval(out)
    mx.save_safetensors(str(dest / "model.safetensors"), out)
    tokens = mx.array([TOKENS])
    expected = {"tokens": tokens[0].astype(mx.int32), "logits": model(tokens)[0]}
    mx.save_safetensors(str(dest / "expected.safetensors"), expected)
    print("alt", len(out), "tensors", expected["logits"].shape)


def main() -> None:
    here = Path(__file__).parent
    vision(here)
    alt_layout(here)
    for name, dtype in (("f32", mx.float32), ("bf16", mx.bfloat16)):
        model = build(dtype)
        out = here / f"tiny-{name}"
        out.mkdir(exist_ok=True)
        config = {**CONFIG, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
        (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        mx.save_safetensors(
            str(out / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )

        visits = []
        text = model.model.language_model
        for layer in text.layers:
            call = type(layer).__call__

            def traced(self, *a, _call=call, **kw):
                h, blocks = _call(self, *a, **kw)
                visits.append(h.astype(mx.float32))
                return h, blocks

            layer.__class__ = type(type(layer).__name__, (type(layer),), {"__call__": traced})

        tokens = mx.array([TOKENS])
        expected = {
            "tokens": tokens[0].astype(mx.int32),
            "embed": text.embed(tokens)[0].astype(mx.float32),
            "logits": model(tokens)[0],
        }
        count = len(visits)
        for i, h in enumerate(visits):
            expected[f"visit.{i:02d}"] = h[0]

        # A greedy continuation of the first tokens, one token a step.
        cache = model.make_cache()
        step = model(tokens[:, :PROMPT], cache=cache)
        greedy = []
        for _ in range(GREEDY):
            greedy.append(mx.argmax(step[0, -1]).item())
            step = model(mx.array([[greedy[-1]]]), cache=cache)
        expected["greedy"] = mx.array(greedy, dtype=mx.int32)
        mx.save_safetensors(str(out / "expected.safetensors"), expected)
        print(name, count, "visits", expected["logits"].shape)


if __name__ == "__main__":
    main()
