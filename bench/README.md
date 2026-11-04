# CPU prefill against llama.cpp

The GPU (MLX) numbers are in [mlx.md](mlx.md).

How fast janis reads a long prompt on the CPU (ggml backend), next to
llama.cpp at the ggml version janis links (b8640).

| | |
|---|---|
| Model | Qwen3-1.7B, 4-bit: `mlx-community/Qwen3-1.7B-4bit` for janis, `unsloth/Qwen3-1.7B-GGUF` Q4_K_M for llama.cpp |
| Prompt | 8,404 tokens, one token out |
| janis | `JANIS_BACKEND=ggml`, `MLX_PREFILL_CHUNK=512`, timed by `bench/prefill.py` over HTTP after a short warm-up request |
| llama.cpp | `llama-bench -p 8404 -n 0 -r 1`, CPU only |
| Measured | 2026-11-04 |

## Google Cloud Shell: 4 vCPU Xeon @ 2.20 GHz, AVX2, no AVX-512

Cloud Shell recycled its VM between runs, so the runs are in two groups,
each on one VM of this type.

| VM | | flash attention | prompt tok/s |
|---|---|---|---|
| A | llama.cpp b8640 `-t 4` | off | 20.55 |
| A | llama.cpp b8640 `-t 4` | on | 20.98 |
| A | janis v0.1.12 | off | 16.5 |
| B | llama.cpp b8640 `-t 4` | on | 21.23 |
| B | janis, this change | on (x86 default) | 23.8 |
| B | janis, this change | off | 21.4 |

On VM B janis reads the prompt 12% faster than llama.cpp's faster setting.

## Apple M4, 10 cores (4 performance, 6 efficiency), CPU only

llama.cpp built from b8640 with `-DGGML_METAL=OFF -DGGML_BLAS=OFF
-DGGML_NATIVE=OFF`, the flags janis builds ggml with. The release build sends
large batches to the GPU and to Accelerate even with `-ngl 0`.

| | flash attention | threads | prompt tok/s |
|---|---|---|---|
| llama.cpp b8640 | off | 10 | 208.3 ± 0.9 |
| llama.cpp b8640 | on | 10 | 174.8 (`-p 8192`) |
| janis v0.1.12 | off | 10 | 152.2 (256-token chunks) |
| janis, this change | off (ARM default) | 10 | 201.0, 205.1 |

## What changed

- Attention reads the KV cache padded to 32 tokens, the padding zeroed and
  masked. ggml's f16 GEMM (tinyBLAS) takes only a key count that is a
  multiple of 8; an exact count fell back to `ggml_vec_dot_f16`, one row at
  a time, 40% of a prefill in a `sample` profile on the M4.
- Without flash attention the value cache is kept transposed, tokens
  innermost, as llama.cpp keeps it. v·scores reads it in place.
- Flash attention is on for x86 and off for ARM. `JANIS_GGML_FLASH=1` or `0`
  overrides.
- The KV cache is reserved once per request, for the prompt plus
  `max_tokens`, then grows by half. A fixed 256-token step copied the whole
  cache every other chunk.

## Reproduce

janis, from a checkout with the ggml libraries (`scripts/fetch-ggml.sh`):

```
go build -tags ggml -o janis ./cmd/janis
JANIS_BACKEND=ggml MLX_PREFILL_CHUNK=512 ./janis serve --port 18099 --model Qwen3-1.7B-4bit &
curl -s localhost:18099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-1.7B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null
python3 bench/prefill.py 18099
```

`JANIS_BACKEND=ggml` is needed on a Mac only, where MLX is the default.
llama.cpp:

```
llama-bench -m Qwen3-1.7B-Q4_K_M.gguf -t 4 -fa 0,1 -p 8404 -n 0 -r 1
```
