# MLX against llama.cpp on Metal

How fast janis reads a prompt and writes tokens on an Apple GPU, next to
llama.cpp on Metal and mlx-lm.

| | |
|---|---|
| Machine | Apple M4, 10-core GPU, 16 GB, macOS 26.5 |
| Model | Qwen3.5-9B, 4-bit: `mlx-community/Qwen3.5-9B-MLX-4bit` for janis and mlx-lm, `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (5.28 GiB) for llama.cpp |
| janis | release tarballs (MLX 0.32.4), `janis serve --port 18099 --model Qwen3.5-9B-MLX-4bit`, defaults (256-token prefill chunks) |
| llama.cpp | b12100 macOS release, `llama-server -ngl 99 -c 16384` |
| mlx-lm | 0.32.2 on MLX 0.32.4, `python -m mlx_lm generate` |
| Client | `bench/mlx.py`: three runs each, the median. Every prompt opens with a fresh random string, so no prompt cache skips work. Prefill is `max_tokens: 1`, reported `prompt_tokens` over wall time. Decode streams 128 tokens and times the first to the last. |
| Measured | 2026-11-04, one server at a time |

## Results

| | prefill ~600 tok | prefill ~9,000 tok | decode |
|---|---:|---:|---:|
| janis v0.1.14 | 173.2 tok/s | 166.5 tok/s | 13.2 tok/s |
| janis, this change | 208.8 tok/s | 202.5 tok/s | 19.9 tok/s |
| llama-server b12100 | 209.2 tok/s | 194.6 tok/s | 18.5 tok/s |
| llama-bench b12100 (`-p 512,8192 -n 128`, fa 0) | 215.8 tok/s | 194.6 tok/s | 18.7 tok/s |
| mlx-lm (its own timing, 8,974 tokens) | — | 225.6 tok/s | 22.0 tok/s |

## Decode after a long prompt, and at the server's defaults

`bench/mlx_decode.py`: 128 tokens streamed, greedy after a ~9,000-token
prompt, then the server's default sampling (temperature 0.6, top-p 0.95,
repeat penalty; no `temperature` sent) after a short and a ~9,000-token
prompt. Release tarballs and release-workflow builds of the branches, MLX
0.32.4.

| | greedy after 9k | default sampling, short | default sampling, after 9k |
|---|---:|---:|---:|
| janis v0.1.14 | 8.5 tok/s | 9.5 tok/s | 6.8 tok/s |
| + bf16 stream, intermediates freed (#13) | 15.5 tok/s | 14.8 tok/s | 12.9 tok/s |
| + KV written in place, top-p without a full sort | 17.4–17.9 tok/s | 16.9 tok/s | 16.6 tok/s |
| mlx-lm 0.32.2 (its own timing, 8,990 tokens) | 19.7 tok/s | — | — |

llama.cpp was not run after a long prompt.

Peak memory on the 9B (janis's footprint over a 9,000-token prompt, local
builds on Homebrew MLX 0.31.1): 10 GB before, 7.4 GB after. The weights
take 5.3 GB.

## What changed

- The Gated DeltaNet layer's output goes back to the activation dtype before
  `out_proj`, as mlx-lm does. Left in f32, it made the residual stream f32
  from the first layer on, so every later matmul, norm and KV write moved
  twice the bytes.
- A prefill chunk's or a decode step's intermediates are swept before its
  eval, not after. The Go-side handles had kept every layer's activations
  alive through the pass; MLX now frees each buffer once its last consumer
  has run.
- On MLX, keys and values are written into a KV buffer with spare room
  (`slice_update`, in place when nothing else holds the buffer), as mlx-lm's
  KVCache does. A concatenation per step copied every attention layer's whole
  cache every token.
- Top-p ranks the best candidates with a bounded heap instead of sorting the
  vocabulary: 18 ms a token at 248k, 1.9 ms after (`BenchmarkSampleToken`).
- Each prefill chunk's logits are unpinned when the next chunk starts. They
  had stayed pinned for the life of the server, about 1 MB a chunk on every
  request.

## Reproduce

```
janis serve --port 18099 --model Qwen3.5-9B-MLX-4bit &
python3 bench/mlx.py 18099 Qwen3.5-9B-MLX-4bit
python3 bench/mlx_decode.py 18099 Qwen3.5-9B-MLX-4bit
```

llama.cpp, from the macOS release:

```
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 16384 --port 18098 &
python3 bench/mlx.py 18098 any
llama-bench -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -p 512,8192 -n 128 -fa 0,1 -r 3
```

## Qwen3-8B: janis, llama.cpp and Ollama

The same client (`bench/mlx.py`), one server at a time, 2026-11-05, on the
M4 above. Ollama has no Qwen3.5 model in its library here, so this is
Qwen3-8B.

| | janis v0.1.15 | llama-server b12100 | Ollama 0.20.0 |
|---|---:|---:|---:|
| Weights | `mlx-community/Qwen3-8B-4bit`, 4.3 GB | Ollama's `qwen3:8b` GGUF, Q4_K_M, 5.2 GB | `qwen3:8b-32k` (the same GGUF, `num_ctx` 32768) |
| Server | `janis serve --port 18099 --model Qwen3-8B-4bit` | `llama-server -m <the GGUF> -ngl 99 -c 16384` | `ollama serve`, defaults |
| prefill ~600 tokens | 224.2 tok/s | 219.5 tok/s | 216.4 tok/s |
| prefill ~9,000 tokens | 195.0 tok/s | 174.4 tok/s | 154.5 tok/s |
| decode, 128 tokens | 22.8 tok/s | 19.9 tok/s | 20.3 tok/s |

Decode reads every weight once per token. The MLX 4-bit file is 4.3 GB and
the Q4_K_M 5.2 GB, so part of janis's decode lead is the smaller file.
