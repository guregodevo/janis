# Backends

The model code is written against one tensor interface (`engine.Backend`,
about fifty ops); a backend is an implementation of it. Three exist:

| Flavor | Backend | Where | How |
|---|---|---|---|
| janis | MLX | Apple Silicon | the default build (`make build`) |
| janis-cuda | ggml, llama.cpp's tensor library: CUDA, Metal or CPU kernels | Linux with an NVIDIA GPU; also Metal and CPU on a Mac | `-tags ggml` (`JANIS_BACKEND=ggml` on a Mac; the default on Linux), `JANIS_GGML_DEVICE=gpu` |
| janis-cpu | pure Go | anywhere | `CGO_ENABLED=0`, or `JANIS_BACKEND=cpu` |

The ggml backend (package `ggml`, 2026-10-25) links ggml at llama.cpp's
pinned tag (`scripts/fetch-ggml.sh`: the prebuilt libraries on a Mac; on
Linux built with cmake, about a minute for the CPU, `--cuda` about 20 minutes
once for the GPU present, `--static` for the one-file release). Ops are
recorded and flushed as one graph per step, tensors stay on the device, and
the engine's pin/sweep protocol frees each step's temporaries. MLX's 4-bit
weights are repacked once: on a GPU into ggml's Q4_1 (the same affine
formula); on the CPU into Q4_K in ggml's repack buffer, the tiled int8 matmul
llama.cpp runs a Q4_K_M model with, read straight off the MLX codes (a
sub-block of 32 holds sixteen values at most, so each code maps to its new one
through a table; 2026-10-29). Every op matches the pure-Go backend to float
precision; the quantized matmul within llama.cpp's own 8-bit-activation
precision.

Measured, Qwen3 4-bit, 160 to 512 tokens generated:

| Machine | Model | MLX | ggml Metal | ggml CUDA | ggml CPU | pure Go |
|---|---|---|---|---|---|---|
| Mac M4 | 0.6B | 169 tok/s | 69 | | 66 (4 threads) | 17 |
| Mac M4 | 8B | 18.8 | 13.0 | | 14 (4 threads, Q4_K; 5.6 on Q4_1); prefill 56 tok/s | |
| GitHub runner `ubuntu-latest`, 4 vCPU (AMD EPYC 7763) | 8B | | | | 5.6, prefill 12 tok/s (Ollama on the same CPU: 4.45 and 12.6) | |
| Google g2-standard-4, 1× L4 | 0.6B | | | 52 to 56 (about 100 steady) | 9.4 | |
| Google g2-standard-4, 1× L4 | 8B | | | 10 to 12 (about 38 steady) | 1.1 | |
| Google g2-standard-4, 1× L4 | Qwen3-Coder-30B-A3B (MoE, 128 experts, 8 active) | | | 11 steady, after a 2-minute load | | |
| Google g2-standard-16, 1× L4 | Qwen3.8-27B (the Qwen3.5 hybrid: Gated DeltaNet + attention) | | | 5.3 decode; prefill 250 to 265 tok/s, a 9.8K-token prompt in 37 s at 21.9 GB peak | | |
| vast.ai, 1× H100 SXM 80 GB | Qwen3.8-27B | | | 55.5 to 56.4 decode with recorded steps (`JANIS_GGML_REPLAY=1`), 10.7 to 11.5 without (256 tokens after a 30-token prompt) | | |

The two CUDA figures per row are the whole run, which includes the one-time
Q4_1 repack of every weight on the host (about 19 s for the 8B on four
vCPUs), and the steady rate the per-step compute timings give once that is
done. llama.cpp runs the same 8B at 45 to 50 tok/s on an L4.

MoE runs resident (`JANIS_MOE_RESIDENT=1`): every expert of a layer is
stacked once on the device and the gather picks them by index
(`ggml_mul_mat_id`), the shape of MoE on a GPU with room for the whole
model; the 30B-A3B takes 18 GB of the L4. Its 11 tok/s is overhead, not
kernels: the router's top-k runs on the host, one sync per layer per token
(the GPU sits at 13%), and the 16 GB repack on load is what the two minutes
are. Both are the next changes, with a cached repack. The Qwen3.5 hybrid's
two ops are ggml's own (`ggml_gated_delta_net`, `ggml_ssm_conv`, one node
each, a CUDA kernel for head sizes 16 to 128), so the 27B runs on CUDA since
2026-10-26. A flush lays its batch out with ggml's graph allocator: results
the model dropped share memory in the order they die (the pending list holds
them weakly, a batch over 64 MB asks the collector first), results it still
holds get their own; a 32-token prefill chunk of the 27B that held 2.2 GB of
temporaries holds a few tens of MB, and the prefill chunk can be 64 on a
24 GB card (`MLX_PREFILL_CHUNK`).

With `JANIS_GGML_REPLAY=1` a decode step on a GPU is recorded once and
run again for the tokens after it (`ggml/replay.go`; off by default). ggml
replays a graph as a CUDA graph only when its nodes, addresses, shapes and
parameters are the last run's, so what moves from token to token is an
input the step writes before each run: the token's embedding row (read
from the weight's rows, no readback of the graph), the RoPE position, the
cache rows a token is written to (`set_rows`, the transposed value cache
element by element), and a mask over the caches read padded to a bucket
of 256 tokens; a Gated DeltaNet layer's conv tail and state are written
in place. On the H100 the 27B's step was 34 ms of GPU work and about
32 ms of recording, building and laying out a graph of 5,464 nodes; the
same graph run again is 14.9 ms. A step reads the caches padded, so its
sums are not in the order of an exact read: the logits part by rounding,
about a logit, and the greedy replies after a few dozen tokens. With
`JANIS_GGML_PADDECODE=1` a plain step reads them padded too, and the two
give the same logits bit for bit (`TestReplayedStepsLive`).

