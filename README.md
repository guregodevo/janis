<div align="center">
  <h1>janis</h1>
  <p><strong>Local inference for open-weight models, in Go.</strong><br>An OpenAI-compatible server and a library: MLX on Apple silicon, ggml on Linux and NVIDIA GPUs, pure Go anywhere. One binary, no Python.</p>
  <p>
    <a href="https://github.com/guregodevo/janis/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/tag/guregodevo/janis?sort=semver&label=release"></a>
    <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
    <a href="https://github.com/guregodevo/janis/actions/workflows/release.yml"><img alt="Release build" src="https://github.com/guregodevo/janis/actions/workflows/release.yml/badge.svg"></a>
    <img alt="Go" src="https://img.shields.io/github/go-mod/go-version/guregodevo/janis">
    <img alt="Platforms" src="https://img.shields.io/badge/macOS%20(Apple%20silicon)%20%C2%B7%20Linux%20amd64%20%C2%B7%20arm64-lightgrey">
  </p>
</div>

## Why janis, next to Ollama and llama.cpp

Each row is one machine, one server at a time, 4-bit weights, versions and
commands in [bench/mlx.md](bench/mlx.md) and [docs/backends.md](docs/backends.md):

| | janis | llama.cpp | Ollama |
|---|---:|---:|---:|
| Qwen3-8B, M4 Mac 16 GB: prefill 600 tokens | 224.2 tok/s | 219.5 tok/s | 216.4 tok/s |
| Qwen3-8B, M4 Mac 16 GB: prefill 9,000 tokens | 195.0 tok/s | 174.4 tok/s | 154.5 tok/s |
| Qwen3-8B, M4 Mac 16 GB: decode | 22.8 tok/s | 19.9 tok/s | 20.3 tok/s |
| Qwen3.5-9B, M4 Mac 16 GB: decode | 19.9 tok/s | 18.5 tok/s | not measured |
| Qwen3-8B, GitHub runner, 4 vCPU: prefill / decode | 12 / 5.6 tok/s | not measured | 12.6 / 4.45 tok/s |
| Qwen3-8B, NVIDIA L4: decode | 38 tok/s steady | 45 to 50 tok/s | not measured |

On the Mac, janis runs `mlx-community`'s 4-bit weights (4.3 GB for the 8B)
and llama.cpp and Ollama the Q4_K_M GGUF (5.2 GB). Decode reads every weight
once per token, so part of the decode lead is the smaller file.

What is different:

- **MLX on a Mac.** The Apple-silicon build runs on MLX, Apple's array
  framework, with the `mlx-community` weights as published. No GGUF
  conversion. The same weights run on ggml and pure Go, repacked on load.
- **A Go module.** `import "github.com/guregodevo/janis"` loads a model in
  your process: no server beside your program, no Python.
- **`janis pull` picks the model.** With no name it takes the largest model
  this machine's RAM runs well and its disk can hold. A model that does not
  fit is refused before the download.
- **Typed decisions.** `POST /v1/systemone` answers yes/no, choice and score
  questions with a probability read from one prefill, in tens of
  milliseconds ([Decisions](#decisions-a-local-jev)). No decoding, nothing to
  parse.
- **Installs without root.** One file on Linux, into `~/.local/bin`, sha256
  checked. `janis serve --detach` returns once the model has answered, so a
  CI job can start it and use it on the next line.

Use Ollama or llama.cpp instead for Windows, vision models, architectures
janis does not load ([Models](#models)), several models loaded at once, or
the fastest NVIDIA decode (llama.cpp's 45 to 50 tok/s against 38 on an
L4).

## Install

Linux (amd64, arm64) and macOS (Apple silicon), into `~/.local/bin`, sha256
checked. On a Mac the MLX libraries go to `~/.local/lib/janis`.

```bash
curl -fsSL https://github.com/guregodevo/janis/releases/latest/download/install.sh | sh
janis pull      # the mlx-community model that fits this machine's RAM and disk
janis serve     # http://127.0.0.1:8090/v1
```

```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-8B-4bit", "messages": [{"role": "user", "content": "2+3?"}]}'
```

`janis start` (or `janis serve --detach`) is for a CI job or a script: the
model is pulled when missing, the server runs in the background (its output
appended to `janis-8090.log` in the temp directory, or `--log`), and the
command returns once the model has answered a first request, so the first
client does not pay for loading it. A taken port or a server that dies fails
the command. `JANIS_VERSION=v0.1.2` pins a release. `make dist` builds the
release asset for the machine it runs on ([Build](#build)).

```bash
janis start [--port 8090] [--host 127.0.0.1] [--model NAME] [--log FILE]
janis status [--port 8090]    # pid, model, uptime, GET /health within 2 s; exit 1 if it does not answer
janis stop [--port 8090]      # graceful; SIGKILL after 10 s
janis restart [--port 8090]   # stop, then start on the same flags; --model X replaces the saved one; none running: start
```

Every server, foreground or background, writes its pid and flags to
`~/.janis/run/PORT.json` and removes it when it stops. `stop` and `restart`
act on the server that file names. A server started by janis v0.1.15 or
older has no file: `stop`, `restart` and `status` find its `janis serve` on
the port with `ps` and read its flags, so `janis restart` replaces it on the
same model. `restart` starts the server in the background, one first run
with `janis serve` included. With no janis on the port, `restart` starts one
as `janis start` does.

A server stops gracefully on `janis stop`, SIGTERM, Ctrl-C or its terminal
closing. It takes no new connection, and requests in flight get 5 s to
finish. Those still running are then cancelled: a generation ends at its
next token, a decision before its next question, a request waiting for the
engine leaves the queue. Each is answered 503; a stream cut short ends with
an error event and no `[DONE]`. The model is unloaded last.

## Models

Weights are MLX safetensors from `huggingface.co/mlx-community`, kept in
`~/.cache/huggingface/hub` (`$HF_HOME/hub`, or `HF_HUB_CACHE` when set).
`janis pull` checks the free space first: a model the disk cannot hold is
refused before the download, and with no name it takes the largest RAM tier
that fits the disk. `janis list` shows the installed models and whether each
fits.

| Family | Package |
|---|---|
| Qwen2.5, Qwen3 and Llama, dense; Qwen3 MoE (Qwen3-Coder-30B-A3B) | `qwen` |
| Qwen3.5 hybrid, Gated DeltaNet and attention (0.8B, 9B, Qwen3.8-27B) | `qwen35` |
| Gemma 2 | `gemma` |
| Whisper (speech to text), BGE (embeddings), a reranker | `whisper`, `bge`, `reranker`, from Go only |

## Serve

`janis serve` serves the installed models to any OpenAI-compatible client:
`GET /v1/models` and `POST /v1/chat/completions`, streamed or not, with tools.
The tool schemas are rendered into the system prompt the Hermes way, the
engine's grammar keeps every `<tool_call>` well-formed, and the call comes
back as OpenAI `tool_calls`. One model is loaded at a time. A request naming
another installed model swaps it. `/v1/models` carries each model's
`context_length` from its config. `--host 0.0.0.0` opens it to the network.
Memdoor connects it with `memdoor connect local --base http://localhost:8090/v1`,
no key, on this machine or on a box reached over ssh.

### Decisions (a local Jev)

The same server answers typed questions the way a System One / Jev decision
model does, on the loaded model: `POST /v1/systemone` takes one piece of
state and questions about it, and answers each from a single prefill, the
probability read off the model's next-token distribution over the option
letters. No decoding, no prose to parse, a few tens of milliseconds a
question.

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "func add(a, b int) int { return a + b }",
  "questions": {
    "relevant": {"type": "noul",   "instructions": "Is this about the add function?",
                 "criteria": {"true": "it is", "false": "it is about something else"}},
    "kind":     {"type": "choice", "instructions": "What kind of code is this?",
                 "criteria": {"test": "a test", "impl": "an implementation", "doc": "documentation"}},
    "clarity":  {"type": "score",  "instructions": "How clear is it?",
                 "criteria": ["unclear", "fine", "clear"]}}}'
```

```json
{"model":"Qwen3-8B-4bit","answers":{
  "relevant":{"type":"noul","noul":0.93,"confidence":0.93},
  "kind":{"type":"choice","choice":"impl","probabilities":{"doc":0.02,"impl":0.95,"test":0.03},"confidence":0.95},
  "clarity":{"type":"score","score":1.8,"probabilities":{"0":0.05,"1":0.1,"2":0.85},"confidence":0.85}}}
```

Three question types: `noul` answers P(yes) (its `criteria` may spell out what
true and false mean); `choice` picks one of the labelled `criteria` and gives
the distribution; `score` places the state on an ordered rubric, `score`
being the expected zero-based level. A `model` in the request is honoured
when it names an installed model, otherwise the loaded one answers. The same
call from Go is `engine.Decide(state, questions)` (`decide.go`).

Memdoor uses this as its decision model: `memdoor connect local` on a janis
keeps `local-decisions` next to the chat provider, and from the next gateway
start the turn's effort, the tool routing, the judged reads and the stop
verdict are decided here, with no key. The numbers above are illustrative;
a 0.6B model answers the wire but judges poorly, an 8B is the sensible floor.

Requests run one at a time on the engine. A request whose client has gone
costs no more compute: it leaves the queue if it is waiting, or stops before
its next question. `GET /v1/models` and `GET /health` answer while a request
runs. `engine.DecideContext(ctx, state, questions)` is the Go call that stops
with `ctx`.

## As a library

```go
import (
	llm "github.com/guregodevo/janis"
	"github.com/guregodevo/janis/hostfit"
)

models, _ := hostfit.ListInstalled()          // the snapshots in the Hugging Face cache
e, err := llm.Open(models[0].Path)
if err != nil {
	return err
}
defer e.Close()
reply := e.Chat([]llm.Message{{Role: "user", Content: "2+3?"}}, llm.Options{MaxTokens: 64})
```

`ChatStream` takes a callback per delta. `Options` carries temperature, top-p,
a seed, a cancel hook and the tool-call grammar.

| Package | What it is |
|---|---|
| `serve`, `cmd/janis` | the OpenAI endpoints over the engine (`janis serve`, `start`, `stop`, `restart`, `status`, `list`, `pull`) |
| `janis` (the root package) | load a model, chat, stream, decide, swap the loaded model live |
| `engine` | the tensor interface every backend implements |
| `qwen`, `qwen35`, `gemma` | model families |
| `whisper`, `bge`, `reranker` | speech to text, embeddings, reranking |
| `grammar` | constrained decoding for tool calls |
| `hostfit` | which model fits this host's RAM, and how well it will run |
| `mlxc`, `ggml`, `cpu` | the backends |
| `safetensors`, `gguf`, `tokenizer` | the loaders. `gguf` reads the container and the GGML quantization types and leaves the architecture to the caller. |

## Backends

The model code is written against one tensor interface (`engine.Backend`,
about fifty ops). A backend is an implementation of it.

| Flavor | Backend | Where | How |
|---|---|---|---|
| janis | MLX | Apple Silicon | the default build (`make build`) |
| janis-cuda | ggml, llama.cpp's tensor library: CUDA, Metal or CPU kernels | Linux with an NVIDIA GPU; also Metal and CPU on a Mac | `-tags ggml` (`JANIS_BACKEND=ggml` on a Mac; the default on Linux), `JANIS_GGML_DEVICE=gpu` |
| janis-cpu | pure Go | anywhere | `CGO_ENABLED=0`, or `JANIS_BACKEND=cpu` |

[docs/backends.md](docs/backends.md) has how the ggml backend is built and
how it repacks MLX's 4-bit weights, with the numbers on a Mac, a GitHub
runner and an NVIDIA L4.

## Build

Apple Silicon with MLX:

```bash
brew install mlx mlx-c          # mlx-c is linked from /opt/homebrew
make build                      # provisions lib/libtokenizers.a, then go build ./...
make test
```

Linux, one file (ggml's CPU kernels and the tokenizer linked in, ggml the
backend unless `JANIS_BACKEND=cpu`): the release's `install.sh` ([Install](#install)),
or `scripts/fetch-ggml.sh --static` and `go build -tags ggml ./cmd/janis`.

Anywhere, without MLX or cgo:

```bash
CGO_ENABLED=0 go build ./...
```

`lib/` holds the HuggingFace tokenizers static library per platform
(`lib/libtokenizers.a` on macOS, `lib/linux-<arch>/` on Linux). It is
gitignored; `make tokenizer-lib` fetches or builds it. A program that imports
janis from Go's module cache has no `lib/`: `make install-lib` copies the
archive to `/opt/homebrew/lib` (macOS) or `/usr/local/lib` (Linux), which the
tokenizer also searches — the same place mlx-c comes from.

## Used by

[Memdoor](https://memdoor.ai), the terminal coding agent, behind
`MEMDOOR_SHOW=local`. Memdoor imports janis as a module (its `make janis`
runs `install-lib`); to edit both at once, a `go.work` in the Memdoor
checkout points at `../janis`.

## License

[Apache 2.0](LICENSE).
