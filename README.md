# janis

Local inference for open-weight models — an Ollama for MLX. One Go module,
no Python: an MLX backend on Apple Silicon, a pure-Go CPU backend elsewhere,
and what it takes to run a model on the machine in front of you. Use it as a
library, or run `janis serve` and talk to it like llama.cpp or Ollama from any
OpenAI-compatible client.

- **serve, cmd/janis** — the OpenAI endpoints over the engine (`janis serve`,
  `janis list`): models, chat completions streamed or not, tool calls.
- **engine** — load a model, chat, stream, swap the loaded model live.
- **qwen, qwen35, gemma** — model families (Qwen3, Qwen3.5 hybrid, Gemma).
- **whisper, bge, reranker** — speech-to-text, embeddings, reranking.
- **grammar** — constrained decoding for tool calls.
- **hostfit** — which model fits this host's RAM, and how well it will run.
- **mlxc, cpu, safetensors, gguf, tokenizer** — the backends and the loaders.
  `gguf` is a container reader only (format, not architecture): it reads the
  GGML quantization types and leaves model-specific meaning to the caller.

Weights come from `huggingface.co/mlx-community` into `~/.cache/huggingface/`.

## Serve

`janis serve` runs the installed models the way llama.cpp or Ollama does,
for any OpenAI-compatible client, on `http://127.0.0.1:8090/v1`:

```bash
go build -o janis ./cmd/janis
./janis list                         # the installed mlx-community models
./janis serve --model Qwen3-8B-4bit  # GET /v1/models, POST /v1/chat/completions
```

Streamed or not, with tools: the schemas are rendered into the system prompt
the Hermes way, the engine's grammar keeps every `<tool_call>` well-formed,
and the call comes back as OpenAI `tool_calls`. One model is loaded at a
time; a request naming another installed model swaps it. `/v1/models`
carries each model's `context_length` from its config. `--host 0.0.0.0`
opens it to the network. Memdoor connects it with
`memdoor connect local --base http://localhost:8090/v1`, no key, on this
machine or on a box reached over ssh.

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

## Build

Apple Silicon with MLX:

```bash
brew install mlx mlx-c          # mlx-c is linked from /opt/homebrew
make build                      # provisions lib/libtokenizers.a, then go build ./...
make test
```

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
