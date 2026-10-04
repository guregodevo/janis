# janis

Local inference for open-weight models — an Ollama for MLX. One Go module,
no daemon, no Python: an MLX backend on Apple Silicon, a pure-Go CPU backend
elsewhere, and what it takes to run a model on the machine in front of you.

- **engine** — load a model, chat, stream, swap the loaded model live.
- **qwen, qwen35, gemma** — model families (Qwen3, Qwen3.5 hybrid, Gemma).
- **whisper, bge, reranker** — speech-to-text, embeddings, reranking.
- **grammar** — constrained decoding for tool calls.
- **hostfit** — which model fits this host's RAM, and how well it will run.
- **mlxc, cpu, safetensors, gguf, tokenizer** — the backends and the loaders.
  `gguf` is a container reader only (format, not architecture): it reads the
  GGML quantization types and leaves model-specific meaning to the caller.

Weights come from `huggingface.co/mlx-community` into `~/.cache/huggingface/`.

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
