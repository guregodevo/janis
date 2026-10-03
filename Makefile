.PHONY: all tokenizer-lib build test vet help

# janis — local inference for open-weight models on Apple Silicon (MLX),
# with a pure-Go CPU fallback. The MLX backend links mlx-c from Homebrew and
# the HuggingFace tokenizers C library (lib/, gitignored) via cgo.

all: build

# tokenizer-lib provisions the arch-matched libtokenizers.a under lib/ that
# tokenizer/tokenizer.go links. Gitignored on every platform: on a fresh
# checkout the script fetches the prebuilt release or builds it with cargo.
# Idempotent — a no-op when the lib is already there.
tokenizer-lib:
	@bash scripts/build-tokenizer.sh

build: tokenizer-lib
	@go build ./...
	@echo "✓ janis built"

vet: tokenizer-lib
	@go vet ./...

test: tokenizer-lib
	@go test ./...
	@echo "✓ Tests passed"

help:
	@echo "make tokenizer-lib  provision lib/libtokenizers.a for this platform"
	@echo "make build          build every package (MLX on Apple Silicon, CPU elsewhere)"
	@echo "make test           run the tests"
	@echo "CGO_ENABLED=0 go build ./...   the cgo-free build (no MLX, no tokenizer lib)"
