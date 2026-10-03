.PHONY: all tokenizer-lib install-lib build test vet help

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

# install-lib copies the lib into the system library dir the tokenizer also
# searches (/opt/homebrew/lib on macOS, /usr/local/lib on Linux), so a
# program that imports janis from Go's module cache links without a checkout.
LIBDIR := $(if $(filter Darwin,$(shell uname -s)),/opt/homebrew/lib,/usr/local/lib)
install-lib: tokenizer-lib
	@src=$$( [ "$$(uname -s)" = Darwin ] && echo lib/libtokenizers.a || echo lib/linux-$$(dpkg --print-architecture 2>/dev/null || uname -m)/libtokenizers.a ); \
	 cmp -s $$src $(LIBDIR)/libtokenizers.a 2>/dev/null || { cp $$src $(LIBDIR)/libtokenizers.a && echo "✓ libtokenizers.a → $(LIBDIR)"; }

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
	@echo "make install-lib    copy it to the system lib dir (for programs importing janis)"
	@echo "make build          build every package (MLX on Apple Silicon, CPU elsewhere)"
	@echo "make test           run the tests"
	@echo "CGO_ENABLED=0 go build ./...   the cgo-free build (no MLX, no tokenizer lib)"
