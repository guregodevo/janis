.PHONY: all tokenizer-lib install-lib build test vet dist dist-check help

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

# dist builds the release asset for THIS machine's OS and arch, into dist/:
#   Linux   dist/janis-linux-<arch>          one static file: ggml's CPU kernels
#                                            and the tokenizer library linked in
#   macOS   dist/janis-darwin-arm64.tar.gz   the binary with MLX, plus the three
#                                            files it loads from Homebrew's mlx
#                                            and mlx-c (scripts/bundle-mlx.sh)
# Each asset gets its .sha256 beside it, the file install.sh checks. Neither OS
# can build the other's asset (both link native libraries), so the release
# workflow runs this once per runner. Linux: a cmake toolchain for ggml.
# macOS: brew install mlx mlx-c.
GOOS   := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
DIST   := dist
SHA256 := $(shell command -v sha256sum 2>/dev/null || echo "shasum -a 256")
ifeq ($(GOOS),darwin)
ASSET := janis-darwin-$(GOARCH).tar.gz
else
ASSET := janis-$(GOOS)-$(GOARCH)
endif

dist: tokenizer-lib
	@mkdir -p $(DIST)
ifeq ($(GOOS),darwin)
	@bash scripts/bundle-mlx.sh $(DIST)/$(ASSET)
else
	@bash scripts/fetch-ggml.sh --static
	@go build -tags ggml -trimpath -ldflags "-s -w" -o $(DIST)/$(ASSET) ./cmd/janis
	@ldd $(DIST)/$(ASSET) || true
endif
	@cd $(DIST) && $(SHA256) $(ASSET) > $(ASSET).sha256
	@ls -la $(DIST)/$(ASSET) $(DIST)/$(ASSET).sha256

# dist-check runs the asset as a user gets it, from an empty directory: the
# archive extracted on macOS, the file copied on Linux; then one answer from
# Qwen3-0.6B-4bit (scripts/smoke.sh). PORT= picks another port than 8090.
PORT ?= 8090
dist-check:
	@d=$$(mktemp -d) && \
	 if [ "$(GOOS)" = darwin ]; then tar -xzf $(DIST)/$(ASSET) -C $$d; else cp $(DIST)/$(ASSET) $$d/janis; fi && \
	 bash scripts/smoke.sh $$d/janis $(PORT)

help:
	@echo "make tokenizer-lib  provision lib/libtokenizers.a for this platform"
	@echo "make install-lib    copy it to the system lib dir (for programs importing janis)"
	@echo "make build          build every package (MLX on Apple Silicon, CPU elsewhere)"
	@echo "make test           run the tests"
	@echo "make dist           the release asset for this OS and arch, into dist/ (with its .sha256)"
	@echo "make dist-check     run that asset from an empty directory: one answer from Qwen3-0.6B-4bit"
	@echo "CGO_ENABLED=0 go build ./...   the cgo-free build (no MLX, no tokenizer lib)"
