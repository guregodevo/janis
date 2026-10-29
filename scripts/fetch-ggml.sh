#!/usr/bin/env bash
# scripts/fetch-ggml.sh — the ggml libraries janis's ggml backend links, at
# llama.cpp's pinned tag: on a Mac the prebuilt ones from its release plus the
# matching headers from the source tarball. Into ggml/third_party/<platform>/
# (gitignored); idempotent.
#
# Linux builds from source: llama.cpp's Linux release carries the CPU
# kernels only as per-microarchitecture variants loaded at run time
# (libggml-cpu-haswell.so, -zen4.so, ...), never the libggml-cpu.so the
# backend links, so the ggml libraries are built here from the same tag with
# cmake, for AVX2 (any x86-64 of the last ten years, a GitHub runner
# included; not -march=native, so a cached build runs on the next machine).
# --cuda (Linux with an NVIDIA GPU) adds the CUDA kernels with the toolkit on
# the machine (a Deep Learning VM has both); the backend loads
# libggml-cuda.so at run time (JANIS_GGML_DEVICE=gpu).
set -euo pipefail
TAG=${GGML_TAG:-b8640}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$(uname -s)" = Linux ]; then
	cuda=; [ "${1:-}" = "--cuda" ] && cuda=1
	command -v cmake >/dev/null || { echo "cmake is needed to build ggml on Linux" >&2; exit 1; }
	[ -z "$cuda" ] || command -v nvcc >/dev/null || { echo "nvcc (the CUDA toolkit) is needed" >&2; exit 1; }
	DEST="$ROOT/ggml/third_party/ubuntu-$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/')"
	flavor=${cuda:+cuda}; flavor=${flavor:-cpu}
	if [ -f "$DEST/.tag" ] && [ "$(cat "$DEST/.tag")" = "$TAG-$flavor" ]; then
		exit 0
	fi
	mkdir -p "$DEST/lib" "$DEST/include"
	# The source and the build live under the repo (gitignored), not /tmp: a
	# machine stopped mid-build resumes where it was instead of starting over.
	tmp="$ROOT/ggml/third_party/build-$flavor-$TAG"
	mkdir -p "$tmp"
	[ -d "$tmp"/llama.cpp-* ] || curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz" | tar -xz -C "$tmp"
	src=$(echo "$tmp"/llama.cpp-*)
	# ggml's own CMake does not configure standalone inside the llama.cpp
	# tree (no ggml.pc.in there); the root configures with everything but
	# ggml switched off, and only the ggml targets are built.
	# One CUDA architecture, the GPU in the machine as nvidia-smi reports it
	# (GGML_CUDA_ARCH names others, e.g. "75;86;89"; cmake 3.22 has no
	# "native"): every fattn template instance otherwise compiles once per
	# architecture, an hour on four cores.
	gpu=(); targets=(ggml ggml-base ggml-cpu)
	if [ -n "$cuda" ]; then
		gpu=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${GGML_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}")
		targets+=(ggml-cuda)
	fi
	cmake -S "$src" -B "$tmp/build" "${gpu[@]}" -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
		-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF >/dev/null
	cmake --build "$tmp/build" --config Release -j"$(nproc)" --target "${targets[@]}" 2>&1 | tail -3
	find "$tmp/build" -name 'libggml*.so*' -exec cp -a {} "$DEST/lib/" \;
	cp "$src"/ggml/include/*.h "$DEST/include/"
	echo "$TAG-$flavor" > "$DEST/.tag"
	echo "✓ ggml $TAG ($flavor) → $DEST ($(ls "$DEST/lib" | wc -l | tr -d ' ') libraries)"
	exit 0
fi
case "$(uname -s)-$(uname -m)" in
	Darwin-arm64) PLATFORM=macos-arm64 ;;
	*) echo "no prebuilt ggml for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
DEST="$ROOT/ggml/third_party/$PLATFORM"
if [ -f "$DEST/.tag" ] && [ "$(cat "$DEST/.tag")" = "$TAG" ]; then
	exit 0
fi
rm -rf "$DEST" && mkdir -p "$DEST/lib" "$DEST/include"
tmp=$(mktemp -d)
curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-bin-$PLATFORM.tar.gz" | tar -xz -C "$tmp"
find "$tmp" \( -name 'libggml*.dylib' -o -name 'libggml*.so*' \) -exec cp -a {} "$DEST/lib/" \;
mkdir -p "$tmp/src"
curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz" | tar -xz -C "$tmp/src"
cp "$tmp"/src/llama.cpp-*/ggml/include/*.h "$DEST/include/"
rm -rf "$tmp"
echo "$TAG" > "$DEST/.tag"
echo "✓ ggml $TAG → $DEST ($(ls "$DEST/lib" | wc -l | tr -d ' ') libraries)"
