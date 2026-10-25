#!/usr/bin/env bash
# scripts/fetch-ggml.sh — the prebuilt ggml libraries janis's ggml backend
# links, from llama.cpp's release of the pinned tag, plus the matching
# headers from the source tarball. Into ggml/third_party/<platform>/
# (gitignored); idempotent. CUDA on Linux comes from the same release line
# (the win-cuda and ubuntu zips carry libggml-cuda).
#
# --cuda (Linux with an NVIDIA GPU): llama.cpp publishes no CUDA build for
# Linux, so the ggml libraries are built here from the same tag with
# cmake and the CUDA toolkit on the machine (a Deep Learning VM has both),
# into ggml/third_party/ubuntu-x64/lib next to the CPU ones; the backend
# loads libggml-cuda.so at run time (JANIS_GGML_DEVICE=gpu).
set -euo pipefail
TAG=${GGML_TAG:-b8640}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = "--cuda" ]; then
	command -v cmake >/dev/null || { echo "cmake is needed to build ggml with CUDA" >&2; exit 1; }
	command -v nvcc >/dev/null || { echo "nvcc (the CUDA toolkit) is needed" >&2; exit 1; }
	DEST="$ROOT/ggml/third_party/ubuntu-$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/')"
	mkdir -p "$DEST/lib" "$DEST/include"
	tmp=$(mktemp -d)
	curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz" | tar -xz -C "$tmp"
	src=$(echo "$tmp"/llama.cpp-*)
	cmake -S "$src/ggml" -B "$tmp/build" -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
	cmake --build "$tmp/build" --config Release -j"$(nproc)" 2>&1 | tail -3
	find "$tmp/build" -name 'libggml*.so*' -exec cp -a {} "$DEST/lib/" \;
	cp "$src"/ggml/include/*.h "$DEST/include/"
	rm -rf "$tmp"
	echo "$TAG" > "$DEST/.tag"
	echo "✓ ggml $TAG with CUDA → $DEST ($(ls "$DEST/lib" | wc -l | tr -d ' ') libraries)"
	exit 0
fi
case "$(uname -s)-$(uname -m)" in
	Darwin-arm64) PLATFORM=macos-arm64 ;;
	Linux-x86_64) PLATFORM=ubuntu-x64 ;;
	Linux-aarch64) PLATFORM=ubuntu-arm64 ;;
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
