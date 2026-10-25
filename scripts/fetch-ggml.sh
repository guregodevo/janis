#!/usr/bin/env bash
# scripts/fetch-ggml.sh — the prebuilt ggml libraries janis's ggml backend
# links, from llama.cpp's release of the pinned tag, plus the matching
# headers from the source tarball. Into ggml/third_party/<platform>/
# (gitignored); idempotent. CUDA on Linux comes from the same release line
# (the win-cuda and ubuntu zips carry libggml-cuda).
set -euo pipefail
TAG=${GGML_TAG:-b8640}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
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
