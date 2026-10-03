#!/usr/bin/env bash
#
# Ensure the platform-appropriate libtokenizers.a exists before `go build`.
#
# tokenizer/tokenizer.go links the HuggingFace tokenizers C library via cgo.
# Its LDFLAGS resolve the archive per platform:
#   darwin        -> lib/libtokenizers.a             (gitignored)
#   linux/amd64   -> lib/linux-amd64/libtokenizers.a  (gitignored)
#   linux/arm64   -> lib/linux-arm64/libtokenizers.a  (gitignored)
#
# All three archives are gitignored (the 39 MB darwin one used to be committed;
# untracked 2026-08-28 to keep it out of every clone). A fresh checkout has no
# lib and the build fails at link with `cannot find -ltokenizers`. This script
# provisions it: try the prebuilt release first, and fall back to building from
# source with cargo — the fallback that works in networks where the GitHub
# release asset is blocked but crates.io is reachable.
#
# Idempotent: exits immediately when the archive is already present.
set -euo pipefail

TOK_VERSION="v1.27.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

os="$(uname -s)"
arch="$(uname -m)"
case "$arch" in
	x86_64 | amd64) arch="amd64" ;;
	arm64 | aarch64) arch="arm64" ;;
	*) echo "✗ unsupported arch: $arch" >&2; exit 1 ;;
esac

case "$os" in
	Darwin) dest_dir="$REPO_ROOT/lib" ;;
	Linux) dest_dir="$REPO_ROOT/lib/linux-$arch" ;;
	*) echo "✗ unsupported OS: $os" >&2; exit 1 ;;
esac
dest="$dest_dir/libtokenizers.a"
mkdir -p "$dest_dir"

if [ -s "$dest" ]; then
	echo "✓ libtokenizers.a present ($dest)"
	exit 0
fi

os_lower="$(echo "$os" | tr '[:upper:]' '[:lower:]')"

# 1) Fast path: the prebuilt release archive.
url="https://github.com/daulet/tokenizers/releases/download/$TOK_VERSION/libtokenizers.$os_lower-$arch.tar.gz"
echo "→ fetching prebuilt libtokenizers ($os_lower-$arch)..."
if curl -fsSL --max-time 120 "$url" | tar -xz -C "$dest_dir" 2>/dev/null && [ -s "$dest" ]; then
	echo "✓ fetched prebuilt libtokenizers.a → $dest"
	exit 0
fi
echo "  prebuilt unavailable — building from source with cargo"

# 2) Fallback: build the static lib from the module-cache source with cargo.
if ! command -v cargo >/dev/null 2>&1; then
	echo "✗ cargo not found: install Rust (https://rustup.rs) to build libtokenizers from source" >&2
	exit 1
fi
mod_dir="$(cd "$REPO_ROOT" && go list -m -f '{{.Dir}}' github.com/daulet/tokenizers 2>/dev/null || true)"
if [ -z "$mod_dir" ] || [ ! -d "$mod_dir" ]; then
	echo "✗ could not locate the github.com/daulet/tokenizers module (run 'go mod download' first)" >&2
	exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -R "$mod_dir"/. "$work"/
chmod -R u+w "$work"
# Persist the cargo target dir so a re-run after a partial build is incremental.
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/memdoor-tokenizers-target"
echo "→ cargo build --release -p tokenizers-ffi (first build takes ~1 min)"
(cd "$work" && cargo build --release -p tokenizers-ffi)
cp "$CARGO_TARGET_DIR/release/libtokenizers_ffi.a" "$dest"
echo "✓ built libtokenizers.a from source → $dest"
