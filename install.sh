#!/bin/sh
# janis install: into ~/.local/bin, checked against the release's sha256.
#
#   curl -fsSL https://github.com/guregodevo/janis/releases/latest/download/install.sh | sh
#
# JANIS_VERSION=v0.1.2 pins a release (default: the latest); JANIS_INSTALL_DIR
# changes where the binary goes. Linux (amd64, arm64): one file. macOS (Apple
# silicon): the binary, and the three MLX files it loads into ../lib/janis
# beside it (~/.local/lib/janis). Nothing else is installed or changed.
set -eu

repo=https://github.com/guregodevo/janis
version=${JANIS_VERSION:-latest}
dir=${JANIS_INSTALL_DIR:-$HOME/.local/bin}

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  *) echo "janis runs on Linux and macOS, not $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "no janis build for $(uname -m): amd64 and arm64 only" >&2; exit 1 ;;
esac
if [ "$os" = darwin ]; then
  [ "$arch" = arm64 ] || { echo "janis on a Mac needs Apple silicon (MLX has no Metal on Intel): $repo#build" >&2; exit 1; }
  asset=janis-darwin-arm64.tar.gz
else
  asset=janis-linux-$arch
fi
if [ "$version" = latest ]; then base=$repo/releases/latest/download
else base=$repo/releases/download/$version; fi
if command -v sha256sum >/dev/null 2>&1; then sha256() { sha256sum "$1"; }
else sha256() { shasum -a 256 "$1"; }; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/$asset" "$base/$asset"
curl -fsSL -o "$tmp/$asset.sha256" "$base/$asset.sha256"
want=$(cut -d' ' -f1 < "$tmp/$asset.sha256")
got=$(sha256 "$tmp/$asset" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "sha256 mismatch for $asset: want $want, got $got" >&2; exit 1; }

mkdir -p "$dir"
if [ "$os" = darwin ]; then
  # The binary looks for the libraries beside itself, then in ../lib/janis.
  libdir=$(cd "$dir/.." && pwd)/lib/janis
  mkdir -p "$libdir" "$tmp/x"
  tar -xzf "$tmp/$asset" -C "$tmp/x"
  mv "$tmp/x/libmlx.dylib" "$tmp/x/libmlxc.dylib" "$tmp/x/mlx.metallib" "$libdir/"
  mv "$tmp/x/janis" "$dir/janis"
  placed="$dir/janis (MLX in $libdir)"
else
  chmod +x "$tmp/$asset"
  mv "$tmp/$asset" "$dir/janis"
  placed="$dir/janis"
fi
echo "✓ janis ($version, $os/$arch) → $placed · sha256 verified"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "  $dir is not on PATH: export PATH=\"$dir:\$PATH\"" ;;
esac
echo "  janis serve --detach --model Qwen3-8B-4bit   pulls it, serves it on 127.0.0.1:8090/v1, returns once it answers"
