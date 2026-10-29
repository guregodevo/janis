#!/bin/sh
# janis install: one file into ~/.local/bin, checked against the release's sha256.
#
#   curl -fsSL https://github.com/guregodevo/janis/releases/latest/download/install.sh | sh
#
# JANIS_VERSION=v0.1.2 pins a release (default: the latest); JANIS_INSTALL_DIR
# changes where it goes. Linux (amd64, arm64); on a Mac janis builds from source
# with MLX (README, "Build"). Nothing else is installed or changed.
set -eu

repo=https://github.com/guregodevo/janis
version=${JANIS_VERSION:-latest}
dir=${JANIS_INSTALL_DIR:-$HOME/.local/bin}

case "$(uname -s)" in
  Linux) ;;
  Darwin) echo "janis on a Mac builds from source, with MLX: $repo#build" >&2; exit 1 ;;
  *) echo "janis runs on Linux and macOS, not $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "no janis build for $(uname -m): amd64 and arm64 only" >&2; exit 1 ;;
esac
asset=janis-linux-$arch
if [ "$version" = latest ]; then base=$repo/releases/latest/download
else base=$repo/releases/download/$version; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/$asset" "$base/$asset"
curl -fsSL -o "$tmp/$asset.sha256" "$base/$asset.sha256"
want=$(cut -d' ' -f1 < "$tmp/$asset.sha256")
got=$(sha256sum "$tmp/$asset" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "sha256 mismatch for $asset: want $want, got $got" >&2; exit 1; }

mkdir -p "$dir"
chmod +x "$tmp/$asset"
mv "$tmp/$asset" "$dir/janis"
echo "✓ janis ($version, $arch) → $dir/janis · sha256 verified"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "  $dir is not on PATH: export PATH=\"$dir:\$PATH\"" ;;
esac
echo "  janis serve --detach --model Qwen3-8B-4bit   pulls it, serves it on 127.0.0.1:8090/v1, returns once it answers"
