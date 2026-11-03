#!/usr/bin/env bash
# scripts/bundle-mlx.sh <out.tar.gz> — janis for a Mac as one archive.
#
# The MLX backend links libmlxc and libmlx from Homebrew, and libmlx loads
# mlx.metallib (the Metal kernels) from the directory it was loaded from.
# Homebrew builds neither as a static library, so the release ships the
# binary with those three files. Their load commands are rewritten so the
# binary looks for the libraries, in this order:
#   1. beside itself            the archive as extracted (a CI job, a scratch dir)
#   2. ../lib/janis             where install.sh puts them (~/.local/lib/janis)
#   3. Homebrew's lib           a machine that has brew install mlx mlx-c
# Apple silicon only: MLX has no Metal on Intel.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <out.tar.gz>" >&2; exit 1; }
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "✗ the MLX bundle builds on Apple silicon only" >&2; exit 1; }
mkdir -p "$(dirname "$1")"
out=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")

brew_lib=$(brew --prefix)/lib
mlx_lib=$(brew --prefix mlx 2>/dev/null || true)/lib
mlxc_lib=$(brew --prefix mlx-c 2>/dev/null || true)/lib
for f in "$mlx_lib/libmlx.dylib" "$mlx_lib/mlx.metallib" "$mlxc_lib/libmlxc.dylib"; do
	[ -f "$f" ] || { echo "✗ $f is missing: brew install mlx mlx-c" >&2; exit 1; }
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
go build -trimpath -ldflags "-s -w" -o "$work/janis" ./cmd/janis
cp "$mlx_lib/libmlx.dylib" "$mlx_lib/mlx.metallib" "$mlxc_lib/libmlxc.dylib" "$work/"
chmod u+w "$work"/*.dylib

# loaded_path <mach-o> <lib> — the path a Mach-O file loads lib from today.
loaded_path() { otool -L "$1" | awk -v lib="$2" 'NR > 1 && index($1, lib) { print $1; exit }'; }

install_name_tool -id @rpath/libmlx.dylib "$work/libmlx.dylib"
install_name_tool -id @rpath/libmlxc.dylib \
	-change "$(loaded_path "$work/libmlxc.dylib" libmlx.dylib)" @loader_path/libmlx.dylib \
	"$work/libmlxc.dylib"
install_name_tool \
	-change "$(loaded_path "$work/janis" libmlxc.dylib)" @rpath/libmlxc.dylib \
	-change "$(loaded_path "$work/janis" libmlx.dylib)" @rpath/libmlx.dylib \
	"$work/janis"
# The cgo link put Homebrew's lib first; the search order above wants it last.
install_name_tool -delete_rpath "$brew_lib" "$work/janis" 2>/dev/null || true
for rpath in @executable_path @executable_path/../lib/janis "$brew_lib"; do
	install_name_tool -add_rpath "$rpath" "$work/janis"
done
# Rewriting load commands breaks the signatures; macOS refuses an unsigned arm64 binary.
codesign --force --sign - "$work/janis" "$work/libmlx.dylib" "$work/libmlxc.dylib"

for f in janis libmlx.dylib libmlxc.dylib; do
	otool -L "$work/$f" | grep -q /opt/homebrew/opt && { echo "✗ $f still loads from Homebrew's Cellar" >&2; otool -L "$work/$f" >&2; exit 1; }
done
COPYFILE_DISABLE=1 tar -C "$work" -czf "$out" janis libmlx.dylib libmlxc.dylib mlx.metallib
echo "✓ $out: janis, libmlx.dylib, libmlxc.dylib, mlx.metallib (mlx $(ls "$(brew --cellar mlx)" | tail -1), mlx-c $(ls "$(brew --cellar mlx-c)" | tail -1))"
