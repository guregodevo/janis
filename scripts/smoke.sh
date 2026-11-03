#!/usr/bin/env bash
# scripts/smoke.sh <janis> [port] — the release as a user runs it: serve
# --detach pulls Qwen3-0.6B-4bit and serves it; one question, and the answer
# must hold 5. Run from the directory the binary sits in, as a CI job would.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 <janis> [port]" >&2; exit 1; }
janis=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
port=${2:-8090}
cd "$(dirname "$janis")"
log=$PWD/janis.log

"$janis" serve --detach --model Qwen3-0.6B-4bit --port "$port" --log "$log"
out=$(curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
	-d '{"model":"Qwen3-0.6B-4bit","messages":[{"role":"user","content":"What is 2+3? Answer with the number."}],"max_tokens":8,"temperature":0}' || true)
echo "$out"
tail -40 "$log"
[ -r /proc/cpuinfo ] && { grep -o -m1 -E "asimddp|i8mm|sve" /proc/cpuinfo | sort -u | tr "\n" " "; echo; } || true
echo "$out" | grep -q '"content":"[^"]*5' || { echo "✗ no 5 in the answer" >&2; exit 1; }
echo "✓ $janis answered"
