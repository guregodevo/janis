"""Time one prefill on a janis server: an 8,404-token prompt, one token out.

    python3 bench/prefill.py [port] [model]

Warm the server first with a short request (the first answer repacks the
weights), and send nothing else meanwhile: a prompt sharing a prefix with an
earlier one is read from the cache, not prefilled.
"""
import json
import sys
import time
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8090"
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-1.7B-4bit"
words = "the quick brown fox jumps over the lazy dog and then reads a file " * 600
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": words + " Say OK."}],
    "max_tokens": 1,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
start = time.time()
with urllib.request.urlopen(req, timeout=3600) as r:
    usage = json.load(r).get("usage", {})
took = time.time() - start
n = usage.get("prompt_tokens", 0)
print(f"prompt_tokens {n} seconds {took:.1f} tok/s {n / took:.1f}")
