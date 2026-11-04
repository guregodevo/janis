# usage: python3 bench/mlx.py PORT MODEL — prefill at ~600 and ~9,000 tokens, decode over 128 tokens; 3 runs each, median.
# Every prompt opens with a fresh nonce so no prompt cache can skip the work.
import json, urllib.request, time, sys, uuid, statistics
port, model = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:%s/v1/chat/completions" % port
UNIT = "the quick brown fox jumps over the lazy dog and then reads a file "

def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)

def prefill(reps):
    text = uuid.uuid4().hex + " " + UNIT * reps + " Say OK."
    t = time.time()
    d = json.load(post({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0}))
    dt = time.time() - t
    n = d["usage"]["prompt_tokens"]
    return n, n / dt

def decode(n=128):
    body = {"model": model, "stream": True, "max_tokens": n, "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": uuid.uuid4().hex + " Count from 1 to 500, one number per line, nothing else."}]}
    r = post(body)
    times, usage = [], None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        ev = json.loads(p)
        if ev.get("usage"):
            usage = ev["usage"]
        for c in ev.get("choices", []):
            delta = c.get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                times.append(time.time())
    toks = (usage or {}).get("completion_tokens") or len(times)
    return toks, (toks - 1) / (times[-1] - times[0])

prefill(20)  # warm-up: load and first-answer costs
for reps, label in ((40, "pp~512"), (640, "pp~8k")):
    runs = [prefill(reps) for _ in range(3)]
    print("%s prompt_tokens %d tok/s %s median %.1f" % (label, runs[0][0], [round(r[1], 1) for r in runs], statistics.median(r[1] for r in runs)), flush=True)
runs = [decode() for _ in range(3)]
print("tg128 tokens %s tok/s %s median %.1f" % ([r[0] for r in runs], [round(r[1], 1) for r in runs], statistics.median(r[1] for r in runs)), flush=True)
