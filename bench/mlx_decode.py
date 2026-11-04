# usage: python3 bench/mlx_decode.py PORT MODEL — decode tok/s over 128 streamed tokens:
# greedy after a ~9,000-token prompt (2 runs), and the server's default sampling
# (no temperature sent) after a short and a ~9,000-token prompt (3 runs each, median).
# Every prompt opens with a fresh nonce so no prompt cache skips the work.
import json, urllib.request, time, sys, uuid, statistics

port, model = sys.argv[1], sys.argv[2]
LONG = "the quick brown fox jumps over the lazy dog and then reads a file " * 640


def decode(prefix, ask, greedy):
    body = {"model": model, "stream": True, "max_tokens": 128, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": uuid.uuid4().hex + " " + prefix + ask}]}
    if greedy:
        body["temperature"] = 0
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
    times = []
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            ev = json.loads(line[5:])
            if any(c.get("delta", {}).get("content") for c in ev.get("choices", [])):
                times.append(time.time())
    return (len(times) - 1) / (times[-1] - times[0])


decode("", "Say hi.", True)  # warm-up
xs = [decode(LONG, " Now count from 1 to 500, one number per line, nothing else.", True) for _ in range(2)]
print("greedy after 9k: %s tok/s" % [round(x, 1) for x in xs], flush=True)
for label, prefix in (("short", ""), ("after 9k", LONG)):
    xs = [decode(prefix, "Write a long story about a lighthouse keeper.", False) for _ in range(3)]
    print("default sampling %s: %s median %.1f tok/s" % (label, [round(x, 1) for x in xs], statistics.median(xs)), flush=True)
