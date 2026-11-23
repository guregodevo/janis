#!/usr/bin/env python3
"""Agent traffic against janis serve, replayed from memdoor meters.

Each line of a memdoor meter.jsonl is one model call: its session, its input
tokens and its output tokens. A session is one workflow step's conversation;
its calls grow it a tool result at a time. This replays those shapes: N agents
each take the next session and send its calls in order, the conversation
growing by the call's new input (filler text) and the reply janis gave, with
max_tokens the call's recorded output. Every 10 s it prints the output tok/s
and the calls in flight; at the end, the totals.

  python3 bench/agent_load.py --port 8093 --model Qwen3.8-27B-4bit \\
      --agents 10 --seconds 600 meter.jsonl [meter2.jsonl ...]

Run janis with JANIS_PARALLEL=N (and JANIS_BATCH_STATS=1 for where the
time goes).
"""
import argparse
import json
import random
import threading
import time
import urllib.request

WORDS = ("time year people way day man thing woman life child world school "
         "state family student group country problem hand part place case week "
         "company system program question work government number night point "
         "home water room mother area money story fact month lot right study "
         "book eye job word business issue side kind head house service friend "
         "father power hour game line end member law car city community name").split()


SHARED = 8000  # tokens every session's prompt starts with


def filler(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(max(n, 1)))


def sessions(paths):
    calls = {}
    order = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            s = p + "|" + r.get("session", "?")
            if s not in calls:
                calls[s] = []
                order.append(s)
            calls[s].append((r["input_tokens"], r["output_tokens"]))
    return [calls[s] for s in order]


COMMON = "Write a long story about the words you are given. Never stop early.\n" + filler(random.Random(-1), SHARED - 20)


class Load:
    def __init__(self, a, todo):
        self.a, self.todo, self.mu = a, todo, threading.Lock()
        self.out = self.calls = self.inflight = self.errors = 0
        self.win_out = 0
        self.durs = []

    def next_session(self):
        with self.mu:
            return self.todo.pop(0) if self.todo else None

    def post(self, msgs, max_tokens):
        body = json.dumps({"model": self.a.model, "messages": msgs, "max_tokens": max_tokens,
                           "temperature": 0.7, "stream": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.a.port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)

    def agent(self, k, stop):
        rng = random.Random(k)
        while not stop.is_set():
            calls = self.next_session()
            if calls is None:
                return
            # Every step's conversation opens with the same system prompt and
            # tools (here COMMON), then its own task.
            msgs = [{"role": "system", "content": COMMON},
                    {"role": "user", "content": filler(rng, calls[0][0] - SHARED)}]
            sent = calls[0][0]
            for i, (inp, out) in enumerate(calls):
                if stop.is_set():
                    return
                if i > 0:
                    msgs.append({"role": "user", "content": filler(rng, max(inp - sent, 1))})
                with self.mu:
                    self.inflight += 1
                t = time.time()
                try:
                    r = self.post(msgs, max(out, 1))
                    text = r["choices"][0]["message"].get("content") or ""
                    n = r.get("usage", {}).get("completion_tokens", 0)
                    sent = r.get("usage", {}).get("prompt_tokens", sent) + n
                except Exception as e:  # noqa: BLE001 - counted, the run goes on
                    text, n = "", 0
                    with self.mu:
                        self.errors += 1
                    print("error:", e, flush=True)
                with self.mu:
                    self.inflight -= 1
                    self.calls += 1
                    self.out += n
                    self.win_out += n
                    self.durs.append(time.time() - t)
                msgs.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--model", required=True)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("meters", nargs="+")
    a = ap.parse_args()
    todo = sessions(a.meters)
    random.Random(0).shuffle(todo)
    ld = Load(a, todo)
    stop = threading.Event()
    ts = [threading.Thread(target=ld.agent, args=(k, stop), daemon=True) for k in range(a.agents)]
    t0 = time.time()
    for t in ts:
        t.start()
    last = t0
    while time.time() - t0 < a.seconds and any(t.is_alive() for t in ts):
        time.sleep(10)
        now = time.time()
        with ld.mu:
            w, ld.win_out = ld.win_out, 0
            print(f"{now - t0:6.0f}s  {w / (now - last):6.1f} out tok/s  {ld.inflight} in flight  "
                  f"{ld.calls} calls  {ld.errors} errors", flush=True)
        last = now
    stop.set()
    wall = time.time() - t0
    d = sorted(ld.durs) or [0]
    print(f"total: {ld.out} out tok in {wall:.0f}s = {ld.out / wall:.1f} tok/s, {ld.calls} calls, "
          f"median call {d[len(d) // 2]:.1f}s, {ld.errors} errors", flush=True)


if __name__ == "__main__":
    main()
