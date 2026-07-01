#!/usr/bin/env python3
"""Simple throughput bench for an OpenAI-compatible vLLM endpoint.
Measured by usage.completion_tokens (NOT by SSE events).
c=1  -> single-stream decode tok/s
c=N  -> aggregate tok/s = sum(completion_tokens)/wall + mean per-stream."""
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor

URL   = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PROMPT = ("Describe, step by step, the history of artificial intelligence from the "
          "1950s to today. Do not use lists; write in connected prose.")

def one(_):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAXTOK, "temperature": 0.7, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt

def run(conc, label=""):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    toks = sum(c for c, _ in res)
    per  = [c / d for c, d in res if d > 0]
    agg  = toks / wall if wall > 0 else 0
    avg_per = sum(per) / len(per) if per else 0
    print(f"{label}c={conc:<2} wall={wall:6.1f}s  tok={toks:<5}  "
          f"aggregate={agg:6.1f} tok/s  per-stream avg={avg_per:5.1f} tok/s")
    return {"c": conc, "wall": round(wall, 1), "tok": toks,
            "aggregate": round(agg, 1), "per_stream": round(avg_per, 1)}

print(f"# target={URL} model={MODEL} max_tokens={MAXTOK}")
run(1, "warmup ")  # warmup, result ignored
results = [run(c) for c in (1, 4, 8)]
print("JSON " + json.dumps(results))
