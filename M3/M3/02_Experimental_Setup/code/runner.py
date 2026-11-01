"""Resumable parallel task runner: every experiment script defines tasks (hashable keys) and a worker function;
results are appended to a JSONL file, one line per task, so an interrupted run continues where it stopped."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")
import json, time, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(name):
    f = os.path.join(DATA, name)
    if not os.path.exists(f): return []
    out = []
    for l in open(f):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except json.JSONDecodeError: pass
    return out


def _safe(fn, key):
    t0 = time.perf_counter()
    try:
        r = fn(key)
    except Exception:
        r = dict(error=traceback.format_exc()[-1500:])
    r["key"] = list(key) if isinstance(key, tuple) else key
    r.setdefault("wall", time.perf_counter() - t0)
    return r


def _tup(x):
    return tuple(_tup(y) for y in x) if isinstance(x, (list, tuple)) else x


def run(name, keys, fn, workers=int(os.environ.get("NWORK", 11)), init=None, show=None):
    done = {_tup(r["key"]) for r in load(name) if "error" not in r}
    todo = [k for k in keys if _tup(k) not in done]
    print(f"[{name}] {len(todo)} of {len(keys)} tasks to run on {workers} workers", flush=True)
    if not todo: return
    t0 = time.time()
    path = os.path.join(DATA, name)
    with ProcessPoolExecutor(workers, initializer=init) as ex, open(path, "a") as f:
        futs = [ex.submit(_safe, fn, k) for k in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r) + "\n"); f.flush()
            msg = show(r) if (show and "error" not in r) else (r.get("error", "")[-300:] if "error" in r else "")
            print(f"[{name} {i}/{len(todo)} {time.time()-t0:6.0f}s] {r['key']} {msg}", flush=True)
