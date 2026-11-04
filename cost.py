"""Dollar cost of the AI calls, from logs/api_usage.jsonl (every Muse Spark and Jev call is logged there).

  python cost.py                 # cost per mission run (from logs/m5_run*.log start/end times) + total
  cost.since(t0) / cost.between(t0, t1)   # used by mission.py for the live cost line

Prices (USD per 1M tokens), checked 2026-11-05:
  muse-spark-1.3 / muse-wonder-1.0 (Meta Model API): $1.25 input, $4.25 output (reasoning tokens count as output),
      cached input $0.15. Source: third-party listings of Meta's rate sheet (dev.meta.ai/docs/pricing-rate-limits).
  jev (TypeSafe): $0.042 input, output free. Source: typesafe.ai / docs.typesafe.ai/go/models.
Local models (Glimmer) and the motors cost $0.
"""
import json, time
from pathlib import Path

HERE = Path(__file__).parent
PRICE = {  # $ per token
    "meta_api": {"in": 1.25e-6, "cached": 0.15e-6, "out": 4.25e-6},
    "typesafe": {"in": 0.042e-6, "out": 0.0},
}


def call_cost(rec):
    u = rec.get("usage") or {}
    if rec.get("backend") == "meta_api":
        p = PRICE["meta_api"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        return (u.get("prompt_tokens", 0) - cached) * p["in"] + cached * p["cached"] + u.get("completion_tokens", 0) * p["out"]
    if rec.get("backend") == "typesafe":
        return u.get("input_tokens", 0) * PRICE["typesafe"]["in"] + u.get("output_tokens", 0) * PRICE["typesafe"]["out"]
    return 0.0


def _records():
    f = HERE / "logs" / "api_usage.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line); r["_ts"] = time.mktime(time.strptime(r["t"], "%Y-%m-%dT%H:%M:%S")); out.append(r)
        except Exception:
            pass
    return out


def between(t0, t1=None):
    """dict: total $, per model, number of calls, for calls logged in [t0, t1] (unix time)."""
    t1 = t1 or time.time() + 1
    res = {"usd": 0.0, "calls": 0, "by_model": {}}
    for r in _records():
        if t0 - 1 <= r["_ts"] <= t1 + 1:
            c = call_cost(r)
            res["usd"] += c; res["calls"] += 1
            m = res["by_model"].setdefault(r.get("model", "?"), {"usd": 0.0, "calls": 0})
            m["usd"] += c; m["calls"] += 1
    return res


def since(t0):
    return between(t0)


def fmt(res):
    parts = ", ".join(f"{m} {v['calls']} calls ${v['usd']:.4f}" for m, v in res["by_model"].items())
    return f"${res['usd']:.4f} ({res['calls']} AI calls{': ' + parts if parts else ''})"


if __name__ == "__main__":
    recs = _records()
    if recs:
        print("all logged AI calls:", fmt(between(recs[0]["_ts"])))
    for f in sorted((HERE / "logs").glob("mission_*.json")):
        try:
            d = json.loads(f.read_text())
            if isinstance(d, dict) and "cost" in d:
                print(f.name, d["cost"].get("text"))
        except Exception:
            pass
