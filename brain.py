"""Chand brain. Backends:
  meta_api      -> Meta Model API (https://api.meta.ai/v1, OpenAI-compatible), model muse-spark-1.3
                   (fallback muse-wonder-1.0). Every call is logged to logs/api_usage.jsonl; hard cap per process.
  glimmer_local -> llama.cpp server on 127.0.0.1 (Muse Glimmer GGUF), added for Moon mode.
"""
import json, os, time
from pathlib import Path
import httpx

HERE = Path(__file__).parent
META_BASE = "https://api.meta.ai/v1"
META_MODELS = ["muse-spark-1.3", "muse-wonder-1.0"]
LOCAL_BASE = "http://127.0.0.1:8080/v1"
MAX_API_CALLS = int(os.environ.get("CHAND_MAX_API_CALLS", "20"))
_calls = 0


def _key():
    for line in (HERE / ".env").read_text().splitlines():
        if line.startswith("META_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("META_API_KEY missing in .env")


def _log(rec):
    (HERE / "logs").mkdir(exist_ok=True)
    with open(HERE / "logs" / "api_usage.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def chat(messages, backend="meta_api", max_tokens=6000, dry=False, tag="", **extra):
    """Returns the reply text. dry=True prints the prompt and sends nothing."""
    global _calls
    if dry:
        print("---- DRY RUN prompt ----"); [print(m["role"], ":", m["content"]) for m in messages]
        return None
    if backend == "meta_api":
        if _calls >= MAX_API_CALLS:
            raise RuntimeError(f"API call cap reached ({MAX_API_CALLS})")
        base, headers, models = META_BASE, {"Authorization": f"Bearer {_key()}"}, META_MODELS
    else:
        base, headers, models = LOCAL_BASE, {}, ["glimmer"]
    last = None
    for model in models:
        t0 = time.time()
        try:
            if backend == "meta_api":
                _calls += 1
            r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=120,
                           json={"model": model, "messages": messages, "max_tokens": max_tokens, **extra})
            d = r.json()
            usage = d.get("usage", {})
            _log({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": backend, "model": model, "tag": tag,
                  "status": r.status_code, "s": round(time.time() - t0, 1), "usage": usage})
            text = d["choices"][0]["message"]["content"]
            if text:
                return text
            last = f"{model}: empty content ({usage})"
        except Exception as e:  # try the fallback model
            last = f"{model}: {e}"
    raise RuntimeError(f"all models failed: {last}")


def parse_json(text):
    """First {...} block in the reply."""
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])
