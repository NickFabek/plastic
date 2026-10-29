"""Cached, spend-tracked clients for Jev (TypeSafe System One) and OpenAI. Every response is cached on disk
keyed by a hash of the request body, so nothing is paid for twice."""
import hashlib, json, os, time, threading, httpx
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache"); SPEND = os.path.join(ROOT, "api_spend.json")
PRICES = {"jev": {"in": 0.04e-6, "out": 0.0}, "gpt-6-luna": {"in": 0.10e-6, "out": 0.50e-6}}
_lock = threading.Lock()
def _env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        if "=" in line: k, v = line.strip().split("=", 1); env[k] = v.strip().strip('"')
    return env
ENV = _env()
def _load_spend():
    try: return json.load(open(SPEND))
    except Exception: return {}
def add_spend(provider, model, tin, tout):
    with _lock:
        s = _load_spend(); p = s.setdefault(provider, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0})
        pr = PRICES.get(model, PRICES["jev"] if provider == "jev" else PRICES["gpt-6-luna"])
        p["usd"] += tin * pr["in"] + tout * pr["out"]; p["input_tokens"] += tin; p["output_tokens"] += tout; p["requests"] += 1
        for i in range(20):
            try:
                json.dump(s, open(SPEND + ".tmp", "w"), indent=1); os.replace(SPEND + ".tmp", SPEND); break
            except OSError: time.sleep(0.05 * (i + 1))
def spend():
    return _load_spend()
def _cache_path(kind, body):
    h = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    d = os.path.join(CACHE, kind, h[:2]); os.makedirs(d, exist_ok=True); return os.path.join(d, h + ".json")
def _post(url, headers, body, tries=8):
    delay = 2.0
    for i in range(tries):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=300)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(delay); delay = min(delay * 2, 60); continue
            if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            time.sleep(delay); delay = min(delay * 2, 60)
    raise RuntimeError("too many retries")
def jev(state, questions, model="jev-latest"):
    body = {"state": state, "model": model, "questions": questions}
    path = _cache_path("jev", body)
    if os.path.exists(path): return json.load(open(path, encoding="utf-8"))
    resp = _post("https://api.typesafe.ai/v1/systemone",
                 {"Authorization": f"Bearer {ENV['TYPESAFE_API_KEY']}", "Content-Type": "application/json"}, body)
    json.dump(resp, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    u = resp.get("usage", {}); add_spend("jev", "jev", u.get("input_tokens", 0), u.get("output_tokens", 0)); return resp
def openai_chat(messages, model="gpt-6-luna", **kw):
    body = {"model": model, "messages": messages, **kw}
    path = _cache_path("openai", body)
    if os.path.exists(path): return json.load(open(path, encoding="utf-8"))
    resp = _post("https://api.openai.com/v1/chat/completions",
                 {"Authorization": f"Bearer {ENV['OPENAI_API_KEY']}", "Content-Type": "application/json"}, body)
    json.dump(resp, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    u = resp.get("usage", {}); add_spend("openai", model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)); return resp
