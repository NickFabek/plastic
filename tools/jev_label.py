"""Bin passages into the taxonomy with Jev Choice questions (calibrated distribution over all options).
Resumable: every response is cached on disk by request hash. Usage:
  python tools/jev_label.py <passages.json> <out.npz> [variant=desc|nodesc] [limit] [workers]"""
import json, sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "tools"); from apiclient import jev, spend
tax = json.load(open("labels/taxonomy_raw.json", encoding="utf-8"))
TOPICS = [c for g in tax["topic"] for c in g["concepts"]]
GROUP_OF = [gi for gi, g in enumerate(tax["topic"]) for _ in g["concepts"]]
FORMATS = tax["format"]
def questions(variant):
    d = variant == "desc"
    return {"topic": {"type": "choice", "instructions": "Which subject-matter concept best describes this passage?",
                      "criteria": {c["id"]: (c["desc"] if d else None) for c in TOPICS}},
            "format": {"type": "choice", "instructions": "Which genre / document type / format best describes this passage?",
                       "criteria": {c["id"]: (c["desc"] if d else None) for c in FORMATS}}}
def label(text, variant):
    r = jev(text, questions(variant))
    a = r["answers"]
    pt = np.array([a["topic"]["probabilities"].get(c["id"], 0.0) for c in TOPICS], dtype=np.float32)
    pf = np.array([a["format"]["probabilities"].get(c["id"], 0.0) for c in FORMATS], dtype=np.float32)
    return pt, pf, a["topic"]["confidence"], a["format"]["confidence"], r["usage"]["input_tokens"]
if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "desc"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    P = json.load(open(src, encoding="utf-8"))[:limit]
    t0 = time.time(); res = [None] * len(P); done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(label, p["text"], variant): i for i, p in enumerate(P)}
        for f in futs:
            i = futs[f]
            try: res[i] = f.result()
            except Exception as e: print("ERROR", i, str(e)[:200], flush=True)
            done += 1
            if done % 500 == 0: print(f"PROGRESS {done}/{len(P)} t={time.time()-t0:.0f}s spend={spend().get('jev',{}).get('usd',0):.3f}", flush=True)
    ok = [i for i, r in enumerate(res) if r is not None]
    np.savez_compressed(out, idx=np.array(ok), topic=np.stack([res[i][0] for i in ok]), fmt=np.stack([res[i][1] for i in ok]),
                        conf_topic=np.array([res[i][2] for i in ok]), conf_fmt=np.array([res[i][3] for i in ok]),
                        tokens=np.array([res[i][4] for i in ok]), group_of=np.array(GROUP_OF),
                        start=np.array([P[i]["start"] for i in ok]), end=np.array([P[i]["end"] for i in ok]))
    tk = np.array([res[i][4] for i in ok])
    print(f"DONE n={len(ok)} failed={len(P)-len(ok)} mean_input_tokens={tk.mean():.0f} est_usd_per_1k={tk.mean()*0.04e-6*1000:.4f} t={time.time()-t0:.0f}s", flush=True)
