"""Sample passages from the training stream. A passage is the <=128-token span ending at a random position e,
clipped to the containing document; the model-side window is train[e-256:e]."""
import numpy as np, json, sys, tiktoken
n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
split = sys.argv[4] if len(sys.argv) > 4 else "train"
tok = np.memmap(f"data_bin/{split}.u16", dtype=np.uint16, mode="r")
docs = np.fromfile(f"data_bin/{split}_docs.i64", dtype=np.int64)
enc = tiktoken.get_encoding("gpt2"); rng = np.random.default_rng(seed)
res = []
while len(res) < n:
    e = int(rng.integers(512, len(tok)))
    d = int(np.searchsorted(docs, e - 1, side="right") - 1)    # doc containing position e-1
    s = max(e - 128, int(docs[d]) + 1)
    if e - s < 32 or (d + 1 < len(docs) and e > docs[d + 1]): continue
    ids = tok[s:e].tolist()
    res.append({"start": s, "end": e, "doc": d, "text": enc.decode(ids)})
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(len(res), "passages; mean tokens", np.mean([r["end"] - r["start"] for r in res]))
