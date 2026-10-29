"""Rebuild an fp32 checkpoint from a web export (int8 + scales), to measure what quantization costs."""
import json, sys, os, numpy as np
sys.path.insert(0, "tools"); import ckpt
d, out = sys.argv[1], sys.argv[2]; meta = json.load(open(f"{d}/meta.json"))
blob = b"".join(open(f"{d}/{c}", "rb").read() for c in meta["chunks"]); T = {}
for t in meta["tensors"]:
    a = np.frombuffer(blob, dtype=np.int8 if t["kind"] == "int8" else np.float32, count=t["bytes"] // (1 if t["kind"] == "int8" else 4), offset=t["offset"])
    T[t["name"]] = a.reshape(t["shape"])
W = {n: (T[n].astype(np.float32) * T[n + ".scale"][:, None] if T[n].dtype == np.int8 else T[n]) for n in ckpt.names(meta["cfg"]["L"])}
ckpt.write(out, meta["cfg"], W)
