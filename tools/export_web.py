"""Export a trainer checkpoint for the browser demo: int8 matrices with per-row fp32 scales, fp32 biases,
packed into <=12 MB chunks + meta.json.  Optional concept metadata (names of constructed neurons).
usage: export_web.py ckpt.bin out_dir [--concepts meta.json] [--label NAME]"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import ckpt
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--concepts", default="")
ap.add_argument("--label", default=""); ap.add_argument("--b64", type=int, default=1, help="write base64 .txt chunks (artifact-servable)"); ap.add_argument("--neuron_labels", default="", help="json: per-layer list of names for every neuron")
a = ap.parse_args()
cfg, W = ckpt.read(a.ckpt); os.makedirs(a.out, exist_ok=True)
blob = bytearray(); tensors = []
def put(name, arr, kind):
    global blob
    while len(blob) % 16: blob += b"\0"
    off = len(blob); data = np.ascontiguousarray(arr).tobytes(); blob += data
    tensors.append({"name": name, "kind": kind, "shape": list(arr.shape), "offset": off, "bytes": len(data)})
err = {}
for n in ckpt.names(cfg["L"]):
    x = W[n]
    if x.ndim == 2:
        s = np.abs(x).max(1) / 127.0; s[s == 0] = 1.0
        q = np.clip(np.round(x / s[:, None]), -127, 127).astype(np.int8)
        err[n] = float(np.abs(q * s[:, None] - x).mean() / (np.abs(x).mean() + 1e-12))
        put(n, q, "int8"); put(n + ".scale", s.astype(np.float32), "f32")
    else: put(n, x.astype(np.float32), "f32")
import base64, glob
for old in glob.glob(os.path.join(a.out, "w*.bin")) + glob.glob(os.path.join(a.out, "w*.txt")): os.remove(old)
CH = (10 << 20) if a.b64 else (12 << 20); chunks = []
for i in range(0, len(blob), CH):
    if a.b64:
        fn = f"w{i // CH}.txt"; open(os.path.join(a.out, fn), "wb").write(base64.b64encode(bytes(blob[i:i + CH])))
    else:
        fn = f"w{i // CH}.bin"; open(os.path.join(a.out, fn), "wb").write(blob[i:i + CH])
    chunks.append(fn)
meta = {"cfg": cfg, "tensors": tensors, "chunks": chunks, "chunk_size": CH, "total_bytes": len(blob), "label": a.label, "encoding": "b64" if a.b64 else "raw"}
if a.concepts:
    m = json.load(open(a.concepts)); meta["concepts"] = m
if a.neuron_labels: meta["neuron_labels"] = json.load(open(a.neuron_labels))
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
print(f"exported {a.ckpt} -> {a.out}: {len(blob)/1e6:.1f} MB in {len(chunks)} chunks; mean rel quant err {np.mean(list(err.values())):.4f}")
