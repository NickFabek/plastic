"""Read/write PLASTIC trainer checkpoints (int32 hdr[16] + tensors in layout order, fp32)."""
import numpy as np
MAGIC = 0x504c4153
def names(L):
    out = ["wte"]
    for l in range(L): out += [f"wqkv{l}", f"wo{l}", f"w1_{l}", f"b1_{l}", f"w2_{l}"]
    return out + ["lm_head", "lm_bias"]
def shapes(cfg):
    Vp, d, L, m = cfg["Vp"], cfg["d"], cfg["L"], cfg["m"]
    s = {"wte": (Vp, d), "lm_head": (Vp, d), "lm_bias": (Vp,)}
    for l in range(L): s.update({f"wqkv{l}": (3 * d, d), f"wo{l}": (d, d), f"w1_{l}": (m, d), f"b1_{l}": (m,), f"w2_{l}": (d, m)})
    return s
def read(path):
    with open(path, "rb") as f:
        h = np.fromfile(f, dtype=np.int32, count=16)
        assert h[0] == MAGIC, path
        cfg = dict(Vp=int(h[2]), d=int(h[3]), L=int(h[4]), H=int(h[5]), m=int(h[6]), T=int(h[7]), V=int(h[8]))
        sh = shapes(cfg); W = {}
        for n in names(cfg["L"]): W[n] = np.fromfile(f, dtype=np.float32, count=int(np.prod(sh[n]))).reshape(sh[n])
    return cfg, W
def write(path, cfg, W):
    sh = shapes(cfg)
    with open(path, "wb") as f:
        h = np.zeros(16, dtype=np.int32); h[:9] = [MAGIC, 1, cfg["Vp"], cfg["d"], cfg["L"], cfg["H"], cfg["m"], cfg["T"], cfg["V"]]
        h.tofile(f)
        for n in names(cfg["L"]):
            a = np.ascontiguousarray(W[n], dtype=np.float32); assert a.shape == sh[n], (n, a.shape, sh[n]); a.tofile(f)
