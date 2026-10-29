"""NumPy reference forward pass of the PLASTIC GPT (fp32), used to validate the CUDA trainer and the browser engine.
usage: ref_forward.py ckpt.bin [n_windows=32]   -> mean loss on the first n val windows (compare: train.exe mode=eval
       eval_tokens=n*T B=n); also writes runs/ref_logits.npy for the first 16 tokens of window 0 (browser test)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__)); import ckpt
cfg, W = ckpt.read(sys.argv[1]); nw = int(sys.argv[2]) if len(sys.argv) > 2 else 32
V, d, L, H, m, T = cfg["V"], cfg["d"], cfg["L"], cfg["H"], cfg["m"], cfg["T"]; hd = d // H; h2 = hd // 2; EOT = 50256
fr = 10000.0 ** (-2 * np.arange(h2) / hd); ang = np.arange(T)[:, None] * fr[None]; COS, SIN = np.cos(ang), np.sin(ang)
def rms(x): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
def rope(x):  # (H, T, hd)
    n = x.shape[1]; c, s = COS[:n], SIN[:n]; a, b = x[..., :h2], x[..., h2:]; return np.concatenate([a * c - b * s, a * s + b * c], -1)
def forward(tok):
    Tn = len(tok); x = W["wte"][tok].astype(np.float64)
    ds = np.zeros(Tn, dtype=int); cur = 0
    for i, t in enumerate(tok):
        if t == EOT: cur = i
        ds[i] = cur
    mask = (np.arange(Tn)[None, :] <= np.arange(Tn)[:, None]) & (np.arange(Tn)[None, :] >= ds[:, None])
    acts = []
    for l in range(L):
        qkv = rms(x) @ W[f"wqkv{l}"].T
        q, k, v = [qkv[:, i * d:(i + 1) * d].reshape(Tn, H, hd).transpose(1, 0, 2) for i in range(3)]
        q, k = rope(q), rope(k)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd); s = np.where(mask[None], s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        y = (p @ v).transpose(1, 0, 2).reshape(Tn, d)
        x = x + y @ W[f"wo{l}"].T
        a = rms(x) @ W[f"w1_{l}"].T + W[f"b1_{l}"]; acts.append(a)
        x = x + (np.maximum(a, 0) ** 2) @ W[f"w2_{l}"].T
    logits = rms(x) @ W["lm_head"][:V].T + W["lm_bias"][:V]
    return logits, acts
val = np.fromfile("data_bin/val.u16", dtype=np.uint16).astype(np.int64)
losses = []
for w in range(nw):
    tok, tgt = val[w * T:(w + 1) * T], val[w * T + 1:(w + 1) * T + 1]
    lg, acts = forward(tok); mx = lg.max(-1, keepdims=True); lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
    losses.append(lse - lg[np.arange(T), tgt])
    if w == 0:
        lg16, acts16 = forward(np.concatenate([[EOT], tok[:15]]))
        np.save("runs/ref_logits.npy", lg16.astype(np.float32)); np.save("runs/ref_tokens.npy", np.concatenate([[EOT], tok[:15]]).astype(np.int32))
        np.save("runs/ref_acts0.npy", acts16[0].astype(np.float32))
print(f"REF mean loss over {nw} windows = {np.mean(losses):.5f}")
