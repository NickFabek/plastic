"""NumPy fp64 reference forward pass of the PLASTIC GPT (shared by ref_forward.py and analysis tools)."""
import numpy as np
EOT = 50256
def build_forward(cfg, W):
    V, d, L, H, T = cfg["V"], cfg["d"], cfg["L"], cfg["H"], cfg["T"]; hd = d // H; h2 = hd // 2
    fr = 10000.0 ** (-2 * np.arange(h2) / hd); ang = np.arange(T)[:, None] * fr[None]; COS, SIN = np.cos(ang), np.sin(ang)
    def rms(x): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    def rope(x):
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
        return rms(x) @ W["lm_head"][:V].T + W["lm_bias"][:V], acts
    return forward
