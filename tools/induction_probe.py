"""Loss on 'induction-predictable' positions (the token that followed the last earlier occurrence of the current
token in the same doc-window equals the actual next token) vs all other positions.  usage: induction_probe.py ckpt..."""
import numpy as np, sys
sys.path.insert(0, "tools"); import ckpt, gptref
T = 256; val = np.fromfile("data_bin/val.u16", dtype=np.uint16).astype(np.int64)
def ind_mask(x):
    last, m = {}, np.zeros(T, bool)
    for i in range(T):
        t = x[i]
        if t == gptref.EOT: last = {}
        if t in last and x[last[t] + 1] == x[i + 1]: m[i] = True
        last[t] = i
    return m
for path in sys.argv[1:]:
    cfg, W = ckpt.read(path); fwd = gptref.build_forward(cfg, W); li, lo = [], []
    for w in range(24):
        x = val[w * T:(w + 1) * T + 1]; lg, _ = fwd(x[:T]); mx = lg.max(-1, keepdims=True)
        loss = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1)) - lg[np.arange(T), x[1:]]
        m = ind_mask(x); li.append(loss[m]); lo.append(loss[~m])
    li, lo = np.concatenate(li), np.concatenate(lo)
    print(f"{path}: copy-predictable n={len(li)} loss={li.mean():.3f} | others n={len(lo)} loss={lo.mean():.3f} | gap={lo.mean()-li.mean():.3f}", flush=True)
