"""Anchor-strength files for the weight-space righting force. Each constructed entry gets strength 1/rms(tensor at
init), so anchor_lambda=10 means equilibrium deviations of ~10% of that tensor's typical weight size.
usage: anchor_variants.py init.bin mask_anchor.bin out.bin [subset=all|input|output]"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__)); import ckpt
cfg, P = ckpt.read(sys.argv[1]); _, M = ckpt.read(sys.argv[2]); out = sys.argv[3]; sub = sys.argv[4] if len(sys.argv) > 4 else "all"
A = {}
for n, m in M.items():
    sel = m != 0
    if sub == "input" and not (n == "wte" or n.startswith("wqkv") or n.startswith("wo") or n.startswith("w1_") or n.startswith("b1_")): sel = np.zeros_like(sel)
    if sub == "output" and not (n.startswith("w2_") or n == "lm_head"): sel = np.zeros_like(sel)
    r = np.sqrt((P[n][sel] ** 2).mean()) if sel.any() else 1.0
    A[n] = (sel / max(r, 1e-3)).astype(np.float32)
ckpt.write(out, cfg, A)
print(out, {n: int((a != 0).sum()) for n, a in A.items() if (a != 0).any()})
