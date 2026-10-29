"""Plot val loss vs tokens for a set of runs (log-x), plus the difference to a baseline.
usage: plot_runs.py out.png baseline run [run ...]"""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, base, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
def curve(n):
    c = json.load(open(f"runs/{n}/metrics.json"))["curve"]; return np.array([p["tokens"] for p in c], float), np.array([p["val_loss"] for p in c])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
Tb, Lb = curve(base)
for n in [base] + runs:
    try: T, Lc = curve(n)
    except Exception: continue
    m = T > 0
    ax[0].plot(T[m] / 1e6, Lc[m], label=n, lw=2 if n == base else 1.3)
    bmap = dict(zip(Tb, Lb)); common = [(t, l - bmap[t]) for t, l in zip(T, Lc) if t in bmap and t > 0]
    if common: ax[1].plot([c[0] / 1e6 for c in common], [c[1] for c in common], label=n, lw=2 if n == base else 1.3)
ax[0].set_xscale("log"); ax[0].set_xlabel("tokens (M)"); ax[0].set_ylabel("val loss (nats)"); ax[0].set_ylim(top=7.5); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
ax[1].set_xscale("log"); ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("tokens (M)"); ax[1].set_ylabel(f"val loss - {base}"); ax[1].grid(alpha=.3)
lo = min(-0.05, np.nanmin([0])); ax[1].set_ylim(-0.6, 0.3)
plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)
