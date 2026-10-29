"""Compare runs against a baseline from runs/<name>/metrics.json.
For each run: final val loss, delta vs baseline, and the *token multiplier*: how many baseline tokens are needed to
reach this run's loss at a given token count (log-log interpolation of the baseline curve). >1 = run is ahead.
usage: compare_runs.py <baseline> <run> [run ...]   (glob patterns allowed)"""
import json, sys, glob, numpy as np
def load(n):
    m = json.load(open(f"runs/{n}/metrics.json")); c = m["curve"]
    return m, np.array([p["tokens"] for p in c], float), np.array([p["val_loss"] for p in c], float), c
def tokens_to_reach(T, Lc, target):   # first crossing of the (monotone-ish) baseline curve, linear in log-tokens
    Lm = np.minimum.accumulate(Lc)
    if target < Lm[-1]: return np.inf
    i = int(np.argmax(Lm <= target))
    if i == 0: return T[0]
    t0, t1, l0, l1 = np.log(max(T[i - 1], 1)), np.log(T[i]), Lm[i - 1], Lm[i]
    return float(np.exp(t0 + (t1 - t0) * (l0 - target) / max(l0 - l1, 1e-9)))
base = sys.argv[1]; names = []
import os
for pat in sys.argv[2:]:
    hits = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"runs/{pat}/metrics.json"))
    names += hits if hits else [pat]
mb, Tb, Lb, _ = load(base)
print(f"{'run':34s} {'final':>7s} {'d_base':>7s} {'step0':>7s} {'mult@25%':>8s} {'mult@50%':>8s} {'mult@end':>8s}  extra")
for n in [base] + [x for x in names if x != base]:
    try: m, T, Lc, c = load(n)
    except Exception as e: print(f"{n:34s} (missing)"); continue
    mults = []
    for frac in (0.25, 0.5, 1.0):
        j = int(np.argmin(np.abs(T - frac * T[-1]))); tb = tokens_to_reach(Tb, Lb, Lc[j]); mults.append(tb / T[j] if T[j] > 0 else np.nan)
    ex = " ".join(f"{k}={c[-1][k]:.3f}" for k in ("fid0", "fid1", "concept_ce", "anchor_dev") if k in c[-1])
    print(f"{n:34s} {Lc[-1]:7.4f} {Lc[-1] - Lb[-1]:+7.4f} {Lc[0]:7.3f} {mults[0]:8.2f} {mults[1]:8.2f} {mults[2]:8.2f}  {ex}")

# budget-curve multiplier: baselines trained to completion at several budgets (each with its own LR schedule)
import os as _os
bud = [(n, json.load(open(f"runs/{n}/metrics.json"))) for n in ("bud_base_12M", "bud_base_25M", base, "bud_base_100M") if _os.path.exists(f"runs/{n}/metrics.json")]
bud = sorted([(m["curve"][-1]["tokens"], m["final_val_loss"]) for n, m in bud if m["curve"][-1]["tokens"] > 0])
if len(bud) >= 2:
    BT, BL = np.log(np.array([b[0] for b in bud], float)), np.array([b[1] for b in bud])
    print(f"\nbaseline budget curve: " + ", ".join(f"{np.exp(t)/1e6:.0f}M->{l:.4f}" for t, l in zip(BT, BL)))
    print(f"{'run':34s} {'final':>7s} {'eff. baseline tokens':>21s} {'multiplier':>10s}")
    for n in [x for x in names if x != base]:
        try: m = json.load(open(f"runs/{n}/metrics.json"))
        except Exception: continue
        Lr, Tr = m["final_val_loss"], m["curve"][-1]["tokens"]
        # BL decreases with budget; interpolate/extrapolate log-tokens linearly in loss
        i = int(np.clip(np.searchsorted(-BL, -Lr), 1, len(BL) - 1))
        te = BT[i - 1] + (BT[i] - BT[i - 1]) * (BL[i - 1] - Lr) / (BL[i - 1] - BL[i])
        flag = "" if BL.min() <= Lr <= BL.max() else " (extrapolated)"
        print(f"{n:34s} {Lr:7.4f} {np.exp(te)/1e6:18.1f}M {np.exp(te)/Tr:10.2f}{flag}")
