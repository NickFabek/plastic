"""H1 analysis: do MLP neurons line up with LLM-named abstractions better than random directions, and at what
granularity?  Input: a trainer dump (per probe passage, per layer, mean post-activation h of every neuron) + Jev
probe labels (argmax fine topic / coarse group / format).
Per unit (neuron, or unit-norm random direction in the same activation space), one-sided (positive pole) metrics:
  purity@20  = share of the unit's 20 top-activating passages that carry its most common label
  max AUROC  = best AUROC over concepts with >= 15 probe passages
Both are compared with a label-permutation null (same units, labels shuffled across passages); a unit is
"aligned" if its score beats the null's 99th percentile.
usage: analyze_neurons.py dump.bin out.json [n_random_dirs=1024]"""
import numpy as np, json, sys
dump, out = sys.argv[1], sys.argv[2]; n_rand = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
with open(dump, "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=8); n, L, m = int(h[2]), int(h[3]), int(h[4])
    H = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)
z = np.load("labels/jev_probe.npz"); assert len(z["topic"]) == n
fine = z["topic"].argmax(1); grp = z["group_of"][fine]; fmt = z["fmt"].argmax(1)
rng = np.random.default_rng(0); perm = rng.permutation(n)
facets = {"fine": fine, "coarse": grp, "format": fmt}
def ranks(X):
    R = np.empty(X.shape, dtype=np.float64); idx = np.argsort(X, axis=0, kind="stable")
    np.put_along_axis(R, idx, np.broadcast_to(np.arange(1, X.shape[0] + 1, dtype=np.float64)[:, None], X.shape), axis=0); return R
def max_auc(Rk, lab):
    K = lab.max() + 1; Y = np.zeros((len(lab), K)); Y[np.arange(len(lab)), lab] = 1; npos = Y.sum(0); keep = npos >= 15
    Y, npos = Y[:, keep], npos[keep]; A = (Rk.T @ Y - npos * (npos + 1) / 2) / (npos * (len(lab) - npos)); return A.max(1)
def purity(X, lab, k=20):
    top = np.argpartition(-X, k, axis=0)[:k]; labs = lab[top]
    return np.array([np.bincount(labs[:, j]).max() / k for j in range(X.shape[1])])
res = {"n_probe": n, "L": L, "m": m, "layers": []}
for l in range(L):
    X = H[:, l, :]; live = X.std(0) > 1e-8
    Rd = rng.standard_normal((m, n_rand)); Rd /= np.linalg.norm(Rd, axis=0, keepdims=True)
    units = {"neuron": X[:, live], "random_dir": X @ Rd}
    entry = {"layer": l, "live_neurons": int(live.sum())}
    for uname, U in units.items():
        Rk = ranks(U); e = {}
        for fn, lab in facets.items():
            pu, pu0 = purity(U, lab), purity(U, lab[perm]); au, au0 = max_auc(Rk, lab), max_auc(Rk, lab[perm])
            e[f"{fn}_purity"] = float(pu.mean()); e[f"{fn}_purity_null"] = float(pu0.mean())
            e[f"{fn}_purity_aligned"] = float((pu > np.percentile(pu0, 99)).mean())
            e[f"{fn}_auc"] = float(au.mean()); e[f"{fn}_auc_null"] = float(au0.mean())
            e[f"{fn}_auc_aligned"] = float((au > np.percentile(au0, 99)).mean())
            if fn != "format": e[f"_{fn}_pu"] = pu   # kept for the granularity statistic
        # granularity: among units whose top passages are coherent at the coarse level, how many are also fine-coherent
        thr_c, thr_f = np.percentile(purity(U, grp[perm]), 99), np.percentile(purity(U, fine[perm]), 99)
        coarse_ok = e.pop("_coarse_pu") > thr_c; fine_ok = e.pop("_fine_pu") > thr_f
        e["fine_given_coarse"] = float(fine_ok[coarse_ok].mean()) if coarse_ok.any() else 0.0
        entry[uname] = e
    res["layers"].append(entry)
json.dump(res, open(out, "w"), indent=0)
for e in res["layers"]:
    s = f"L{e['layer']} live={e['live_neurons']:4d}"
    for u in ("neuron", "random_dir"):
        x = e[u]; s += (f" | {u[:6]}: pur(fine/coarse/fmt)={x['fine_purity']:.2f}/{x['coarse_purity']:.2f}/{x['format_purity']:.2f} "
                        f"aligned={x['fine_purity_aligned']:.2f}/{x['coarse_purity_aligned']:.2f}/{x['format_purity_aligned']:.2f} "
                        f"aucAligned(f/c)={x['fine_auc_aligned']:.2f}/{x['coarse_auc_aligned']:.2f} fine|coarse={x['fine_given_coarse']:.2f}")
    print(s)
