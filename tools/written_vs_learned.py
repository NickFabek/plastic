"""Layer-0 purity@20 (fine topic / coarse group / format) of the written concept neurons (0..K-1) vs the learned
neurons (K..m-1) of the same layer, for one or more dumps.  usage: written_vs_learned.py dump.bin ... """
import numpy as np, sys
z = np.load("labels/jev_probe.npz"); fine = z["topic"].argmax(1); grp = z["group_of"][fine]; fmt = z["fmt"].argmax(1); K = 200
def purity(X, lab, k=20):
    top = np.argpartition(-X, k, axis=0)[:k]; labs = lab[top]
    return np.array([np.bincount(labs[:, j]).max() / k for j in range(X.shape[1])])
for p in sys.argv[1:]:
    with open(p, "rb") as f:
        h = np.fromfile(f, dtype=np.int32, count=8); n, L, m = int(h[2]), int(h[3]), int(h[4])
        Hh = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)
        Ha = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)
    for nm, X in (("post-act h", Hh[:, 0, :]), ("pre-act a", Ha[:, 0, :])):
        live = X.std(0) > 1e-8
        w, l = X[:, :K][:, live[:K]], X[:, K:][:, live[K:]]
        r = lambda U: "/".join(f"{purity(U, lab).mean():.2f}" for lab in (fine, grp, fmt))
        own_t = (X[:, :160].argmax(1) == fine).mean(); own_f = (X[:, 160:200].argmax(1) == fmt).mean()
        print(f"{p.split('/')[-2]:10s} {nm:10s} written({w.shape[1]}) {r(w)} | learned({l.shape[1]}) {r(l)} | argmax-acc topic {own_t:.3f} fmt {own_f:.3f}")
