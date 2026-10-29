"""A cheap "Jev-like" labeler: semi-supervised naive Bayes (Nigam et al. 2000), closed form.
Fit token|concept statistics on the Jev-labeled passages, pick the NB temperature on held-out Jev labels, then
soft-label many more unlabeled passages and refit (EM). Writes labels/<name>.npz with start/end/Y/facets for
construct.py (labels=<name>), and reports held-out agreement with Jev + the oracle-mixture next-token gain.
usage: propagate_labels.py <source: jev|kmeans> <n_unlabeled> <out_name>"""
import numpy as np, scipy.sparse as sp, sys, time
src, n_unl, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
V = 50257; t0 = time.time(); rng = np.random.default_rng(0)
tok = np.memmap("data_bin/train.u16", dtype=np.uint16, mode="r"); docs = np.fromfile("data_bin/train_docs.i64", dtype=np.int64)
cnt = np.load("stats/unigram.npy").astype(np.float64); p = (cnt + 0.5) / (cnt.sum() + 0.5 * V); logp = np.log(p)
z = np.load("labels/jev_train.npz")
if src == "jev": Y0 = np.concatenate([z["topic"], z["fmt"]], 1).astype(np.float64); facets = [160, 40]
else: Y0 = np.load(f"labels/{src}.npz")["Y"].astype(np.float64); facets = [Y0.shape[1]]
def counts(starts, ends):
    rows = np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(zip(starts, ends))])
    cols = np.concatenate([np.asarray(tok[s:e], dtype=np.int64) for s, e in zip(starts, ends)])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(starts), V))
X0 = counts(z["start"], z["end"])
# sample unlabeled passages the same way as the labeled ones (<=128 tokens ending at a random position, within a doc)
us, ue = [], []
labeled = set(zip(z["start"].tolist(), z["end"].tolist()))
while len(ue) < n_unl:
    e = rng.integers(512, len(tok), size=n_unl); d = np.searchsorted(docs, e - 1, side="right") - 1
    s = np.maximum(e - 128, docs[d] + 1); ok = (e - s) >= 32
    us += s[ok].tolist(); ue += e[ok].tolist()
us, ue = np.array(us[:n_unl]), np.array(ue[:n_unl]); Xu = counts(us, ue)
print(f"unlabeled passages={len(us)} t={time.time()-t0:.0f}s", flush=True)
def fit(X, Y, beta=1.0):
    C = np.asarray(X.T @ Y); nc = C.sum(0); bc = beta * nc
    return np.log((C + bc * p[:, None]) / (nc + bc)) - logp[:, None], nc / nc.sum()
def posterior(X, PMI, prior, tau):
    out = []; off = 0
    S = np.asarray(X @ PMI)
    for fs in facets:
        sl = slice(off, off + fs); l = np.log(prior[sl] / prior[sl].sum()) + tau * S[:, sl]
        l -= l.max(1, keepdims=True); e = np.exp(l); out.append(e / e.sum(1, keepdims=True)); off += fs
    return np.concatenate(out, 1)
def soft_ce(P, Y):   # mean over facets of CE(Y, P)
    off = 0; tot = 0
    for fs in facets: sl = slice(off, off + fs); tot += -(Y[:, sl] * np.log(P[:, sl] + 1e-12)).sum(1).mean(); off += fs
    return tot / len(facets)
def acc(P, Y):
    off = 0; r = []
    for fs in facets: sl = slice(off, off + fs); r.append(float((P[:, sl].argmax(1) == Y[:, sl].argmax(1)).mean())); off += fs
    return r
def oracle_gain(C_PMI_fit_X, C_PMI_fit_Y, Xte, Yte, beta=1.0):
    C = np.asarray(C_PMI_fit_X.T @ C_PMI_fit_Y); off = 0; gains = []
    H_uni = -(np.asarray(Xte.sum(0)).ravel() * logp).sum() / Xte.sum()
    for fs in facets:
        sl = slice(off, off + fs); Cc = C[:, sl]; nc = Cc.sum(0); pc = (Cc + beta * nc * p[:, None]) / (nc * (1 + beta))
        ce = -(Xte.multiply(np.log(Yte[:, sl] / np.maximum(Yte[:, sl].sum(1, keepdims=True), 1e-9) @ pc.T))).sum() / Xte.sum()
        gains.append(float(H_uni - ce)); off += fs
    return gains
n = len(Y0); idx = rng.permutation(n); tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
PMI, prior = fit(X0[tr], Y0[tr])
best = None
for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    ce = soft_ce(posterior(X0[te], PMI, prior, tau), Y0[te])
    if best is None or ce < best[1]: best = (tau, ce)
tau = best[0]; P_te = posterior(X0[te], PMI, prior, tau)
print(f"NB tau={tau} held-out soft CE vs Jev={best[1]:.3f} argmax acc={acc(P_te, Y0[te])} t={time.time()-t0:.0f}s", flush=True)
print("oracle gain, stats from labeled 80% only:", [round(g, 3) for g in oracle_gain(X0[tr], Y0[tr], X0[te], Y0[te])], flush=True)
Yu = None
for it in range(2):   # EM: label unlabeled with current model, refit on labeled(train split) + unlabeled
    Yu = posterior(Xu, PMI, prior, tau)
    Xall = sp.vstack([X0[tr], Xu]); Yall = np.concatenate([Y0[tr], Yu])
    PMI, prior = fit(Xall, Yall)
    P_te = posterior(X0[te], PMI, prior, tau)
    print(f"EM{it}: held-out acc={acc(P_te, Y0[te])} soft CE={soft_ce(P_te, Y0[te]):.3f}  oracle gain={[round(g, 3) for g in oracle_gain(Xall, Yall, X0[te], Y0[te])]} t={time.time()-t0:.0f}s", flush=True)
# final propagated set = ALL Jev-labeled passages (true labels) + pseudo-labeled ones
Yu = posterior(Xu, PMI, prior, tau)
np.savez_compressed(f"labels/{out}.npz", start=np.concatenate([z["start"], us]), end=np.concatenate([z["end"], ue]),
                    Y=np.concatenate([Y0, Yu]).astype(np.float32), facets=np.array(facets), tau=tau)
print(f"DONE wrote labels/{out}.npz n={n + len(us)} t={time.time()-t0:.0f}s", flush=True)
