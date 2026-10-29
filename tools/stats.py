"""Corpus statistics for closed-form construction (CPU only). Cached in stats/.
  unigram.npy           counts of every token over the full train stream
  bigram_svd.npz        rank-128 randomized SVD of S[v,w] = log(1 + c(v,w) / (kappa * p(w)))
                        (= smoothed log p(w|v) - log p(w), minus a per-context constant that softmax ignores)"""
import numpy as np, scipy.sparse as sp, time, os, sys
V = 50257; os.makedirs("stats", exist_ok=True); t0 = time.time()
tok = np.memmap("data_bin/train.u16", dtype=np.uint16, mode="r")
if not os.path.exists("stats/unigram.npy"):
    cnt = np.zeros(V, dtype=np.int64)
    for i in range(0, len(tok), 50_000_000): cnt += np.bincount(tok[i:i + 50_000_000], minlength=V)[:V]
    np.save("stats/unigram.npy", cnt); print(f"unigram done t={time.time()-t0:.0f}s", flush=True)
cnt = np.load("stats/unigram.npy"); p = (cnt + 0.5) / (cnt.sum() + 0.5 * V)

def rsvd(M, k, p_over=24, iters=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = M @ rng.standard_normal((M.shape[1], k + p_over)).astype(np.float32)
    for _ in range(iters):
        Y, _ = np.linalg.qr(Y); Z, _ = np.linalg.qr(M.T @ Y); Y = M @ Z
    Q, _ = np.linalg.qr(Y)
    Ub, s, Vt = np.linalg.svd((M.T @ Q).T, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000_000; KAPPA = 4.0
if not os.path.exists("stats/bigram_svd.npz"):
    keys, counts = [], []
    for i in range(0, NB, 25_000_000):
        a = np.asarray(tok[i:i + 25_000_001], dtype=np.int64)
        k, c = np.unique(a[:-1] * V + a[1:], return_counts=True); keys.append(k); counts.append(c)
        print(f"bigram chunk {i//25_000_000} uniq={len(k)} t={time.time()-t0:.0f}s", flush=True)
    k = np.concatenate(keys); c = np.concatenate(counts)
    k, inv = np.unique(k, return_inverse=True); c = np.bincount(inv, weights=c).astype(np.float64)
    r, w = k // V, k % V
    # smoothed bigram:  p(w|v) ~ (c(v,w) + KAPPA p(w)) / (c(v) + KAPPA)  => S = log(1 + c(v,w) / (KAPPA p(w)))
    vals = np.log1p(c / (KAPPA * p[w])).astype(np.float32)
    S = sp.csr_matrix((vals, (r, w)), shape=(V, V))
    print(f"bigram matrix nnz={S.nnz} t={time.time()-t0:.0f}s", flush=True)
    # row-weighted factorization: R spans the top right-singular space of diag(sqrt(c(v))) S, so frequent contexts
    # (which dominate the loss) are fit best; every row (rare ones too) is then folded in by projection L = S R.
    cv = np.asarray(S.getnnz(axis=1) * 0 + np.bincount(r, weights=c, minlength=V), dtype=np.float64)
    Sw = sp.diags(np.sqrt(cv).astype(np.float32)) @ S
    _, s, Vt = rsvd(Sw.tocsr(), 128)
    R = Vt.T.astype(np.float32); Lf = np.asarray(S @ R, dtype=np.float32)
    tot = float((cv[r] * vals.astype(np.float64) ** 2).sum()); cap = float((cv[:, None] * Lf.astype(np.float64) ** 2).sum())
    np.savez("stats/bigram_svd.npz", L=Lf, R=R, s=s.astype(np.float32), ctx_count=cv.astype(np.float32), kappa=KAPPA, n_tokens=NB)
    print(f"bigram svd done weighted_energy_captured={cap/tot:.3f} top_s={s[:4]} t={time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
