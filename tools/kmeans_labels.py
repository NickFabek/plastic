"""Control bins without any LLM: spherical k-means on TF-IDF bag-of-words of the same 20k passages -> K one-hot bins."""
import numpy as np, scipy.sparse as sp, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 200; out = sys.argv[2] if len(sys.argv) > 2 else "labels/kmeans.npz"
V = 50257; z = np.load("labels/jev_train.npz"); start, end = z["start"], z["end"]
tok = np.memmap("data_bin/train.u16", dtype=np.uint16, mode="r")
rows = np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(zip(start, end))])
cols = np.concatenate([np.asarray(tok[s:e], dtype=np.int64) for s, e in zip(start, end)])
X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(start), V)); X.sum_duplicates()
df = np.asarray((X > 0).sum(0)).ravel(); idf = np.log(len(start) / (1 + df)).astype(np.float32)
X.data = np.log1p(X.data); X = X @ sp.diags(idf); X = sp.csr_matrix(X)
nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9; X = sp.diags(1 / nr) @ X
rng = np.random.default_rng(0); Cn = X[rng.choice(X.shape[0], K, replace=False)].toarray()
for it in range(30):
    sim = np.asarray(X @ Cn.T); asg = sim.argmax(1)
    M = sp.csr_matrix((np.ones(len(asg)), (asg, np.arange(len(asg)))), shape=(K, len(asg)))
    Cn = np.asarray((M @ X).todense()); empty = np.asarray(M.sum(1)).ravel() == 0
    Cn[empty] = X[rng.choice(X.shape[0], int(empty.sum()))].toarray()
    Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-9
Y = np.zeros((len(asg), K), dtype=np.float32); Y[np.arange(len(asg)), asg] = 1
sizes = np.bincount(asg, minlength=K)
np.savez(out, Y=Y, facets=np.array([K]))
print(f"kmeans K={K} mean_sim={sim.max(1).mean():.3f} cluster sizes min/median/max={sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
# agreement with Jev topics: normalized mutual information between k-means bins and Jev argmax topic
jt = z["topic"].argmax(1); P = np.zeros((K, 160)); np.add.at(P, (asg, jt), 1); P /= P.sum()
pa, pb = P.sum(1), P.sum(0); nz = P > 0
mi = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None])[nz])).sum(); ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(); hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
print(f"NMI(kmeans, jev_topic)={2 * mi / (ha + hb):.3f}")
