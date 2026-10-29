"""Token-level binning -> class-based bigram model (Brown et al. 1992 style), all closed form:
  1. bin tokens into C classes: weighted spherical k-means on distributional features (bigram SVD factors)
     or LLM token classes (labels file)
  2. class transition table  M[a,b] = log p(class b | class a)   and within-class unigram  log p(w | class(w))
  3. held-out cross-entropy of  p(w|v) = p(c_w | c_v) p(w | c_w)  on the val stream
Writes stats/class_bigram_<name>.npz (cls, M, logp_in_class).  usage: class_bigram.py <C> <name> [classes.npy]"""
import numpy as np, sys, time
C = int(sys.argv[1]); name = sys.argv[2]; V = 50257; t0 = time.time()
cnt = np.load("stats/unigram.npy").astype(np.float64)
bc = np.load("stats/bigram_counts.npz"); r, w, c = bc["r"].astype(np.int64), bc["w"].astype(np.int64), bc["c"].astype(np.float64)
if len(sys.argv) > 3:
    cls = np.load(sys.argv[3]).astype(np.int64)
else:
    z = np.load("stats/bigram_svd.npz"); F = np.concatenate([z["L"][:, :64], z["R"][:, :64] * np.sqrt(z["s"][:64])[None]], 1).astype(np.float64)
    F[:, :64] /= np.linalg.norm(F[:, :64], axis=1, keepdims=True) + 1e-9; F[:, 64:] /= np.linalg.norm(F[:, 64:], axis=1, keepdims=True) + 1e-9
    wt = np.sqrt(cnt + 1.0); top = np.argsort(-cnt)[:30000]
    rng = np.random.default_rng(0); cen = F[top[rng.choice(len(top), C, replace=False, p=wt[top] / wt[top].sum())]]
    for it in range(25):
        asg = (F[top] @ cen.T).argmax(1)
        for k in range(C):
            mk = asg == k
            cen[k] = (F[top][mk] * wt[top][mk, None]).sum(0) if mk.any() else F[top[rng.integers(len(top))]]
        cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9
    cls = (F @ cen.T).argmax(1)
    cls[cnt == 0] = cls[np.argmax(cnt)]   # unseen tokens -> class of the most frequent token (irrelevant)
N = np.zeros((C, C)); np.add.at(N, (cls[r], cls[w]), c)
alpha = 0.1
M = np.log((N + alpha) / (N.sum(1, keepdims=True) + C * alpha))
ccount = np.bincount(cls, weights=cnt + 0.5, minlength=C)
logp_in = np.log((cnt + 0.5) / ccount[cls])
# held-out evaluation on the val stream
val = np.fromfile("data_bin/val.u16", dtype=np.uint16)[:4_000_000].astype(np.int64)
v_, w_ = val[:-1], val[1:]
ce_cls = -(M[cls[v_], cls[w_]] + logp_in[w_]).mean()
p_uni = (cnt + 0.5) / (cnt.sum() + 0.5 * V); ce_uni = -np.log(p_uni[w_]).mean()
sizes = np.bincount(cls, minlength=C)
print(f"class bigram C={C} ({name}): held-out CE={ce_cls:.4f}  unigram CE={ce_uni:.4f}  gain={ce_uni - ce_cls:+.4f}  class sizes min/med/max={sizes.min()}/{int(np.median(sizes))}/{sizes.max()} t={time.time()-t0:.0f}s")
np.savez(f"stats/class_bigram_{name}.npz", cls=cls, M=M.astype(np.float32), logp_in_class=logp_in.astype(np.float32), ce=ce_cls, ce_uni=ce_uni)
