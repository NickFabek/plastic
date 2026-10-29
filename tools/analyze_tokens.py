"""Token-level H1: purity of each unit's top-20 activating token positions with respect to (a) the LLM-named class
of the current token, (b) the distributional (k-means) class of the current token, (c) the LLM class of the NEXT
token, (d) exact token identity. Neurons vs. unit-norm random directions in the same post-activation space, each
against a label-permutation null.   usage: analyze_tokens.py tokdump.bin out.json"""
import numpy as np, json, sys
p, out = sys.argv[1], sys.argv[2]
with open(p, "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=8); n, T, L, m = (int(x) for x in h[2:6])
    tok = np.fromfile(f, dtype=np.int32, count=n * T)
    acts = [np.fromfile(f, dtype=np.float16, count=n * T * m).reshape(n * T, m).astype(np.float32) for _ in range(L)]
val = np.fromfile("data_bin/val.u16", dtype=np.uint16).astype(np.int64)
nxt = np.concatenate([val[w * T + 1:(w + 1) * T + 1] for w in range(n)])
llm = np.load("labels/token_classes_jev_full.npy"); km = np.load("stats/class_bigram_km64.npz")["cls"]
labs = {"cur_llm": llm[tok], "cur_km": km[tok], "next_llm": llm[nxt], "token_id": tok.astype(np.int64)}
rng = np.random.default_rng(0); perm = rng.permutation(len(tok)); K = 20
def purity(U, lab):
    top = np.argpartition(-U, K, axis=0)[:K]; l = lab[top]
    return np.array([np.unique(l[:, j], return_counts=True)[1].max() / K for j in range(U.shape[1])])
res = {"n_tokens": int(len(tok)), "layers": []}
for li, X in enumerate(acts):
    live = (X.max(0) > 0) & (X.std(0) > 1e-6)
    Rd = rng.standard_normal((m, 1024)); Rd /= np.linalg.norm(Rd, axis=0, keepdims=True)
    e = {"layer": li, "live": int(live.sum())}
    for uname, U in (("neuron", X[:, live]), ("random_dir", X @ Rd)):
        d = {}
        for ln, lab in labs.items():
            pu, p0 = purity(U, lab), purity(U, lab[perm])
            d[ln] = float(pu.mean()); d[ln + "_null"] = float(p0.mean()); d[ln + "_aligned"] = float((pu > np.percentile(p0, 99)).mean())
        e[uname] = d
    res["layers"].append(e)
    print(f"L{li} live={e['live']:4d} | " + " | ".join(f"{u[:6]}: " + " ".join(f"{k}={e[u][k]:.2f}(null {e[u][k + '_null']:.2f})" for k in labs) for u in ("neuron", "random_dir")))
json.dump(res, open(out, "w"), indent=0)
