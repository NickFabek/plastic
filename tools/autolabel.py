"""Name every MLP neuron of a trained model after its best-matching LLM abstraction, Goodfire-microscope style.
Inputs: passage dump (mean activation per probe passage) and token dump (per-token activations). For each neuron:
  topic   = fine Jev topic with the highest activation lift (mean over its passages / mean over all), with the lift
  format  = same for the 40 formats
  tok     = most common LLM token class among its 20 top-activating tokens, with that purity
  next    = most common LLM class of the NEXT token among those 20 positions, with that purity
usage: autolabel.py dump.bin tokdump.bin out.json"""
import numpy as np, json, sys
dp, tp, out = sys.argv[1:4]
with open(dp, "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=8); n, L, m = int(h[2]), int(h[3]), int(h[4])
    H = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)
with open(tp, "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=8); nw, T, L2, m2 = (int(x) for x in h[2:6])
    tok = np.fromfile(f, dtype=np.int32, count=nw * T)
    A = [np.fromfile(f, dtype=np.float16, count=nw * T * m).reshape(nw * T, m).astype(np.float32) for _ in range(L)]
z = np.load("labels/jev_probe.npz"); fine = z["topic"].argmax(1); fmt = z["fmt"].argmax(1)
val = np.fromfile("data_bin/val.u16", dtype=np.uint16).astype(np.int64)
nxt = np.concatenate([val[w * T + 1:(w + 1) * T + 1] for w in range(nw)])
llm = np.load("labels/token_classes_jev_full.npy"); cur_c, nxt_c = llm[tok], llm[nxt]
def lift(X, lab, K):
    Y = np.zeros((len(lab), K)); Y[np.arange(len(lab)), lab] = 1; cnt = Y.sum(0)
    mean_c = (Y.T @ X) / np.maximum(cnt, 1)[:, None]; mu = X.mean(0) + 1e-9
    L_ = mean_c / mu; L_[cnt < 8] = 0; return L_.argmax(0), L_.max(0)
res = []
for l in range(L):
    X = H[:, l, :]; t_i, t_l = lift(X, fine, 160); f_i, f_l = lift(X, fmt, 40)
    top = np.argpartition(-A[l], 20, axis=0)[:20]
    def mode(lab):
        c = lab[top]; o = np.zeros(m, int); p = np.zeros(m)
        for j in range(m): u, k = np.unique(c[:, j], return_counts=True); o[j] = u[k.argmax()]; p[j] = k.max() / 20
        return o, p
    tc, tp_ = mode(cur_c); nc, np_ = mode(nxt_c)
    res.append({"topic": t_i.tolist(), "topic_lift": np.round(t_l, 2).tolist(), "format": f_i.tolist(), "format_lift": np.round(f_l, 2).tolist(),
                "tok": tc.tolist(), "tok_purity": np.round(tp_, 2).tolist(), "next": nc.tolist(), "next_purity": np.round(np_, 2).tolist()})
json.dump(res, open(out, "w"), separators=(",", ":"))
print("wrote", out, [f"L{l}: median topic lift {np.median(r['topic_lift']):.2f}, median tok purity {np.median(r['tok_purity']):.2f}" for l, r in enumerate(res)])
