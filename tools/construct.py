"""Closed-form construction ("binning -> neurons", no gradient descent) of a PLASTIC GPT init.

Starts from the trainer's random init (same seed as the baseline, so every non-constructed weight is identical) and
writes data statistics into dedicated residual subspaces:
  S_big  bigram block: E[v] gets L_v, lm_head[w] gets R_w   (low-rank smoothed log p(w|v)/p(w)) + lm_bias = log p(w)
  S_ev   concept evidence: E[v] gets A_v, rows of the row-centred token/concept PMI   PMI ~ A B^T
  S_ctx  layer-0 head 0 has q=k=0 (uniform doc-causal attention); its value reads S_ev, its output writes S_ctx
  S_out  layer-0 MLP neuron c (one per concept) reads B_c from S_ctx, writes Q_c into S_out; lm_head[w] gets P_w
         (PMI ~ P Q^T is the concept's next-token effect)
Free dims carry a random padding that equalises every embedding row's norm (so RMSNorm scales are constant).

usage: construct.py --base init.bin --out out.bin --mode {stat,concept,both} --labels {jev,shuf,kmeans,coarse} [scales]"""
import argparse, json, os, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__)); import ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="both", choices=["stat", "concept", "both", "unigram"])
ap.add_argument("--labels", default="jev", help="jev | shuf | kmeans | coarse | file.npz")
ap.add_argument("--r_big", type=int, default=64); ap.add_argument("--r_ev", type=int, default=48); ap.add_argument("--r_out", type=int, default=64)
ap.add_argument("--k_big", type=float, default=1.0, help="bigram logit scale")
ap.add_argument("--big_frac", type=float, default=0.35, help="share of embedding norm^2 budget for the bigram block (95th pct)")
ap.add_argument("--ev_frac", type=float, default=0.35, help="share of embedding norm^2 budget for evidence (95th pct)")
ap.add_argument("--g_in", type=float, default=4.0, help="concept-neuron input gain (on mean-PMI units)")
ap.add_argument("--theta", type=float, default=0.0, help="concept-neuron bias offset")
ap.add_argument("--prior_w", type=float, default=1.0, help="weight of log prior in neuron bias")
ap.add_argument("--k_out", type=float, default=1.0, help="concept output logit scale")
ap.add_argument("--ctx_gain", type=float, default=4.0, help="scale of the averaged evidence written to S_ctx")
ap.add_argument("--out_res", type=float, default=1.0, help="residual magnitude of concept outputs (lm_head compensates)")
ap.add_argument("--const_frac", type=float, default=0.0, help="constant 'massive' dim with norm^2 = const_frac*d in every embedding;"
                " keeps RMSNorm's denominator ~constant so constructed paths decouple")
ap.add_argument("--beta", type=float, default=1.0, help="PMI shrinkage pseudo-count, in units of each concept's token mass")
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--big_kind", default="svd", choices=["svd", "cls"], help="bigram block: low-rank SVD or class-based bigram")
ap.add_argument("--cls_name", default="km64", help="stats/class_bigram_<name>.npz for big_kind=cls")
ap.add_argument("--anchor_out", default="", help="also write an anchor-strength ckpt (1 on constructed weights)")
ap.add_argument("--meta_out", default="")
# copy (induction) circuit: a "pattern" construction instead of an abstraction one
ap.add_argument("--bias_mode", default="prior", choices=["prior", "quantile"],
                help="prior: b = theta + prior_w*log p(c); quantile: b set so neuron c fires on ~fire_mult*p(c) of positions")
ap.add_argument("--fire_mult", type=float, default=1.0)
ap.add_argument("--no_neurons", type=int, default=0, help="write the evidence embedding + averaging head but not the concept neurons")
ap.add_argument("--induction", type=int, default=0)
ap.add_argument("--ind_rid", type=int, default=64, help="token identity code dims (output path)")
ap.add_argument("--ind_nlo", type=int, default=10, help="low-frequency RoPE pairs used for matching (2*nlo dims)")
ap.add_argument("--ind_nhi", type=int, default=4, help="high-frequency RoPE pairs used by the previous-token head")
ap.add_argument("--ind_prev", type=float, default=40.0, help="previous-token head peak score")
ap.add_argument("--ind_beta", type=float, default=20.0, help="induction head score for an exact match")
ap.add_argument("--ind_gamma", type=float, default=2.0, help="logit boost for the copied token")
ap.add_argument("--id_frac", type=float, default=0.3, help="embedding norm^2 share of the identity code")
a = ap.parse_args()

cfg, W = ckpt.read(a.base); V, Vp, d, m, H = cfg["V"], cfg["Vp"], cfg["d"], cfg["m"], cfg["H"]; hd = d // H
rng = np.random.default_rng(12345)
cnt = np.load("stats/unigram.npy").astype(np.float64); p_uni = (cnt + 0.5) / (cnt.sum() + 0.5 * V)
logp = np.log(p_uni)
# residual subspaces (only the blocks this mode uses; the rest of the stream is free/random)
use_big, use_con = a.mode in ("stat", "both"), a.mode in ("concept", "both")
if a.big_kind == "cls": a.r_big = int(np.load(f"stats/class_bigram_{a.cls_name}.npz")["M"].shape[0])
o = 0; S_big = slice(o, o + a.r_big * use_big); o += a.r_big * use_big
S_ev = slice(o, o + a.r_ev * use_con); o += a.r_ev * use_con
S_ctx = slice(o, o + a.r_ev * use_con); o += a.r_ev * use_con
S_out = slice(o, o + a.r_out * use_con); o += a.r_out * use_con
if a.induction:
    if a.const_frac <= 0: a.const_frac = 1.0
    c_dim = o; o += 1
    r_m = 2 * a.ind_nlo
    S_id = slice(o, o + a.ind_rid); o += a.ind_rid
    S_prev = slice(o, o + r_m); o += r_m
    S_ind = slice(o, o + a.ind_rid); o += a.ind_rid
free = slice(o, d); n_free = d - o
assert n_free >= 8, f"not enough residual dims ({o} used of {d})"
assert a.r_ev <= hd
ANC = {k: np.zeros_like(v) for k, v in W.items()} if a.anchor_out else None
meta = {"S_big": [S_big.start, S_big.stop], "S_ev": [S_ev.start, S_ev.stop], "S_ctx": [S_ctx.start, S_ctx.stop],
        "S_out": [S_out.start, S_out.stop], "free": [free.start, free.stop], "args": vars(a)}

W["lm_bias"][:V] = logp.astype(np.float32); W["lm_bias"][V:] = -30.0
E = np.zeros((Vp, d), dtype=np.float64); Uh = W["lm_head"].astype(np.float64)
budget = float(d)   # every embedding row gets norm^2 = d (rms 1), like the baseline's N(0,1) init
rho = float(np.sqrt(1.0 + a.const_frac))   # rms of the stream once the constant dim is added
used = np.zeros(Vp)

if a.mode in ("stat", "both") and a.big_kind == "cls":
    # token bins -> class-based bigram written into E/U: logit(w|v) = k_big * M[cls v, cls w] + log p(w | cls w)
    z = np.load(f"stats/class_bigram_{a.cls_name}.npz"); cls, M = z["cls"], z["M"].astype(np.float64)
    s_c = np.sqrt(a.big_frac * budget)
    E[np.arange(V), S_big.start + cls[:V]] = s_c; used[:V] += s_c ** 2
    Uh[:V, S_big] = M[:, cls[:V]].T * (a.k_big / s_c) * rho
    W["lm_bias"][:V] = z["logp_in_class"][:V]
    if ANC is not None: ANC["wte"][:V, S_big] = 1; ANC["lm_head"][:V, S_big] = 1
    meta["cls_ce"] = float(z["ce"])
elif a.mode in ("stat", "both"):
    z = np.load("stats/bigram_svd.npz"); L, R = z["L"][:, :a.r_big].astype(np.float64), z["R"][:, :a.r_big].astype(np.float64)
    nrm = np.linalg.norm(L, axis=1); s_e = np.sqrt(a.big_frac * budget) / max(np.percentile(nrm[cnt > 100], 95), 1e-9)
    Lb = L * s_e; over = np.linalg.norm(Lb, axis=1) > np.sqrt(0.9 * budget); Lb[over] *= (np.sqrt(0.9 * budget) / np.linalg.norm(Lb[over], axis=1))[:, None]
    E[:V, S_big] = Lb; used[:V] += (Lb ** 2).sum(1)
    Uh[:V, S_big] = R * (a.k_big / s_e) * rho  # logit = (k_big/s_e) R_w . (s_e L_v) / rms(x), rms ~ rho
    if ANC is not None: ANC["wte"][:V, S_big] = 1; ANC["lm_head"][:V, S_big] = 1
    meta["bigram_scale_e"] = float(s_e)

if a.mode in ("concept", "both"):
    # ---- bins: passage x concept soft labels (two facets: 160 topics + 40 formats, or variants)
    z = np.load("labels/jev_train.npz"); start, end = z["start"], z["end"]
    if a.labels in ("jev", "shuf", "coarse"):
        Y = np.concatenate([z["topic"], z["fmt"]], 1).astype(np.float64); facets = [160, 40]
        if a.labels == "coarse":   # 16 coarse topic groups (sum of fine probabilities) + 40 formats
            G = np.zeros((160, 16)); G[np.arange(160), z["group_of"]] = 1
            Y = np.concatenate([z["topic"] @ G, z["fmt"]], 1); facets = [16, 40]
        if a.labels == "shuf": Y = Y[np.random.default_rng(99).permutation(len(Y))]
    else:
        zz = np.load(a.labels if a.labels.endswith(".npz") else f"labels/{a.labels}.npz"); Y = zz["Y"].astype(np.float64); facets = [int(x) for x in zz["facets"]]
        if "start" in zz: start, end = zz["start"], zz["end"]
    K = Y.shape[1]; assert K <= m
    # ---- passage token counts X (n x V) and concept/token mass C = X^T Y  (V x K)
    tok = np.memmap("data_bin/train.u16", dtype=np.uint16, mode="r")
    rows = np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(zip(start, end))])
    cols = np.concatenate([np.asarray(tok[s:e], dtype=np.int64) for s, e in zip(start, end)])
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(start), V))
    C = np.asarray((X.T @ Y)); n_c = C.sum(0)
    # ---- shrunk PMI: p(v|c) = (C[v,c] + b_c p(v)) / (n_c + b_c), b_c = beta * n_c
    bc = a.beta * n_c
    PMI = np.log((C + bc[None] * p_uni[:, None]) / (n_c + bc)[None]) - logp[:, None]
    prior = n_c / n_c.sum(); off = 0
    for fs in facets:   # centre each token's evidence across concepts within a facet (weighted by concept prior)
        sl = slice(off, off + fs); pw = prior[sl] / prior[sl].sum()
        PMI[:, sl] -= (PMI[:, sl] @ pw)[:, None]; off += fs
    # ---- low-rank factorisation, weighted by token frequency (the loss is a frequency-weighted average)
    wv = np.sqrt(p_uni / p_uni.max())
    Uu, s, Vt = np.linalg.svd(PMI * wv[:, None], full_matrices=False)
    Bf = (Vt.T * np.sqrt(s))            # K x rank   (concept side)
    Af = PMI @ (Vt.T / np.sqrt(s))       # V x rank   (token side, folded in: Af Bf^T = PMI projected)
    padr = lambda X, r: np.pad(X[:, :r], ((0, 0), (0, max(0, r - X.shape[1]))))   # fewer concepts than dims -> zero-pad
    A_ev, B_ev = padr(Af, a.r_ev), padr(Bf, a.r_ev)
    P_out, Q_out = padr(Af, a.r_out), padr(Bf, a.r_out)
    cap = (s[:a.r_ev] ** 2).sum() / (s ** 2).sum()
    # ---- embedding evidence block
    nrm = np.linalg.norm(A_ev, axis=1); s_ev = np.sqrt(a.ev_frac * budget) / max(np.percentile(nrm[cnt > 100], 95), 1e-9)
    Ae = A_ev * s_ev
    room = np.maximum(0.9 * budget - used[:V], 1e-6); over = (Ae ** 2).sum(1) > room
    Ae[over] *= np.sqrt(room[over] / (Ae[over] ** 2).sum(1))[:, None]
    E[:V, S_ev] = Ae; used[:V] += (Ae ** 2).sum(1)
    # ---- layer-0 head 0: uniform attention (q=k=0), value = identity on S_ev, output -> S_ctx (gain ctx_gain)
    l = a.layer; Wqkv, Wo = W[f"wqkv{l}"], W[f"wo{l}"]
    Wqkv[0:hd, :] = 0; Wqkv[d:d + hd, :] = 0; Wqkv[2 * d:2 * d + hd, :] = 0
    for i in range(a.r_ev): Wqkv[2 * d + i, S_ev.start + i] = rho
    Wo[:, 0:hd] = 0
    for i in range(a.r_ev): Wo[S_ctx.start + i, i] = a.ctx_gain
    # ---- concept neurons 0..K-1: a_c = g_in * B_c . ctx/(s_ev*ctx_gain) + theta + prior_w*log p(c); write Q_c to S_out
    W1, b1, W2 = W[f"w1_{l}"], W[f"b1_{l}"], W[f"w2_{l}"]
    if a.no_neurons: W1 = W1.copy(); b1 = b1.copy(); W2 = W2.copy(); Uh_save = Uh[:V, S_out].copy()
    W1[:K, :] = 0; W1[:K, S_ctx] = (a.g_in / (s_ev * a.ctx_gain)) * B_ev * rho
    logprior = np.log(np.maximum(prior, 1e-9)); off = 0
    for fs in facets: sl = slice(off, off + fs); logprior[sl] -= logprior[sl].max(); off += fs
    b1[:K] = a.theta + a.prior_w * logprior
    if a.bias_mode == "quantile":
        # simulate the circuit's pre-bias score on real windows: uniform average of the evidence over the doc prefix
        Aev = Ae / s_ev; sel = np.random.default_rng(3).choice(len(end), min(4000, len(end)), replace=False)
        scores = []
        for i in sel:
            w = np.asarray(tok[end[i] - cfg["T"]:end[i]], dtype=np.int64); ev = Aev[w]
            dstart = np.maximum.accumulate(np.where(w == 50256, np.arange(len(w)), 0))
            cs = np.cumsum(ev, 0); prev = np.where(dstart[:, None] > 0, cs[np.maximum(dstart - 1, 0)], 0) * (dstart[:, None] > 0)
            mean = (cs - prev) / (np.arange(len(w)) - dstart + 1)[:, None]
            scores.append(mean[::8] @ B_ev.T * a.g_in)
        S = np.concatenate(scores)   # positions x K
        off = 0; rate = np.zeros(K)
        for fs in facets:
            sl = slice(off, off + fs); pr = prior[sl] / prior[sl].sum(); rate[sl] = np.clip(a.fire_mult * pr, 0.005, 0.25); off += fs
        b1[:K] = -np.array([np.quantile(S[:, c], 1 - rate[c]) for c in range(K)]) + a.theta
        meta["fire_rate"] = rate.tolist()
    W2[:, :K] = 0; W2[S_out, :K] = Q_out.T * a.out_res
    Uh[:V, S_out] = P_out * (a.k_out / a.out_res) * rho
    if a.no_neurons: Uh[:V, S_out] = Uh_save   # discard the neuron writes (W1/b1/W2 copies are dropped)
    if ANC is not None:
        ANC["wte"][:V, S_ev] = 1; ANC[f"wqkv{l}"][0:hd, :] = 1; ANC[f"wqkv{l}"][d:d + hd, :] = 1; ANC[f"wqkv{l}"][2 * d:2 * d + hd, :] = 1
        ANC[f"wo{l}"][:, 0:hd] = 1; ANC[f"w1_{l}"][:K, :] = 1; ANC[f"b1_{l}"][:K] = 1; ANC[f"w2_{l}"][:, :K] = 1; ANC["lm_head"][:V, S_out] = 1
    meta.update({"K": K, "facets": facets, "ev_scale": float(s_ev), "pmi_rank_energy": float(cap), "labels": a.labels,
                 "logprior": logprior.tolist(), "layer": l})

if a.induction:
    hd_ = hd; h2 = hd // 2; th = 10000.0 ** (-2 * np.arange(h2) / hd_)
    C = np.sqrt(a.const_frac * budget); E[:, c_dim] = C      # constant "massive" dim (outside the norm budget)
    ecode = rng.standard_normal((Vp, a.ind_rid)); ecode /= np.linalg.norm(ecode, axis=1, keepdims=True)
    s_id = np.sqrt(a.id_frac * budget); E[:, S_id] = ecode * s_id; used[:] += s_id ** 2
    e2 = (ecode[:, :r_m] ** 2).sum(1).mean()                 # mean |matching code|^2
    rho1 = np.sqrt(1 + a.const_frac + a.id_frac * r_m / a.ind_rid * 0)   # rms at layer 0 input (approx)
    rho2 = np.sqrt(1 + a.const_frac + a.id_frac * e2)                    # after the prev-token head wrote S_prev
    # ---- layer 0, last head: previous-token head (q, k from the constant dim; RoPE phase puts the peak at t-1)
    hp = H - 1; Wq0 = W["wqkv0"]; Wo0 = W["wo0"]
    Wq0[hp * hd_:(hp + 1) * hd_, :] = 0; Wq0[d + hp * hd_:d + (hp + 1) * hd_, :] = 0; Wq0[2 * d + hp * hd_:2 * d + (hp + 1) * hd_, :] = 0
    a_ = np.sqrt(a.ind_prev * np.sqrt(hd_)) * rho1 / C
    for i in range(a.ind_nhi):
        Wq0[hp * hd_ + i, c_dim] = a_ / np.sqrt(a.ind_nhi)
        Wq0[d + hp * hd_ + i, c_dim] = a_ * np.cos(th[i]) / np.sqrt(a.ind_nhi)
        Wq0[d + hp * hd_ + i + h2, c_dim] = a_ * np.sin(th[i]) / np.sqrt(a.ind_nhi)
    for j in range(r_m): Wq0[2 * d + hp * hd_ + j, S_id.start + j] = rho1
    Wo0[:, hp * hd_:(hp + 1) * hd_] = 0
    for j in range(r_m): Wo0[S_prev.start + j, hp * hd_ + j] = 1.0
    # ---- layer 1, head 0: induction head (query = my token's code, key = the previous token's code at s)
    Wq1 = W["wqkv1"]; Wo1 = W["wo1"]
    Wq1[0:hd_, :] = 0; Wq1[d:d + hd_, :] = 0; Wq1[2 * d:2 * d + hd_, :] = 0
    b_ = np.sqrt(a.ind_beta * np.sqrt(hd_) / (s_id ** 2 * e2)) * rho2
    for k in range(a.ind_nlo):
        i = h2 - a.ind_nlo + k
        Wq1[i, S_id.start + 2 * k] = b_; Wq1[i + h2, S_id.start + 2 * k + 1] = b_
        Wq1[d + i, S_prev.start + 2 * k] = b_; Wq1[d + i + h2, S_prev.start + 2 * k + 1] = b_
    for j in range(a.ind_rid): Wq1[2 * d + j, S_id.start + j] = rho2
    Wo1[:, 0:hd_] = 0
    for j in range(a.ind_rid): Wo1[S_ind.start + j, j] = 1.0
    Uh[:V, S_ind] = ecode[:V] * (a.ind_gamma * rho2 / s_id)
    if ANC is not None:
        ANC["wte"][:, S_id] = 1; ANC["wte"][:, c_dim] = 1; ANC["lm_head"][:V, S_ind] = 1
        for nm, hh in (("wqkv0", hp), ("wqkv1", 0)):
            for blk in range(3): ANC[nm][blk * d + hh * hd_: blk * d + (hh + 1) * hd_, :] = 1
        ANC["wo0"][:, hp * hd_:(hp + 1) * hd_] = 1; ANC["wo1"][:, 0:hd_] = 1
    meta.update({"induction": {"c_dim": c_dim, "S_id": [S_id.start, S_id.stop], "S_prev": [S_prev.start, S_prev.stop], "S_ind": [S_ind.start, S_ind.stop]}})

if a.mode == "unigram":
    E = W["wte"].astype(np.float64)
else:
    # ---- random padding in the free dims (and a little everywhere for identity) to give every row norm^2 = budget
    pad = rng.standard_normal((Vp, n_free)); pad /= np.linalg.norm(pad, axis=1, keepdims=True)
    E[:, free] = pad * np.sqrt(np.maximum(budget - used, 0.05 * budget))[:, None]
    if a.const_frac > 0 and not a.induction:   # last free dim becomes the constant "massive" dim (same value for every token)
        cd = d - 1; E[:, cd] = np.sqrt(a.const_frac * budget); meta["const_dim"] = cd
W["wte"] = E.astype(np.float32); W["lm_head"] = Uh.astype(np.float32)
ckpt.write(a.out, cfg, W)
if ANC is not None: ckpt.write(a.anchor_out, cfg, ANC)
if a.meta_out: json.dump(meta, open(a.meta_out, "w"), indent=1)
print("constructed", a.out, json.dumps({k: meta[k] for k in meta if k not in ("args",)}))
