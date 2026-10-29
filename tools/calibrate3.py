"""Fast joint-grid calibration of the concept circuit: construct once (g_in=1, k_out=1, theta=0), then each trial
only rescales W1[:K] (g_in), b1[:K] (theta + prior_w*log prior) and lm_head[:, S_out] (k_out) in memory.
Same grid / stages as calibrate2.py.  usage: calibrate3.py <name> <labels> [model args...]"""
import json, os, subprocess, sys, re, itertools, numpy as np
sys.path.insert(0, "tools"); import ckpt
name, labels = sys.argv[1:3]; model_args = sys.argv[3:]
rd = f"runs/calib2_{name}"; os.makedirs(rd, exist_ok=True); log = open(f"{rd}/log.txt", "a")
fixed = {"k_big": 1.0, "big_frac": 0.35, "ctx_gain": 1, "out_res": 1.0, "ev_frac": 0.35}
base_ck, meta_p = f"{rd}/base.bin", f"{rd}/base_meta.json"
subprocess.run([sys.executable, "tools/construct.py", "--base", os.environ.get("CALIB_BASE", "runs/init_s/init.bin"), "--out", base_ck, "--mode", "both", "--labels", labels,
                "--big_kind", "cls", "--cls_name", "km64", "--g_in", "1", "--k_out", "1", "--theta", "0", "--prior_w", "1", "--meta_out", meta_p]
               + sum([[f"--{k}", str(v)] for k, v in fixed.items()], []), check=True, capture_output=True)
cfg, W0 = ckpt.read(base_ck); meta = json.load(open(meta_p)); K, l = meta["K"], meta["layer"]; so = slice(*meta["S_out"])
logprior = np.array(meta["logprior"], dtype=np.float32)
W1k, Uo = W0[f"w1_{l}"][:K].copy(), W0["lm_head"][:, so].copy()
cache = {}
def ev(p):
    key = json.dumps(p, sort_keys=True)
    if key in cache: return cache[key]
    W = dict(W0); W[f"w1_{l}"] = W0[f"w1_{l}"].copy(); W[f"w1_{l}"][:K] = W1k * p["g_in"]
    W[f"b1_{l}"] = W0[f"b1_{l}"].copy(); W[f"b1_{l}"][:K] = p["theta"] + p["prior_w"] * logprior
    W["lm_head"] = W0["lm_head"].copy(); W["lm_head"][:, so] = Uo * p["k_out"]
    out = f"{rd}/trial.bin"; ckpt.write(out, cfg, W)
    r = subprocess.run(["./build/train.exe", "mode=eval", f"init_from={out}", "eval_tokens=262144", "B=16"] + model_args, capture_output=True, text=True)
    m = re.search(r"DONE val_loss=([\d.]+)", r.stdout)
    if not m: print("ERROR", r.stdout[-400:], flush=True); sys.exit(1)
    cache[key] = float(m.group(1)); log.write(f"{cache[key]:.4f} {key}\n"); log.flush(); return cache[key]
def grid(base, axes):
    best = (1e9, None)
    for vals in itertools.product(*axes.values()):
        p = dict(base); p.update(dict(zip(axes.keys(), vals))); v = ev(p)
        if v < best[0]: best = (v, p)
    return best
cur = {"g_in": 1, "k_out": 1, "theta": 0, "prior_w": 1}
v, cur = grid(cur, {"g_in": [1, 2, 4], "k_out": [0.25, 0.5, 1]}); print(f"stage1 {v:.4f} {cur}", flush=True)
v, cur = grid(cur, {"theta": [-0.5, 0, 0.5], "prior_w": [0.5, 1, 2]}); print(f"stage2 {v:.4f} {cur}", flush=True)
g, k = cur["g_in"], cur["k_out"]
v, cur = grid(cur, {"g_in": [g / 1.5, g, g * 1.5], "k_out": [k / 1.5, k, k * 1.5]}); print(f"stage3 {v:.4f} {cur}", flush=True)
q = dict(fixed); q.update(cur)
json.dump({"val_loss": v, "params": q, "labels": labels}, open(f"{rd}/best.json", "w"), indent=1)
print(f"DONE {name} best={v:.4f} {q}", flush=True)
