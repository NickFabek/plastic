"""Joint-grid calibration of the concept circuit (same procedure for every bin source, for a fair comparison).
Stage 1: grid g_in x k_out;  stage 2: grid theta x prior_w;  stage 3: local g_in x k_out refinement.
Bigram block fixed (class bigram, calibrated), ctx_gain=1, out_res=1.
usage: calibrate2.py <name> <labels> [model args...]  -> runs/calib2_<name>/best.json"""
import json, os, subprocess, sys, re, itertools
name, labels = sys.argv[1:3]; model_args = sys.argv[3:]
rd = f"runs/calib2_{name}"; os.makedirs(rd, exist_ok=True); log = open(f"{rd}/log.txt", "a")
fixed = {"k_big": 1.0, "big_frac": 0.35, "ctx_gain": 1, "out_res": 1.0, "ev_frac": 0.35}
cache = {}
def ev(p):
    q = dict(fixed); q.update(p); key = json.dumps(q, sort_keys=True)
    if key in cache: return cache[key]
    out = f"{rd}/trial.bin"
    subprocess.run([sys.executable, "tools/construct.py", "--base", "runs/init_s/init.bin", "--out", out, "--mode", "both", "--labels", labels,
                    "--big_kind", "cls", "--cls_name", "km64"] + sum([[f"--{k}", str(v)] for k, v in q.items()], []), check=True, capture_output=True)
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
