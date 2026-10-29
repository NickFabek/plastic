"""Pick the handful of global scale knobs of a construction by coordinate search on held-out loss at step 0
(no weights are trained: each trial = closed-form construction + one eval pass on 256k val tokens).
usage: calibrate.py <name> <base_init.bin> <mode> <labels> [model args...]  -> runs/calib_<name>/best.json"""
import json, os, subprocess, sys, itertools, re
name, base, mode, labels = sys.argv[1:5]
model_args = [x for x in sys.argv[5:] if not x.startswith("--")]
cons_args = sum([x.split("=", 1) for x in sys.argv[5:] if x.startswith("--")], [])   # --key=value -> construct.py
rd = f"runs/calib_{name}"; os.makedirs(rd, exist_ok=True); log = open(f"{rd}/log.txt", "a")
grid = {"k_big": [0.25, 0.5, 1.0, 1.5, 2.0], "big_frac": [0.2, 0.35, 0.5, 0.65],
        "g_in": [1, 2, 4, 8, 16, 32], "theta": [-2, -1, -0.5, 0, 0.5, 1], "prior_w": [0, 0.5, 1, 2],
        "k_out": [0.125, 0.25, 0.5, 1, 2, 4], "ev_frac": [0.2, 0.35, 0.5], "ctx_gain": [0.25, 0.5, 1, 2, 4, 8],
        "out_res": [0.01, 0.03, 0.1, 0.3, 1.0]}
keys = {"stat": ["k_big", "big_frac"], "both": ["k_big", "big_frac", "k_out", "g_in", "theta", "prior_w", "ev_frac", "ctx_gain"],
        "concept": ["k_out", "g_in", "theta", "prior_w", "ev_frac", "ctx_gain"],
        "bothc": ["out_res", "ctx_gain", "k_out", "g_in", "theta", "prior_w"]}[mode]   # bothc: bigram block fixed, calibrate concept circuit
cmode = "both" if mode == "bothc" else mode
init_over = json.loads(os.environ.get("CALIB_INIT", "{}"))
cur = {"k_big": 1.0, "big_frac": 0.35, "g_in": 4, "theta": 0, "prior_w": 1, "k_out": 1, "ev_frac": 0.35, "ctx_gain": 4, "out_res": 1.0}
cur.update(init_over); cache = {}
def evaluate(p):
    key = json.dumps(p, sort_keys=True)
    if key in cache: return cache[key]
    out = f"{rd}/trial.bin"
    cmd = [sys.executable, "tools/construct.py", "--base", base, "--out", out, "--mode", cmode, "--labels", labels] + cons_args + sum([[f"--{k}", str(v)] for k, v in p.items()], [])
    subprocess.run(cmd, check=True, capture_output=True)
    r = subprocess.run(["./build/train.exe", "mode=eval", f"init_from={out}", "eval_tokens=262144", "B=32"] + model_args, capture_output=True, text=True)
    mt = re.search(r"DONE val_loss=([\d.]+)", r.stdout)
    if not mt: print("ERROR", r.stdout[-500:], flush=True); sys.exit(1)
    v = float(mt.group(1)); cache[key] = v; log.write(f"{v:.4f} {key}\n"); log.flush(); return v
best = evaluate(cur); print(f"start {best:.4f} {cur}", flush=True)
for sweep in range(2):
    for k in keys:
        for val in grid[k]:
            trial = dict(cur); trial[k] = val; v = evaluate(trial)
            if v < best - 1e-4: best, cur = v, trial
        print(f"pass{sweep} {k}={cur[k]} -> {best:.4f}", flush=True)
json.dump({"val_loss": best, "params": cur, "mode": mode, "labels": labels}, open(f"{rd}/best.json", "w"), indent=1)
print(f"DONE {name} best={best:.4f} {cur}", flush=True)
