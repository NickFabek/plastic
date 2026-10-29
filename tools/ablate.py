"""Does the trained network still USE its written neurons? Zero the output weights (W2 columns) of a neuron set in
layer 0 and measure the validation-loss increase; compare the 200 written concept neurons with 200 random other
neurons of the same layer (3 draws).   usage: ablate.py ckpt.bin [K=200] [layer=0] [extra train.exe args...]"""
import sys, os, re, subprocess, json, numpy as np
sys.path.insert(0, "tools"); import ckpt
path = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 200; layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
extra = sys.argv[4:]
cfg, W = ckpt.read(path); m = cfg["m"]
def ev(idx, tag):
    W2 = W[f"w2_{layer}"].copy(); W2[:, idx] = 0; Wm = dict(W); Wm[f"w2_{layer}"] = W2
    tmp = f"runs/_ablate_{os.getpid()}.bin"; ckpt.write(tmp, cfg, Wm)
    r = subprocess.run(["./build/train_dev.exe", "mode=eval", f"d={cfg['d']}", f"L={cfg['L']}", f"m={m}", f"init_from={tmp}", "eval_tokens=524288", "B=16"] + extra, capture_output=True, text=True)
    os.remove(tmp); return float(re.search(r"DONE val_loss=([\d.]+)", r.stdout).group(1))
base = ev(np.array([], dtype=int), "none")
written = ev(np.arange(K), "written")
rng = np.random.default_rng(0); rand = [ev(rng.choice(np.arange(K, m), K, replace=False), "random") for _ in range(3)]
res = {"ckpt": path, "base": base, "ablate_written": written, "ablate_random": rand,
       "d_written": written - base, "d_random": float(np.mean(rand)) - base}
print(json.dumps(res))
