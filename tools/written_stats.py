"""For runs whose layer-0 neurons 0..199 are written Jev concept neurons: how many are alive, how well they detect
their concept, and how much the trained network relies on them (ablation vs 200 random other layer-0 neurons).
Creates missing dumps / ablations.   usage: written_stats.py run_or_ckpt ...  -> runs/written_stats.json"""
import json, os, subprocess, sys, numpy as np
z = np.load("labels/jev_probe.npz"); fine = z["topic"].argmax(1); fmt = z["fmt"].argmax(1); K = 200
out = json.load(open("runs/written_stats.json")) if os.path.exists("runs/written_stats.json") else {}
for name in sys.argv[1:]:
    ck = name if name.endswith(".bin") else f"runs/{name}/final.bin"; key = name.replace("runs/", "").replace("/", "_").replace(".bin", "")
    d = ck.replace(".bin", "_dump.bin") if name.endswith(".bin") else f"runs/{name}/dump.bin"
    if not os.path.exists(d):
        subprocess.run(["./build/train_dev.exe", "mode=dump", "d=256", "L=4", "B=16", f"init_from={ck}", "probe=labels/probe.bin", f"out={d}"], check=True, capture_output=True)
    with open(d, "rb") as f:
        h = np.fromfile(f, dtype=np.int32, count=8); n, L, m = int(h[2]), int(h[3]), int(h[4])
        Hh = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)[:, 0, :]
        Ha = np.fromfile(f, dtype=np.float32, count=n * L * m).reshape(n, L, m)[:, 0, :]
    live_w = int((Hh[:, :K].max(0) > 0).sum()); live_l = float((Hh[:, K:].max(0) > 0).mean())
    fid_t = float((Ha[:, :160].argmax(1) == fine).mean()); fid_f = float((Ha[:, 160:K].argmax(1) == fmt).mean())
    ab_p = f"runs/{name}/ablate.json" if not name.endswith(".bin") else ck.replace(".bin", "_ablate.json")
    if not os.path.exists(ab_p):
        r = subprocess.run([sys.executable, "tools/ablate.py", ck], capture_output=True, text=True, check=True); open(ab_p, "w").write(r.stdout)
    ab = json.loads(open(ab_p).read())
    out[key] = {"live_written": live_w, "live_learned_frac": round(live_l, 3), "fid_topic": round(fid_t, 4), "fid_format": round(fid_f, 4),
                "ablate_written": round(ab["d_written"], 4), "ablate_random": round(ab["d_random"], 4)}
    print(key, out[key], flush=True)
json.dump(out, open("runs/written_stats.json", "w"), indent=1)
