"""Collect everything the results page needs into web/data/results.json (curves, finals, multipliers, step-0
losses, oracle ceilings, H1 neuron analyses, taxonomy names, API spend)."""
import json, os, glob, numpy as np
R = {}
def met(n):
    p = f"runs/{n}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None
# ---- curves + finals for every run of interest
groups = {
    "h2": ["h2_base", "h2_uni", "h2_stat", "h2_stat_llm", "h2_stat_rnd", "h2_c_jev", "h2_c_kmeans", "h2_c_shuf", "h2_c_coarse"],
    "h4": ["h2_stat_llm", "h4_ind", "h4_ind_w10", "h4_ind_w10_crit", "h4_ind_ema", "early_ind"],
    "h3": ["h2_c_jev", "h3_w1", "h3_w10", "h3_w100", "h3_w10_crit", "h3_w10_ema", "h3_w10_in", "h3_w10_out",
           "h3_f01", "h3_f1", "h3_f1_crit", "h3_f1_rand", "h3_w10_rand", "h2_base"],
    "early": ["early_base", "early_uni", "early_stat", "early_stat_llm", "early_c_jev", "early_c_kmeans", "early_c_shuf", "early_c_coarse"],
    "budget": ["bud_base_12M", "bud_base_25M", "h2_base", "bud_base_100M"],
    "width": ["h1_m256", "h2_base", "h1_m4096"],
    "scale": ["m_base", "m_ind", "m_c_jev"],
    "h3x": ["h2_c_jev_live", "h3_f03_live", "h3_f01", "h3_f03", "h3_f03_t01", "h3_f1_rand_L3", "h3_f03_rand_L3", "h3_f03_rand_L3_t01", "h3_mix_only", "h2_c_jev_nn", "h2_stat_rnd"],
    "seeds": sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("runs/seed_*/metrics.json")),
    "lr": sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("runs/lr_s*/metrics.json")),
}
runs = {}
for g, names in groups.items():
    for n in names:
        m = met(n)
        if not m: continue
        c = m["curve"]
        runs[n] = {"curve": [[p["tokens"], round(p["val_loss"], 4)] + ([round(p.get("fid0", 0), 4), round(p.get("fid1", 0), 4)] if "fid0" in p else [])
                             for p in c],
                   "final": round(m["final_val_loss"], 4), "tokens": c[-1]["tokens"], "tok_per_s": round(m.get("tok_per_s", 0)),
                   "config": {k: v for k, v in m["config"].items() if k not in ("probe", "run_dir")},
                   "anchor_dev": c[-1].get("anchor_dev"), "concept_ce": c[-1].get("concept_ce")}
R["groups"] = groups; R["runs"] = runs
# ---- budget-curve multipliers (baselines trained to completion at each budget)
bud = sorted((runs[n]["tokens"], runs[n]["final"]) for n in groups["budget"] if n in runs and runs[n]["tokens"] > 1e6 and abs(runs[n]["tokens"] - float(runs[n]["config"].get("tokens", 0))) < 0.01 * float(runs[n]["config"].get("tokens", 1)))
if len(bud) >= 2:
    BT, BL = np.log([b[0] for b in bud]), np.array([b[1] for b in bud])
    def eff_tokens(L):
        i = int(np.clip(np.searchsorted(-BL, -L), 1, len(BL) - 1))
        return float(np.exp(BT[i - 1] + (BT[i] - BT[i - 1]) * (BL[i - 1] - L) / (BL[i - 1] - BL[i])))
    R["budget_curve"] = [[int(t), l] for t, l in bud]
    for n, r in runs.items():
        if r["tokens"] >= 4.9e7: r["mult"] = round(eff_tokens(r["final"]) / r["tokens"], 3)
# ---- step-0 losses of the inits (calibration logs)
R["step0"] = {"random": 10.8249, "unigram": 7.4256, "svd_bigram": 7.1161, "class_bigram_km64": 6.8103, "class_bigram_llm64": 6.8994}
for lab in ("jev", "kmeans", "shuf", "coarse"):
    p = f"runs/calib2_{lab}/best.json"
    if os.path.exists(p): R["step0"][f"concepts_{lab}"] = round(json.load(open(p))["val_loss"], 4)
R["oracle"] = {"unigram_ce": 7.542, "topic160": 0.370, "format40": 0.268, "coarse16": 0.246, "kmeans200": 0.375, "shuffled": -0.067}
R["token_bins"] = {"unigram": 7.5246, "full_bigram": 5.3698, "km32": 6.9556, "km64": 6.8580, "km128": 6.7393, "llm64": 6.9495}
# ---- H1 analyses
def h1(path):
    if not os.path.exists(path): return None
    j = json.load(open(path)); out = []
    for e in j["layers"]:
        row = {"layer": e["layer"]}
        for u in ("neuron", "random_dir"):
            row[u] = {k: round(v, 4) for k, v in e[u].items()}
        out.append(row)
    return out
R["h1"] = {"base": h1("runs/h2_base/h1.json"), "m256": h1("runs/h1_m256/h1.json"), "m4096": h1("runs/h1_m4096/h1.json"),
           "growth": {s: h1(f"runs/h1_growth/h1_{s}.json") for s in (500, 1000, 1500, 2000, 2500, 3000)}}
def h1tok(path):
    if not os.path.exists(path): return None
    return [{"layer": e["layer"], "neuron": {k: round(v, 4) for k, v in e["neuron"].items()}, "random_dir": {k: round(v, 4) for k, v in e["random_dir"].items()}} for e in json.load(open(path))["layers"]]
R["h1"]["base_tok"] = h1tok("runs/h2_base/h1_tok.json"); R["h1"]["m256_tok"] = h1tok("runs/h1_m256/h1_tok.json"); R["h1"]["m4096_tok"] = h1tok("runs/h1_m4096/h1_tok.json")
R["ablate"] = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in glob.glob("runs/*/ablate.json")}
for k in list(R["h1"].keys()):
    if k != "growth" and R["h1"][k] is None: del R["h1"][k]
for k, v in R.get("h1x", {}).items(): pass
# extra H1-style analyses of constructed/righting-force models, if present
R["h1_models"] = {n: h1(f"runs/{n}/h1.json") for n in ("h2_c_jev", "h3_f1", "h3_w10", "h3_w100") if os.path.exists(f"runs/{n}/h1.json")}
# ---- names
tax = json.load(open("labels/taxonomy_raw.json", encoding="utf-8"))
R["taxonomy"] = {"topic_groups": [{"group": g["group"], "concepts": [c["id"] for c in g["concepts"]]} for g in tax["topic"]],
                 "formats": [c["id"] for c in tax["format"]]}
R["token_classes"] = [c["id"] for c in json.load(open("labels/token_classes.json", encoding="utf-8"))]
R["spend"] = json.load(open("api_spend.json"))
R["written"] = json.load(open("runs/written_stats.json")) if os.path.exists("runs/written_stats.json") else {}
os.makedirs("web/data", exist_ok=True)
json.dump(R, open("web/data/results.json", "w"), separators=(",", ":"))
print("wrote web/data/results.json", len(runs), "runs", os.path.getsize("web/data/results.json") // 1024, "KB")
