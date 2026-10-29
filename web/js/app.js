// PLASTIC results page: data-driven charts (hand-rolled SVG) + the in-browser neuron microscope.
import { BPE } from "./bpe.js";
import { loadModel, CPUModel, GPUModel, sample } from "./engine.js";

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f3 = (x) => (x == null || isNaN(x) ? "–" : x.toFixed(3));
const f2 = (x) => (x == null || isNaN(x) ? "–" : x.toFixed(2));
const M = (t) => (t >= 1e6 ? (t / 1e6).toFixed(t >= 1e7 ? 0 : 1) + "M" : t >= 1e3 ? (t / 1e3).toFixed(0) + "k" : String(t));

// Fixed categorical slots (validated order, both themes): color follows the entity everywhere on the page.
const SERIES = {
  h2_base: { name: "random init (baseline)", slot: 1 },
  h2_c_kmeans: { name: "class bigram + k-means concepts", slot: 2 },
  h2_stat: { name: "class bigram (k-means token bins)", slot: 3 },
  h2_stat_llm: { name: "class bigram (LLM token classes)", slot: 4 },
  h2_c_jev: { name: "class bigram + Jev concepts", slot: 5 },
  h2_uni: { name: "unigram bias only", slot: 1, dash: "5 4" },
  h2_c_shuf: { name: "… + shuffled Jev bins", slot: 5, dash: "5 4" },
  h2_c_coarse: { name: "… + coarse Jev bins (16+40)", slot: 5, dash: "1.5 3" },
  h2_stat_rnd: { name: "class bigram (random token classes)", slot: 4, dash: "1.5 3" },
  h4_ind: { name: "LLM-class bigram + copy circuit", slot: 6 },
  h4_ind_w10: { name: "… + weight anchor λ=10", slot: 6, dash: "5 4" },
  h4_ind_w10_crit: { name: "… + anchor, released by 50%", slot: 6, dash: "1.5 3" },
  h4_ind_ema: { name: "… + drifting (EMA) anchor", slot: 6, dash: "8 3 2 3" },
};
const early = (n) => n.replace(/^h2_/, "early_");
function slotColor(slot) { return css(`--s${slot}`); }

// ------------------------------------------------------------------------------------------------ tiny SVG kit
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent && parent.appendChild(e); return e; }
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = Math.pow(10, Math.floor(Math.log10(span / n))), err = (span / n) / step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}
function tip(svg) {
  const g = el("g", { "pointer-events": "none", visibility: "hidden" }, svg);
  const r = el("rect", { rx: 4, fill: css("--plate"), stroke: css("--rule") }, g); const t = el("text", { class: "lbl", "font-size": 11 }, g);
  return { show(x, y, lines, W) {
    t.textContent = ""; lines.forEach((s, i) => { const ts = el("tspan", { x: 8, dy: i ? 14 : 15 }, t); ts.textContent = s; });
    const bb = t.getBBox(); r.setAttribute("width", bb.width + 16); r.setAttribute("height", bb.height + 10);
    const tx = Math.min(x + 12, W - bb.width - 20); g.setAttribute("transform", `translate(${tx},${Math.max(4, y - bb.height - 16)})`); g.setAttribute("visibility", "visible");
  }, hide() { g.setAttribute("visibility", "hidden"); } };
}
// line chart: series [{key, name, color, dash, pts:[[x,y]]}], log-x
const chartW = (host) => Math.max(460, Math.round(host.clientWidth || 640));
function lineChart(host, series, o) {
  host.textContent = ""; const W = chartW(host), H = o.h || 300, P = { l: 52, r: 16, t: 12, b: 36 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.aria || "" }, host);
  const xs = series.flatMap(s => s.pts.map(p => p[0])).filter(x => x > 0), ys = series.flatMap(s => s.pts.map(p => p[1]));
  const x0 = Math.log10(o.xmin || Math.min(...xs)), x1 = Math.log10(o.xmax || Math.max(...xs));
  let y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys); if (y1 - y0 < 1e-6) { y0 -= 0.1; y1 += 0.1; }
  const X = (x) => P.l + (Math.log10(x) - x0) / (x1 - x0) * (W - P.l - P.r), Y = (y) => P.t + (y1 - y) / (y1 - y0) * (H - P.t - P.b);
  for (const v of niceTicks(y0, y1, 5)) { el("line", { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: P.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = o.yfmt ? o.yfmt(v) : v; }
  for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (Math.log10(v) < x0 - 1e-9 || Math.log10(v) > x1 + 1e-9) continue;
    el("line", { x1: X(v), x2: X(v), y1: P.t, y2: H - P.b, class: "grid" }, svg); el("text", { x: X(v), y: H - P.b + 16, "text-anchor": "middle" }, svg).textContent = M(v); }
  if (o.zero) el("line", { x1: P.l, x2: W - P.r, y1: Y(0), y2: Y(0), class: "axis" }, svg);
  el("text", { x: W - P.r, y: H - 4, "text-anchor": "end" }, svg).textContent = o.xlabel || "training tokens";
  el("text", { x: 4, y: P.t + 2, transform: `rotate(-90 4 ${P.t + 2})`, "text-anchor": "end", dy: "0.8em" }, svg).textContent = o.ylabel || "";
  for (const s of series) {
    const pts = s.pts.filter(p => p[0] > 0 && p[1] >= y0 - 1e-9 && p[1] <= y1 + 1e-9);
    if (!pts.length) continue;
    el("path", { d: pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" "), fill: "none", stroke: s.color, "stroke-width": s.w || 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = pts[pts.length - 1]; el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 3, fill: s.color }, svg);
  }
  // crosshair + tooltip
  const T = tip(svg), cross = el("line", { y1: P.t, y2: H - P.b, class: "axis", visibility: "hidden", "stroke-dasharray": "3 3" }, svg);
  const hit = el("rect", { x: P.l, y: P.t, width: W - P.l - P.r, height: H - P.t - P.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect(), mx = (ev.clientX - r.left) / r.width * W; const xv = Math.pow(10, x0 + (mx - P.l) / (W - P.l - P.r) * (x1 - x0));
    const lines = [`${M(xv)} tokens`];
    for (const s of series) { let best = null; for (const p of s.pts) if (p[0] > 0 && (!best || Math.abs(Math.log(p[0] / xv)) < Math.abs(Math.log(best[0] / xv)))) best = p; if (best) lines.push(`${s.short || s.name}: ${(o.yfmt || f3)(best[1])}`); }
    cross.setAttribute("x1", mx); cross.setAttribute("x2", mx); cross.setAttribute("visibility", "visible"); T.show(mx, ev.clientY - r.top + 30, lines, W);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); T.hide(); });
}
function legend(host, series) {
  host.textContent = "";
  for (const s of series) { const sp = document.createElement("span"); const i = document.createElement("i");
    i.style.background = s.dash ? `repeating-linear-gradient(90deg, ${s.color} 0 5px, transparent 5px 8px)` : s.color; sp.append(i, s.name); host.append(sp); }
}
// horizontal bars: items [{label, value, color, note}], value axis from vmin..vmax
function barChart(host, items, o) {
  host.textContent = ""; const W = chartW(host), rowH = 26, P = { l: o.labelW || 230, r: 64, t: 6, b: 26 }, H = P.t + P.b + rowH * items.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.aria || "" }, host);
  const X = (v) => P.l + (v - o.vmin) / (o.vmax - o.vmin) * (W - P.l - P.r);
  for (const v of niceTicks(o.vmin, o.vmax, 6)) { el("line", { x1: X(v), x2: X(v), y1: P.t, y2: H - P.b, class: "grid" }, svg); el("text", { x: X(v), y: H - 8, "text-anchor": "middle" }, svg).textContent = v; }
  const T = tip(svg);
  items.forEach((it, i) => {
    const y = P.t + i * rowH;
    el("text", { x: P.l - 8, y: y + rowH / 2 + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = it.label;
    const x0 = X(o.vmin), w = Math.max(2, X(it.value) - x0);
    if (o.dots) { el("line", { x1: x0, x2: x0 + w, y1: y + rowH / 2, y2: y + rowH / 2, class: "grid" }, svg); el("circle", { cx: x0 + w, cy: y + rowH / 2, r: 6, fill: it.color, stroke: css("--plate"), "stroke-width": 2 }, svg); }
    else el("rect", { x: x0, y: y + 5, width: w, height: rowH - 10, rx: 3, fill: it.color }, svg);
    el("text", { x: x0 + w + 6, y: y + rowH / 2 + 4 }, svg).textContent = (o.fmt || f3)(it.value);
    const hit = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => { const b = svg.getBoundingClientRect(); T.show((ev.clientX - b.left) / b.width * W, y + 10, [it.label, `${(o.fmt || f3)(it.value)} ${o.unit || ""}`, ...(it.note ? [it.note] : [])], W); });
    hit.addEventListener("pointerleave", () => T.hide());
  });
}
// scatter with labels
function scatter(host, pts, o) {
  host.textContent = ""; const W = chartW(host), H = o.h || 330, P = { l: 56, r: 20, t: 14, b: 40 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.aria || "" }, host);
  const X = (v) => P.l + (v - o.xmin) / (o.xmax - o.xmin) * (W - P.l - P.r), Y = (v) => P.t + (o.ymax - v) / (o.ymax - o.ymin) * (H - P.t - P.b);
  for (const v of niceTicks(o.ymin, o.ymax, 5)) { el("line", { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: P.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = o.yfmt ? o.yfmt(v) : v; }
  for (const v of niceTicks(o.xmin, o.xmax, 6)) { el("line", { x1: X(v), x2: X(v), y1: P.t, y2: H - P.b, class: "grid" }, svg); el("text", { x: X(v), y: H - P.b + 16, "text-anchor": "middle" }, svg).textContent = o.xfmt ? o.xfmt(v) : v; }
  el("text", { x: W - P.r, y: H - 6, "text-anchor": "end" }, svg).textContent = o.xlabel; el("text", { x: 4, y: P.t, transform: `rotate(-90 4 ${P.t})`, "text-anchor": "end", dy: "0.8em" }, svg).textContent = o.ylabel;
  if (o.hline != null) el("line", { x1: P.l, x2: W - P.r, y1: Y(o.hline), y2: Y(o.hline), class: "axis", "stroke-dasharray": "4 4" }, svg);
  const T = tip(svg);
  for (const p of pts) {
    const cx = X(p.x), cy = Y(p.y);
    el(p.shape === "square" ? "rect" : "circle", p.shape === "square" ? { x: cx - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: p.color, stroke: css("--plate"), "stroke-width": 2 } : { cx, cy, r: 5.5, fill: p.color, stroke: css("--plate"), "stroke-width": 2 }, svg);
    if (p.label) el("text", { x: cx + 8, y: cy + 4, class: "lbl" }, svg).textContent = p.label;
    const hit = el("circle", { cx, cy, r: 12, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", () => T.show(cx, cy, p.tip, W)); hit.addEventListener("pointerleave", () => T.hide());
  }
}

// ------------------------------------------------------------------------------------------------ results
let R = null;
const run = (n) => R.runs[n];
function seriesFor(names, map = (p) => [p[0], p[1]], base) {
  return names.filter(n => run(n)).map(n => { const s = SERIES[n] || { name: n, slot: 1 };
    let pts = run(n).curve.map(map);
    if (base) { const b = new Map(run(base).curve.map(p => [p[0], p[1]])); pts = run(n).curve.filter(p => b.has(p[0])).map(p => [p[0], p[1] - b.get(p[0])]); }
    return { key: n, name: s.name, short: s.name.replace("class bigram", "cls-bigram"), color: slotColor(s.slot), dash: s.dash, pts }; });
}
function render() {
  const S = R.step0, O = R.oracle, TB = R.token_bins, base = run("h2_base");
  // ---- verdict tiles
  const best = ["h2_stat", "h2_stat_llm", "h2_c_jev", "h2_c_kmeans", "h4_ind"].filter(run).map(n => [n, run(n).final, run(n).mult]).sort((a, b) => a[1] - b[1])[0];
  const W0 = R.written || {}, wj = W0.h2_c_jev, f1 = run("h3_f01"), f3r = run("h3_f1"), nf = run("h2_c_jev");
  const tiles = [
    ["Head start at step 0", `${f2(S.random - Math.min(S.concepts_jev ?? 99, S.concepts_kmeans ?? 99, 6.7207))} nats`, `Best written init starts at ${f2(Math.min(S.concepts_kmeans ?? 99, 6.7207))} vs ${f2(S.random)} for random weights; most of the lead is spent within 10M tokens.`],
    ["Lasting gain at 50M tokens", best && best[2] ? `${best[2].toFixed(2)}× tokens` : "–", best ? `Best: ${SERIES[best[0]]?.name || best[0]}, ${f3(best[1])} vs ${f3(base?.final)} for the baseline.` : ""],
    ["Written neurons still used", wj ? (wj.ablate_written < wj.ablate_random / 2 ? "Barely" : "Yes") : "–", wj ? `After training, ablating the 200 written neurons costs ${f3(wj.ablate_written)} nats vs ${f3(wj.ablate_random)} for 200 ordinary ones.` : ""],
    ["Price of a functional keel", f1 && f3r && nf ? `+${(f1.final - nf.final).toFixed(2)}–${(f3r.final - nf.final).toFixed(2)} nats` : "–", "Forcing the 200 neurons to stay Jev concept detectors (weights 0.1–1) makes them sharp and used, at this loss cost."],
    ["API spend", `$${(R.spend.jev.usd + R.spend.openai.usd).toFixed(2)}`, `Jev $${R.spend.jev.usd.toFixed(2)} for ${(R.spend.jev.requests / 1000).toFixed(0)}k binning calls; gpt-6-luna $${R.spend.openai.usd.toFixed(2)}.`],
  ];
  $("#verdict").innerHTML = tiles.map(([k, v, s]) => `<div><span class="k">${k}</span><span class="v">${v}</span><span class="s">${s}</span></div>`).join("");
  // ---- bins
  $("#bins-text").innerHTML = `Before writing any weights, we measured the ceiling. Suppose the model knew a passage's true Jev topic (an oracle). A topic-conditional unigram model then lowers cross-entropy from <span class="num">${O.unigram_ce.toFixed(2)}</span> to <span class="num">${(O.unigram_ce - O.topic160).toFixed(2)}</span> nats, a gain of only <span class="num">${O.topic160.toFixed(2)}</span>. Token-level bins say more: knowing the <em>class</em> of the previous token lowers cross-entropy by <span class="num">${(TB.unigram - TB.llm64).toFixed(2)}</span> nats with the 64 LLM-named classes and <span class="num">${(TB.unigram - TB.km64).toFixed(2)}</span> with 64 distributional clusters. The full bigram table gets <span class="num">${(TB.unigram - TB.full_bigram).toFixed(2)}</span>. A small model learns about 3.5 nats in its first 25M tokens, so passage-level abstractions are a small slice of what it needs.`;
  const binItems = [
    { label: "Jev format, oracle (40)", value: O.format40, color: slotColor(5), note: "passage-level, 4k held-out Jev-labelled passages" },
    { label: "Jev topic, oracle (160)", value: O.topic160, color: slotColor(5), note: "passage-level, 4k held-out Jev-labelled passages" },
    { label: "k-means passage bin, oracle (200)", value: O.kmeans200, color: slotColor(2) },
    { label: "LLM token class of previous token (64)", value: TB.unigram - TB.llm64, color: slotColor(4) },
    { label: "k-means token class of previous token (64)", value: TB.unigram - TB.km64, color: slotColor(3) },
    { label: "full bigram table (50k × 50k)", value: TB.unigram - TB.full_bigram, color: css("--muted") },
  ];
  barChart($("#chart-bins"), binItems, { vmin: 0, vmax: 2.4, unit: "nats saved", aria: "Next-token cross-entropy saved by knowing each kind of bin", labelW: 290 });
  $("#cap-bins").textContent = "Nats of next-token cross-entropy saved relative to a unigram model when the bin is known (higher = more informative). Passage bins are oracles scored on 4k held-out Jev-labelled passages; token bins are scored on 4M validation tokens. Shuffling the Jev labels across passages removes the gain entirely (" + O.shuffled.toFixed(2) + " nats).";
  // ---- step 0
  $("#construct-text").innerHTML = `The token bins become a class-based bigram written straight into the embedding and unembedding: each token's embedding carries a one-hot of its class, and the unembedding stores <span class="mono">log p(class<sub>next</sub> | class)</span>. The passage bins become a three-part circuit. Each token's embedding gets its naive-Bayes evidence for every concept, <span class="mono">log p(token | concept) − log p(token)</span>. One attention head with zeroed queries and keys averages that evidence over the document so far. One MLP neuron per concept (200 of the layer's 1024) reads its average, and its output direction maps through the unembedding to the concept's next-token boost. Only four global scale knobs are tuned, on held-out loss. Every other weight is identical to the baseline's random init.`;
  const stepItems = [
    { label: "unigram bias", value: S.unigram, color: slotColor(1) },
    { label: "class bigram, LLM token classes", value: S.class_bigram_llm64, color: slotColor(4) },
    { label: "class bigram, k-means token bins", value: S.class_bigram_km64, color: slotColor(3) },
    { label: "+ shuffled Jev concepts", value: S.concepts_shuf, color: slotColor(5) },
    { label: "+ coarse Jev concepts (16+40)", value: S.concepts_coarse, color: slotColor(5) },
    { label: "+ Jev concepts (160+40)", value: S.concepts_jev, color: slotColor(5) },
    { label: "+ k-means concepts (200)", value: S.concepts_kmeans, color: slotColor(2) },
  ].filter(x => x.value != null && x.value < 8);
  barChart($("#chart-step0"), stepItems, { vmin: 6.6, vmax: 7.5, unit: "nats", aria: "Step-0 validation loss per init", labelW: 250, dots: true, fmt: (v) => v.toFixed(3) });
  $("#cap-step0").textContent = `Validation loss before any training (256k held-out tokens; lower is better; note the zoomed axis). Random weights start at ${f2(S.random)}. The baseline reaches 6.75 after roughly 2–3M tokens, about 150 optimizer steps.`;
  // ---- training curves
  const h2 = ["h2_stat", "h2_stat_llm", "h2_c_jev", "h2_c_kmeans", "h2_c_shuf", "h2_c_coarse", "h2_uni"];
  if (base) {
    const gap = seriesFor(h2, undefined, "h2_base");
    lineChart($("#chart-gap"), gap, { ymin: -0.25, ymax: 0.05, zero: true, ylabel: "Δ val loss vs baseline (nats)", yfmt: (v) => (v > 0 ? "+" : "") + v.toFixed(2), aria: "Loss difference to baseline over training" });
    legend($("#leg-gap"), gap);
    const ear = ["h2_base", ...h2].filter(n => run(early(n))).map(n => { const s = SERIES[n]; return { key: n, name: s.name, short: s.name, color: slotColor(s.slot), dash: s.dash, pts: run(early(n)).curve.map(p => [p[0], p[1]]) }; });
    if (ear.length) { lineChart($("#chart-early"), ear, { ymin: 5.5, ymax: 11, xmin: 1.5e5, ylabel: "val loss (nats)", aria: "Validation loss over the first 2.5M tokens" }); legend($("#leg-early"), ear); }
  }
  const mults = h2.filter(run).map(n => [n, run(n).mult]).filter(x => x[1]);
  $("#train-text").innerHTML = base ? `Every written init starts far ahead and then loses most of its lead within the first ten million tokens. At 50M tokens the baseline ends at <span class="num">${f3(base.final)}</span>. The table converts each run's final loss into <em>effective baseline tokens</em>, using baselines trained to completion at 12.5M, 25M, 50M and 100M tokens. ${mults.length ? `The multipliers range from <span class="num">${Math.min(...mults.map(x => x[1])).toFixed(2)}×</span> to <span class="num">${Math.max(...mults.map(x => x[1])).toFixed(2)}×</span>.` : ""}` : "";
  $("#cap-gap").textContent = "Validation loss minus the random-init baseline at the same token count (below zero = ahead). Same seed, data order and schedule for every run.";
  $("#cap-early").textContent = "The same comparison in the first 150 optimizer steps (separate runs with identical warmup), where the inits differ most.";
  // ---- table
  const rows = ["h2_base", "h2_uni", "h2_stat", "h2_stat_llm", "h2_stat_rnd", "h2_c_jev", "h2_c_jev_nn", "h2_c_jev_live", "h2_c_kmeans", "h2_c_coarse", "h2_c_shuf",
                "h4_ind", "h4_ind_w10", "h4_ind_ema", "h4_ind_w10_crit",
                "h3_w10", "h3_w100", "h3_w10_crit", "h3_w10_ema", "h3_w10_rand", "h3_mix_only", "h3_f01", "h3_f03", "h3_f03_t01", "h3_f1", "h3_f1_crit", "h3_f03_live", "h3_f1_rand", "h3_f03_rand_L3", "h3_f03_rand_L3_t01", "h3_f1_rand_L3"].filter(run);
  const jevIdx = (n) => { const c = run(n).config; return (c.init_from || "").includes("c_jev") || !!c.concept_w; };
  const fid = (n) => { const c = run(n).curve.at(-1); return c.length > 2 && jevIdx(n) ? c : null; };
  $("#table-runs").innerHTML = `<table><thead><tr><th>run</th><th>init</th><th>righting force</th><th>final loss</th><th>Δ vs base</th><th>eff. tokens ×</th><th>topic fid.</th><th>format fid.</th></tr></thead><tbody>${rows.map(n => {
    const r = run(n), c = r.config, f = fid(n), init = c.init_from ? c.init_from.split("/").pop().replace(".bin", "") : "random";
    const lay = c.concept_layer && c.concept_layer !== "0" ? ` (layer ${+c.concept_layer + 1})` : "", tau = c.concept_tau ? `, τ=${c.concept_tau}` : "";
    const force = c.concept_w && +c.concept_w < 1e-6 ? "labelled-data mixing only" : c.anchor ? `weights λ=${c.anchor_lambda}${c.anchor_sched && c.anchor_sched !== "const" ? " " + c.anchor_sched : ""}${c.anchor_ema ? " EMA" : ""}${c.anchor.includes("_in") ? " (inputs)" : c.anchor.includes("_out") ? " (outputs)" : ""}` : c.concept_w ? `function w=${c.concept_w}${tau}${lay}${c.concept_sched && c.concept_sched !== "const" ? " " + c.concept_sched : ""}` : "—";
    return `<tr><td>${n}</td><td><span class="pill ${init === "random" ? "he" : "eo"}">${init}</span></td><td>${force}</td><td>${f3(r.final)}</td><td>${n === "h2_base" ? "" : (r.final - base.final >= 0 ? "+" : "") + (r.final - base.final).toFixed(3)}</td><td>${r.mult ? r.mult.toFixed(2) : "–"}</td><td>${f ? (f[2] * 100).toFixed(1) + "%" : "–"}</td><td>${f ? (f[3] * 100).toFixed(1) + "%" : "–"}</td></tr>`; }).join("")}</tbody></table>`;
  $("#cap-table").textContent = "Eff. tokens ×: the budget at which a fully trained baseline would reach this run's final loss, divided by 50M (log-linear interpolation of the budget curve). Fidelity: how often the argmax over the 160 topic (or 40 format) concept neurons matches Jev's label on 4,000 held-out passages; chance is under 1% and 2.5%.";
  renderSeeds(); renderForce(); renderCopy(); renderScale(); renderClean(); renderFindings();
}
function renderForce() {
  const names = ["h2_c_jev", "h3_w1", "h3_w10", "h3_w100", "h3_w10_crit", "h3_w10_ema", "h3_w10_in", "h3_w10_out", "h3_f01", "h3_f1", "h3_f1_crit", "h3_f1_rand", "h3_w10_rand", "h2_base"].filter(run);
  const base = run("h2_base"); if (!base || names.length < 3) { $("#force-text").textContent = "Righting-force runs are still training."; return; }
  const WS0 = R.written || {}, nf = run("h2_c_jev"), dl = (n) => run(n) ? (run(n).final - nf.final >= 0 ? "+" : "") + (run(n).final - nf.final).toFixed(3) : "–";
  const ws = (k, f) => WS0[k] ? WS0[k][f] : null;
  $("#force-text").innerHTML = `Left alone, the written neurons fade from use. Only <span class="num">${ws("h2_c_jev", "live_written") ?? "–"}</span> of 200 ever fire after training, and ablating all 200 raises loss by just <span class="num">${f3(ws("h2_c_jev", "ablate_written"))}</span>, against <span class="num">${f3(ws("h2_c_jev", "ablate_random"))}</span> for 200 ordinary neurons of the same layer. Two kinds of righting force were tried. <strong>Weight anchors</strong> pull the written weights back toward their constructed values. They are almost free (<span class="num">${dl("h3_w10")}</span> nats at λ=10, <span class="num">${dl("h3_w10_ema")}</span> with a drifting anchor), but the network routes around the anchored neurons: ablating them then costs <span class="num">${f3(ws("h3_w10", "ablate_written"))}</span>. Anchoring the same entries of a <em>random</em> init costs <span class="num">${run("h3_w10_rand") ? (run("h3_w10_rand").final - run("h2_base").final).toFixed(3) : "–"}</span>, so the written values are as useful as what training would have put there. <strong>Functional anchors</strong> train the 200 neurons to predict Jev's concept on labelled passages. They make the neurons strong detectors that the network does use (topic fidelity <span class="num">${ws("h3_f1", "fid_topic") != null ? (ws("h3_f1", "fid_topic") * 100).toFixed(0) + "%" : "–"}</span>), but they cost loss: <span class="num">${dl("h3_f1")}</span> at full strength, <span class="num">${dl("h3_f01")}</span> at a tenth. Releasing the force halfway through training (a critical period) keeps most of the structure for <span class="num">${dl("h3_f1_crit")}</span>.`;
  const kind = (n) => (n === "h2_base" || n.startsWith("h2_c_jev")) ? 1 : n.includes("_rand") ? 3 : n.startsWith("h3_f") ? 5 : n.startsWith("h3_w") ? 2 : 5;
  const pts = names.map(n => { const r = run(n), c = r.curve.at(-1); const fidv = c.length > 2 ? c[2] : 0;
    return { x: fidv * 100, y: r.final, color: slotColor(kind(n)), shape: n.includes("_rand") || n === "h2_base" ? "square" : "circle", label: n.replace("h3_", "").replace("h2_", ""),
             tip: [n, `final loss ${f3(r.final)} (${(r.final - base.final >= 0 ? "+" : "") + (r.final - base.final).toFixed(3)})`, `topic fidelity ${(fidv * 100).toFixed(1)}%`] }; });
  const ys = pts.map(p => p.y), xs = pts.map(p => p.x);
  scatter($("#chart-force"), pts, { xmin: 0, xmax: Math.max(10, Math.ceil(Math.max(...xs) / 10) * 10), ymin: Math.floor((Math.min(...ys) - 0.01) * 50) / 50, ymax: Math.ceil((Math.max(...ys) + 0.01) * 50) / 50, hline: base.final, xlabel: "topic fidelity of the 160 concept neurons (%)", ylabel: "final val loss", yfmt: (v) => v.toFixed(2), aria: "Final loss against concept fidelity" });
  const WS = R.written || {};
  const wrows = [["init_s_c_jev", "written, prior thresholds (step 0)"], ["h2_c_jev", "… trained 50M, no force"], ["h3_mix_only", "… + labelled-data mixing only"],
                 ["init_s_c_jev_live", "written, quantile thresholds (step 0)"], ["h2_c_jev_live", "… trained 50M, no force"], ["h3_f03_live", "… + functional anchor w=0.3"],
                 ["h3_w10", "prior thresholds + weight anchor λ=10"], ["h3_w100", "… weight anchor λ=100"], ["h3_w10_ema", "… drifting (EMA) anchor"], ["h3_w10_crit", "… anchor released by 50%"],
                 ["h3_f01", "prior thresholds + functional anchor w=0.1"], ["h3_f03", "… functional anchor w=0.3"], ["h3_f1", "… functional anchor w=1"], ["h3_f1_crit", "… w=1 released by 50%"],
                 ["h3_f03_t01", "… w=0.3, temperature 0.1"], ["h3_f1_rand", "random init + functional anchor w=1"]].filter(r => WS[r[0]]);
  $("#table-written").innerHTML = wrows.length ? `<table><thead><tr><th>model</th><th>live written neurons</th><th>topic fid.</th><th>format fid.</th><th>Δ loss, ablate written 200</th><th>Δ loss, ablate 200 others</th></tr></thead><tbody>${wrows.map(([k, label]) => { const w = WS[k];
    return `<tr><td>${label}</td><td>${w.live_written} / 200</td><td>${(w.fid_topic * 100).toFixed(1)}%</td><td>${(w.fid_format * 100).toFixed(1)}%</td><td>+${w.ablate_written.toFixed(3)}</td><td>+${w.ablate_random.toFixed(3)}</td></tr>`; }).join("")}</tbody></table>` : "";
  $("#cap-force").textContent = "Circles: written init (Jev concepts). Squares: random init, with the same force applied to the same 200 neurons. Violet: no force. Orange: weight anchor (λ = how hard weights are pulled back, relative to each tensor's scale). Magenta: functional anchor (the 200 neurons are trained to predict Jev's concept on labelled passages). Dashed line: random-init baseline.";
}
function renderSeeds() {
  const fam = [["base", "random init", "h2_base"], ["stat", "class bigram (k-means token bins)", "h2_stat"], ["stat_llm", "class bigram (LLM token classes)", "h2_stat_llm"],
               ["c_jev", "+ Jev concepts", "h2_c_jev"], ["c_kmeans", "+ k-means concepts", "h2_c_kmeans"]];
  const get = (k, sd, s1) => sd === 1 ? run(s1) : run(`seed_${k}_${sd}`);
  const rows = fam.map(([k, label, s1]) => { const v = [1, 2, 3].map(sd => get(k, sd, s1)?.final).filter(x => x != null); const b = [1, 2, 3].map(sd => [get(k, sd, s1)?.final, get("base", sd, "h2_base")?.final]).filter(x => x[0] != null && x[1] != null).map(x => x[0] - x[1]);
    const mean = (a) => a.reduce((p, q) => p + q, 0) / a.length, sd = (a) => a.length > 1 ? Math.sqrt(a.reduce((p, q) => p + (q - mean(a)) ** 2, 0) / (a.length - 1)) : NaN;
    return { label, n: v.length, m: v.length ? mean(v) : NaN, s: sd(v), dm: b.length ? mean(b) : NaN, ds: sd(b), nb: b.length }; });
  if (!rows.some(r => r.n > 1)) { $("#fig-seeds").hidden = true; return; }
  $("#fig-seeds").hidden = false;
  $("#table-seeds").innerHTML = `<table><thead><tr><th>init</th><th>seeds</th><th>final loss, mean ± sd</th><th>Δ vs same-seed baseline, mean ± sd</th></tr></thead><tbody>${rows.map(r =>
    `<tr><td>${r.label}</td><td>${r.n}</td><td>${f3(r.m)} ± ${isNaN(r.s) ? "–" : r.s.toFixed(3)}</td><td>${r.label === "random init" ? "" : (r.dm >= 0 ? "+" : "") + r.dm.toFixed(3) + " ± " + (isNaN(r.ds) ? "–" : r.ds.toFixed(3))}</td></tr>`).join("")}</tbody></table>`;
  $("#cap-seeds").textContent = "Seed 1 is the run in the table above; seeds 2 and 3 change both the random init (the construction is rebuilt on it) and the data order. Δ is paired with the baseline of the same seed.";
}
function renderScale() {
  const names = ["m_ind", "m_c_jev"].filter(run); if (!run("m_base") || !names.length) { $("#scale-text").textContent = "The 6-layer runs are still training."; return; }
  const S2 = { m_ind: { name: "LLM-class bigram + copy circuit", slot: 6 }, m_c_jev: { name: "class bigram + Jev concepts", slot: 5 } };
  const b = new Map(run("m_base").curve.map(p => [p[0], p[1]]));
  const ser = names.map(n => ({ name: S2[n].name, short: S2[n].name, color: slotColor(S2[n].slot), pts: run(n).curve.filter(p => b.has(p[0])).map(p => [p[0], p[1] - b.get(p[0])]) }));
  lineChart($("#chart-scale"), ser, { ymin: -0.3, ymax: 0.05, zero: true, ylabel: "Δ val loss vs 6-layer baseline", yfmt: (v) => (v > 0 ? "+" : "") + v.toFixed(2), aria: "Scale check loss gap" });
  legend($("#leg-scale"), ser);
  const mb = run("m_base").final;
  $("#scale-text").innerHTML = `The same constructions, unchanged, on a 6-layer d=384 model trained for 100M tokens (learning rate scaled by width, no re-tuning). The baseline ends at <span class="num">${f3(mb)}</span>; ${names.map(n => `${S2[n].name} at <span class="num">${f3(run(n).final)}</span> (${(run(n).final - mb >= 0 ? "+" : "") + (run(n).final - mb).toFixed(3)})`).join(", and ")}.`;
}
function renderCopy() {
  const names = ["h2_stat_llm", "h4_ind", "h4_ind_w10", "h4_ind_w10_crit", "h4_ind_ema"].filter(run);
  if (!run("h2_base") || names.length < 2) { $("#copy-text").textContent = "Copy-circuit runs are still training."; return; }
  const ser = seriesFor(names, undefined, "h2_base");
  lineChart($("#chart-copy"), ser, { ymin: -0.3, ymax: 0.05, zero: true, ylabel: "Δ val loss vs baseline (nats)", yfmt: (v) => (v > 0 ? "+" : "") + v.toFixed(2), aria: "Copy circuit loss gap" });
  legend($("#leg-copy"), ser);
  const T = window.COPY_TEXT; $("#copy-text").innerHTML = T || "";
  $("#cap-copy").textContent = "Same seed, data and schedule as the baseline. The copy circuit is a previous-token head in layer 1 and an induction head in layer 2, written by hand; anchored variants pull those two heads back toward the written pattern.";
}
function renderClean() {
  const h = R.h1; if (!h || !h.base) return;
  const facet = "fine";
  const layers = h.base.map(r => r.layer);
  const mk = (key, color, dash, getter) => ({ name: key, color, dash, pts: layers.map(l => [l + 1, getter(h.base[l])]) });
  const W = [mk("neurons: passage topic", slotColor(5), "", r => r.neuron.fine_purity), mk("random directions: passage topic", slotColor(5), "5 4", r => r.random_dir.fine_purity)];
  if (h.base_tok) {
    const tk = (key, color, dash, get) => ({ name: key, color, dash, pts: h.base_tok.map(r => [r.layer + 1, get(r)]) });
    W.push(tk("neurons: LLM token class", slotColor(4), "", r => r.neuron.cur_llm), tk("random directions: LLM token class", slotColor(4), "5 4", r => r.random_dir.cur_llm));
  }
  linPlot($("#chart-layers"), W, { xlabel: "layer", ylabel: "purity of top-20 activations", xticks: layers.map(l => l + 1), ymin: 0.2, ymax: 0.6 });
  legend($("#leg-layers"), W);
  const steps = Object.keys(h.growth).filter(s => h.growth[s]).map(Number).sort((a, b) => a - b);
  const last = (s) => h.growth[s].at(-1);
  const G = [{ name: "neurons, last layer", color: slotColor(5), pts: steps.map(s => [s * 16384, last(s).neuron.fine_purity]) },
             { name: "random directions, last layer", color: slotColor(5), dash: "5 4", pts: steps.map(s => [s * 16384, last(s).random_dir.fine_purity]) },
             { name: "neurons, format", color: slotColor(3), pts: steps.map(s => [s * 16384, last(s).neuron.format_purity]) },
             { name: "random directions, format", color: slotColor(3), dash: "5 4", pts: steps.map(s => [s * 16384, last(s).random_dir.format_purity]) }];
  lineChart($("#chart-growth"), G, { ymin: 0.2, ymax: 0.5, ylabel: "purity of top-20 passages", yfmt: (v) => v.toFixed(2), aria: "Neuron vs random-direction purity over training" });
  legend($("#leg-growth"), G);
  const L3 = h.base.at(-1);
  $("#clean-text").innerHTML = `For every MLP neuron of the trained baseline we took its 20 most strongly activating held-out passages and asked how many share one Jev concept (purity). As a control we did the same for random directions in the same activation space, and for both against a label-permutation null. In the last layer, neurons' top passages are <span class="num">${(L3.neuron.fine_purity * 100).toFixed(0)}%</span> one fine topic against <span class="num">${(L3.random_dir.fine_purity * 100).toFixed(0)}%</span> for random directions. <span class="num">${(L3.neuron.format_purity_aligned * 100).toFixed(0)}%</span> of neurons beat the null's 99th percentile on format, against <span class="num">${(L3.random_dir.format_purity_aligned * 100).toFixed(0)}%</span> of random directions. In layer 1 the two are indistinguishable. The gap opens during training, so the neuron basis becomes privileged as the model learns, which is what Goodfire's picture predicts. A dense measure (AUROC on passage-mean activations) shows no such gap: nearly every unit correlates with topic.`;
  $("#cap-layers").textContent = "Baseline after 50M tokens. Passage topic: share of a unit's 20 most-activating held-out passages with the same Jev topic (160 classes, null 0.10). LLM token class: share of its 20 most-activating tokens in the same gpt-6-luna token class (64 classes, null 0.20). Solid lines are neurons, dashed are 1,024 random unit directions in the same activation space.";
  $("#cap-growth").textContent = "Last-layer purity at checkpoints every 500 steps of the baseline run.";
}
function linPlot(host, series, o) {   // linear-x line plot for per-layer charts
  host.textContent = ""; const W = chartW(host), H = 260, P = { l: 52, r: 16, t: 12, b: 36 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" }, host);
  const xs = o.xticks, X = (x) => P.l + (x - xs[0]) / (xs.at(-1) - xs[0]) * (W - P.l - P.r), Y = (y) => P.t + (o.ymax - y) / (o.ymax - o.ymin) * (H - P.t - P.b);
  for (const v of niceTicks(o.ymin, o.ymax, 5)) { el("line", { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: P.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(2); }
  for (const v of xs) el("text", { x: X(v), y: H - P.b + 16, "text-anchor": "middle" }, svg).textContent = `L${v}`;
  el("text", { x: W - P.r, y: H - 4, "text-anchor": "end" }, svg).textContent = o.xlabel; el("text", { x: 4, y: P.t, transform: `rotate(-90 4 ${P.t})`, "text-anchor": "end", dy: "0.8em" }, svg).textContent = o.ylabel;
  const T = tip(svg);
  for (const s of series) {
    el("path", { d: s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]) + " " + Y(p[1])).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }, svg);
    for (const p of s.pts) { el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: css("--plate"), "stroke-width": 2 }, svg);
      const hit = el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 10, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", () => T.show(X(p[0]), Y(p[1]), [s.name, `layer ${p[0]}: ${p[1].toFixed(3)}`], W)); hit.addEventListener("pointerleave", () => T.hide()); }
  }
}
function renderFindings() {
  const F = window.FINDINGS || [];
  $("#findings").innerHTML = F.map(f => `<p>${f}</p>`).join("");
  $("#methods-text").innerHTML = (window.METHODS || []).map(f => `<p>${f}</p>`).join("");
  $("#foot").innerHTML = window.FOOT || "";
}

// ------------------------------------------------------------------------------------------------ microscope
const MODELS = window.WEB_MODELS || [];
let bpe = null, cur = { key: null, model: null, engine: null };
const conceptNames = () => [...R.taxonomy.topic_groups.flatMap(g => g.concepts), ...R.taxonomy.formats];
function tint(a, max) { const t = Math.max(0, Math.min(1, a / (max || 1))); return `color-mix(in srgb, var(--eosin) ${Math.round(t * 55)}%, transparent)`; }
async function ensureModel(key) {
  if (cur.key === key) return cur;
  const spec = MODELS.find(m => m.key === key); $("#status").textContent = `Fetching ${spec.label}…`;
  const model = await loadModel(spec.path, (f) => { $("#status").textContent = `Fetching ${spec.label}… ${Math.round(f * 100)}%`; });
  let engine = null;
  try { engine = await GPUModel.create(model); } catch (e) { engine = new CPUModel(model); }
  cur = { key, model, engine, spec }; return cur;
}
function neuronNames(spec, layer, m) {
  if (spec.concepts && layer === 0) { const n = conceptNames(); return Array.from({ length: m }, (_, k) => k < n.length ? n[k] : `unwritten #${k}`); }
  const A = spec._auto?.[layer]; const topics = R.taxonomy.topic_groups.flatMap(g => g.concepts), tc = R.token_classes;
  return Array.from({ length: m }, (_, k) => {
    if (!A) return `neuron #${k}`;
    const cands = [[A.tok_purity[k], `token: ${tc[A.tok[k]]}`], [A.next_purity[k] * 0.95, `next: ${tc[A.next[k]]}`], [Math.min(1, A.topic_lift[k] / 6), `topic: ${topics[A.topic[k]]}`]];
    cands.sort((a, b) => b[0] - a[0]); return `#${k} ${cands[0][0] >= 0.4 ? cands[0][1] : "mixed"}`;
  });
}
async function runScope() {
  const btn = $("#run"); btn.disabled = true;
  try {
    if (!bpe) bpe = new BPE(await (await fetch("assets/gpt2_bpe.json")).json());
    const { engine, model, spec } = await ensureModel($("#m-sel").value);
    if (spec.autolabel && !spec._auto) spec._auto = await (await fetch(spec.autolabel)).json();
    const layer = spec.concepts ? 0 : +$("#layer").value, m = model.cfg.m;
    const K = spec.concepts ? conceptNames().length : m, names = neuronNames(spec, layer, m);
    const ids = [bpe.eot, ...bpe.encode($("#text").value)].slice(0, model.cfg.T - $("#ngen").value - 1);
    engine.reset(); const steps = []; const t0 = performance.now(); let out = null;
    // written concept neurons: rank by pre-activation (their evidence score); trained neurons: ReLU² output (what the network uses)
    const keep = (o) => spec.concepts ? Float32Array.from(o.acts[layer].slice(0, K)) : Float32Array.from(o.acts[layer].slice(0, K), a => (a > 0 ? a * a : 0));
    for (const id of ids) { out = await engine.step(id, bpe.eot); steps.push({ id, acts: keep(out), logits: out.logits }); }
    const ngen = +$("#ngen").value, temp = +$("#temp").value;
    for (let i = 0; i < ngen && steps.length < model.cfg.T; i++) {
      const nx = sample(out.logits, temp); if (nx === bpe.eot) break;
      out = await engine.step(nx, bpe.eot); steps.push({ id: nx, acts: keep(out), logits: out.logits, gen: true });
    }
    const dt = (performance.now() - t0) / steps.length;
    $("#status").textContent = `${spec.label} · ${engine.kind === "webgpu" ? "WebGPU" : "CPU"} · ${steps.length} tokens at ${dt.toFixed(0)} ms/token · ${spec.concepts ? "200 written concept neurons, layer 1" : `all ${m} neurons of layer ${layer + 1}, auto-labelled`}. Underlined tokens were generated.`;
    const box = $("#tokens"); box.textContent = "";
    const peaks = steps.slice(1).map(s => Math.max(...s.acts)), lo = spec.concepts ? Math.min(...peaks) : 0, maxA = Math.max(1e-6, Math.max(...peaks) - lo);
    steps.forEach((s, i) => {
      if (i === 0) return;
      const sp = document.createElement("span"); sp.className = "tok" + (s.gen ? " gen" : ""); sp.textContent = bpe.piece(s.id);
      sp.style.background = tint(Math.max(...s.acts) - lo, maxA); sp.tabIndex = 0;
      const sel = () => { box.querySelectorAll(".sel").forEach(x => x.classList.remove("sel")); sp.classList.add("sel"); inspect(steps, i, names, maxA, spec, lo); };
      sp.addEventListener("click", sel); sp.addEventListener("keydown", (e) => { if (e.key === "Enter") sel(); });
      box.append(sp);
    });
    inspect(steps, Math.min(steps.length - 1, ids.length - 1), names, maxA, spec, lo);
  } catch (e) { $("#status").textContent = "Could not run the model: " + e.message; console.error(e); }
  btn.disabled = false;
}
function inspect(steps, i, names, maxA, spec, lo = 0) {
  const s = steps[i], order = (from, to) => Array.from({ length: to - from }, (_, k) => k + from).sort((a, b) => s.acts[b] - s.acts[a]);
  const idx = spec && spec.concepts ? [...order(0, 160).slice(0, 3), ...order(160, s.acts.length).slice(0, 3)] : order(0, s.acts.length).slice(0, 6);
  $("#ins-title").textContent = spec && spec.concepts ? `Top topic and format neurons at “${bpe.piece(s.id)}”` : `Strongest neurons at “${bpe.piece(s.id)}”`;
  $("#ins-neurons").innerHTML = idx.map(k => `<div class="bar"><span class="lab"><i style="width:${Math.max(0, Math.min(100, (s.acts[k] - lo) / maxA * 100))}%"></i><b>${names[k].startsWith("#") ? names[k] : "#" + k + " " + names[k]}</b></span><span class="val">${s.acts[k].toFixed(2)}</span></div>`).join("");
  const L = s.logits, top = Array.from(L.keys()).sort((a, b) => L[b] - L[a]).slice(0, 6), mx = L[top[0]];
  let z = 0; for (let j = 0; j < L.length; j++) z += Math.exp(L[j] - mx);
  $("#ins-pred").innerHTML = top.map(k => { const p = Math.exp(L[k] - mx) / z; return `<div class="bar"><span class="lab"><i style="width:${p * 100}%"></i><b>${JSON.stringify(bpe.piece(k)).slice(1, -1)}</b></span><span class="val">${(p * 100).toFixed(1)}%</span></div>`; }).join("");
}
function initScope() {
  const sel = $("#m-sel"); sel.innerHTML = MODELS.map(m => `<option value="${m.key}">${m.label}</option>`).join("");
  const syncLayer = () => { const spec = MODELS.find(m => m.key === sel.value); $("#layer-wrap").hidden = !!spec?.concepts; };
  sel.addEventListener("change", syncLayer); syncLayer();
  $("#temp").addEventListener("input", () => { $("#temp-v").textContent = (+$("#temp").value).toFixed(2); });
  $("#ngen").addEventListener("input", () => { $("#ngen-v").textContent = $("#ngen").value; });
  $("#run").addEventListener("click", runScope);
  if (!MODELS.length) { $("#run").disabled = true; $("#status").textContent = "No models are bundled with this page."; }
}

// ------------------------------------------------------------------------------------------------ boot
(async () => {
  try { R = await (await fetch("data/results.json")).json(); } catch (e) { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="col note">Could not load results data (${e.message}).</p>`); return; }
  try { const t = await (await fetch("data/text.json")).json(); window.FINDINGS = t.findings; window.METHODS = t.methods; window.FOOT = t.foot; window.COPY_TEXT = t.copy_text; window.WEB_MODELS = t.models; MODELS.push(...(t.models || [])); } catch (e) {}
  render(); initScope();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  let rt = null, lastW = innerWidth; addEventListener("resize", () => { if (Math.abs(innerWidth - lastW) < 8) return; lastW = innerWidth; clearTimeout(rt); rt = setTimeout(render, 150); });
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
