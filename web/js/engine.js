// PLASTIC inference engine for the browser: the same GPT as the CUDA trainer (RoPE, param-free RMSNorm, ReLU^2 MLP
// with bias, untied lm_head + bias, doc-masked causal attention), int8 weights with per-row scales, KV cache.
// Two backends with identical math: WebGPU (WGSL compute shaders) and a plain-JS CPU fallback.

export async function loadModel(url, onProgress) {
  const meta = await (await fetch(url + "/meta.json")).json();
  const buf = new Uint8Array(meta.total_bytes); let off = 0;
  for (const [i, c] of meta.chunks.entries()) {
    let b;
    if (meta.encoding === "b64") {   // weights ship as base64 text so static hosts serve them; decode to bytes
      const s = atob((await (await fetch(url + "/" + c)).text()).trim()); b = new Uint8Array(s.length);
      for (let k = 0; k < s.length; k++) b[k] = s.charCodeAt(k);
    } else b = new Uint8Array(await (await fetch(url + "/" + c)).arrayBuffer());
    buf.set(b, off); off += b.length; onProgress && onProgress((i + 1) / meta.chunks.length);
  }
  const T = {};
  for (const t of meta.tensors)
    T[t.name] = t.kind === "int8" ? new Int8Array(buf.buffer, t.offset, t.bytes) : new Float32Array(buf.buffer, t.offset, t.bytes / 4);
  return { meta, cfg: meta.cfg, T, buf, tensors: meta.tensors };
}

function ropeTables(T, hd) {
  const h2 = hd / 2, cos = new Float32Array(T * h2), sin = new Float32Array(T * h2);
  for (let t = 0; t < T; t++) for (let i = 0; i < h2; i++) {
    const fr = Math.pow(10000, -2 * i / hd); cos[t * h2 + i] = Math.cos(t * fr); sin[t * h2 + i] = Math.sin(t * fr);
  }
  return { cos, sin };
}

// ------------------------------------------------------------------------------------------------ CPU backend
export class CPUModel {
  constructor(model) {
    this.m = model; const c = (this.c = model.cfg); this.hd = c.d / c.H; this.rope = ropeTables(c.T, this.hd);
    this.kind = "cpu"; this.reset();
  }
  reset() {
    const c = this.c; this.t = 0; this.ds = 0;
    this.K = Array.from({ length: c.L }, () => new Float32Array(c.T * c.d));
    this.V = Array.from({ length: c.L }, () => new Float32Array(c.T * c.d));
  }
  mv(name, x, rows, cols, bias, out) {
    const W = this.m.T[name], S = this.m.T[name + ".scale"]; out = out || new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let acc = 0; const o = r * cols;
      for (let k = 0; k < cols; k++) acc += W[o + k] * x[k];
      out[r] = acc * S[r] + (bias ? bias[r] : 0);
    }
    return out;
  }
  rms(x) { let s = 0; for (const v of x) s += v * v; const r = 1 / Math.sqrt(s / x.length + 1e-6); return x.map(v => v * r); }
  async step(tok, eot) {
    const c = this.c, d = c.d, H = c.H, hd = this.hd, h2 = hd / 2, t = this.t;
    if (tok === eot) this.ds = t;
    const W = this.m.T.wte, S = this.m.T["wte.scale"]; let x = new Float32Array(d);
    for (let k = 0; k < d; k++) x[k] = W[tok * d + k] * S[tok];
    const acts = [];
    for (let l = 0; l < c.L; l++) {
      const qkv = this.mv(`wqkv${l}`, this.rms(x), 3 * d, d);
      const q = new Float32Array(d), Kc = this.K[l], Vc = this.V[l];
      for (let h = 0; h < H; h++) for (let i = 0; i < h2; i++) {
        const co = this.rope.cos[t * h2 + i], si = this.rope.sin[t * h2 + i], b = h * hd;
        const q0 = qkv[b + i], q1 = qkv[b + i + h2], k0 = qkv[d + b + i], k1 = qkv[d + b + i + h2];
        q[b + i] = q0 * co - q1 * si; q[b + i + h2] = q0 * si + q1 * co;
        Kc[t * d + b + i] = k0 * co - k1 * si; Kc[t * d + b + i + h2] = k0 * si + k1 * co;
        Vc[t * d + b + i] = qkv[2 * d + b + i]; Vc[t * d + b + i + h2] = qkv[2 * d + b + i + h2];
      }
      const y = new Float32Array(d), sc = 1 / Math.sqrt(hd);
      for (let h = 0; h < H; h++) {
        const b = h * hd, n = t - this.ds + 1, s = new Float32Array(n); let mx = -Infinity;
        for (let j = 0; j < n; j++) { let a = 0; const kk = (this.ds + j) * d + b; for (let i = 0; i < hd; i++) a += q[b + i] * Kc[kk + i]; s[j] = a * sc; mx = Math.max(mx, s[j]); }
        let z = 0; for (let j = 0; j < n; j++) { s[j] = Math.exp(s[j] - mx); z += s[j]; }
        for (let j = 0; j < n; j++) { const p = s[j] / z, vv = (this.ds + j) * d + b; for (let i = 0; i < hd; i++) y[b + i] += p * Vc[vv + i]; }
      }
      const ao = this.mv(`wo${l}`, y, d, d); for (let k = 0; k < d; k++) x[k] += ao[k];
      const a = this.mv(`w1_${l}`, this.rms(x), c.m, d, this.m.T[`b1_${l}`]);
      const hh = a.map(v => (v > 0 ? v * v : 0)); acts.push(a);
      const mo = this.mv(`w2_${l}`, hh, d, c.m); for (let k = 0; k < d; k++) x[k] += mo[k];
    }
    const logits = this.mv("lm_head", this.rms(x), c.V, d, this.m.T.lm_bias);
    this.t++; return { logits, acts };
  }
}

// ------------------------------------------------------------------------------------------------ WebGPU backend
const WGSL = /* wgsl */`
struct P { rows: u32, cols: u32, flags: u32, t: u32, ds: u32, tok: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> Bv: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
fn q8(w: u32, k: u32) -> f32 { return f32(extractBits(bitcast<i32>(w), k * 8u, 8u)); }
// Y[r] (+)= S[r] * dot(W[r,:], X) (+ Bv[r]);  flags: 1 = add bias, 2 = accumulate into Y
@compute @workgroup_size(64) fn matvec(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x + g.y * 65535u * 64u; if (r >= p.rows) { return; }
  let wpr = p.cols / 4u; var acc = 0.0;
  for (var i = 0u; i < wpr; i++) {
    let w = W[r * wpr + i]; let b = i * 4u;
    acc += q8(w, 0u) * X[b] + q8(w, 1u) * X[b + 1u] + q8(w, 2u) * X[b + 2u] + q8(w, 3u) * X[b + 3u];
  }
  var y = acc * S[r];
  if ((p.flags & 1u) != 0u) { y += Bv[r]; }
  if ((p.flags & 2u) != 0u) { y += Y[r]; }
  Y[r] = y;
}
// Y = row p.tok of the int8 embedding (dequantised)
@compute @workgroup_size(64) fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; if (k >= p.cols) { return; }
  let w = W[(p.tok * p.cols + k) / 4u]; Y[k] = q8(w, k % 4u) * S[p.tok];
}
var<workgroup> red: array<f32, 256>;
// Y = X / rms(X), single workgroup, p.cols elements
@compute @workgroup_size(256) fn rmsnorm(@builtin(local_invocation_id) l: vec3<u32>) {
  var s = 0.0; for (var i = l.x; i < p.cols; i += 256u) { s += X[i] * X[i]; }
  red[l.x] = s; workgroupBarrier();
  for (var o = 128u; o > 0u; o >>= 1u) { if (l.x < o) { red[l.x] += red[l.x + o]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(p.cols) + 1e-6);
  for (var i = l.x; i < p.cols; i += 256u) { Y[i] = X[i] * r; }
}
// h = relu(a)^2 elementwise (X = a, Y = h)
@compute @workgroup_size(64) fn relu2(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.cols) { return; } let a = X[i]; Y[i] = select(0.0, a * a, a > 0.0);
}
`;
const WGSL_ATTN = /* wgsl */`
struct P { rows: u32, cols: u32, flags: u32, t: u32, ds: u32, tok: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read> CS: array<f32>;      // cos table then sin table, (T, hd/2) each
@group(0) @binding(2) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> Q: array<f32>;
@group(0) @binding(5) var<uniform> p: P;                       // cols = d, rows = H, n = T (table rows)
@compute @workgroup_size(64) fn rope(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = p.cols; let hd = d / p.rows; let h2 = hd / 2u; let idx = g.x; if (idx >= p.rows * h2) { return; }
  let h = idx / h2; let i = idx % h2; let b = h * hd; let t = p.t;
  let co = CS[t * h2 + i]; let si = CS[p.n * h2 + t * h2 + i];
  let q0 = QKV[b + i]; let q1 = QKV[b + i + h2]; let k0 = QKV[d + b + i]; let k1 = QKV[d + b + i + h2];
  Q[b + i] = q0 * co - q1 * si; Q[b + i + h2] = q0 * si + q1 * co;
  Kc[t * d + b + i] = k0 * co - k1 * si; Kc[t * d + b + i + h2] = k0 * si + k1 * co;
  Vc[t * d + b + i] = QKV[2u * d + b + i]; Vc[t * d + b + i + h2] = QKV[2u * d + b + i + h2];
}
`;
const WGSL_ATTN2 = /* wgsl */`
struct P { rows: u32, cols: u32, flags: u32, t: u32, ds: u32, tok: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> Kc: array<f32>;
@group(0) @binding(2) var<storage, read> Vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Yo: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 64>;
// one workgroup per head; keys j in [ds, t]
@compute @workgroup_size(64) fn attend(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let d = p.cols; let hd = d / p.rows; let b = wg.x * hd; let n = p.t - p.ds + 1u; let s = inverseSqrt(f32(hd));
  var mx = -1e30;
  for (var j = l.x; j < n; j += 64u) {
    var a = 0.0; let kk = (p.ds + j) * d + b;
    for (var i = 0u; i < hd; i++) { a += Q[b + i] * Kc[kk + i]; }
    sc[j] = a * s; mx = max(mx, a * s);
  }
  red[l.x] = mx; workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) { if (l.x < o) { red[l.x] = max(red[l.x], red[l.x + o]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var z = 0.0; for (var j = l.x; j < n; j += 64u) { let e = exp(sc[j] - m); sc[j] = e; z += e; }
  red[l.x] = z; workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) { if (l.x < o) { red[l.x] += red[l.x + o]; } workgroupBarrier(); }
  let zt = red[0];
  for (var i = l.x; i < hd; i += 64u) {
    var acc = 0.0; for (var j = 0u; j < n; j++) { acc += sc[j] * Vc[(p.ds + j) * d + b + i]; }
    Yo[b + i] = acc / zt;
  }
}
`;
export class GPUModel {
  static async create(model) {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error("no WebGPU adapter");
    const dev = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 << 20) } });
    return new GPUModel(model, dev);
  }
  constructor(model, dev) {
    this.m = model; this.dev = dev; const c = (this.c = model.cfg); this.kind = "webgpu"; this.hd = c.d / c.H;
    const U = GPUBufferUsage; this.B = {};
    const mk = (size, usage, data) => {
      const b = dev.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, mappedAtCreation: !!data });
      if (data) { new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); }
      return b;
    };
    for (const t of model.tensors) this.B[t.name] = mk(t.bytes, U.STORAGE, model.T[t.name]);
    const f = (n) => mk(n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    this.x = f(c.d); this.ln = f(c.d); this.qkv = f(3 * c.d); this.q = f(c.d); this.y = f(c.d);
    this.a = Array.from({ length: c.L }, () => f(c.m)); this.h = f(c.m); this.logits = f(c.Vp);
    this.K = Array.from({ length: c.L }, () => f(c.T * c.d)); this.V = Array.from({ length: c.L }, () => f(c.T * c.d));
    const rt = ropeTables(c.T, this.hd), cs = new Float32Array(rt.cos.length * 2); cs.set(rt.cos); cs.set(rt.sin, rt.cos.length);
    this.cs = mk(cs.byteLength, U.STORAGE, cs); this.dummy = f(4);
    this.readLogits = dev.createBuffer({ size: c.Vp * 4, usage: U.MAP_READ | U.COPY_DST });
    this.readActs = dev.createBuffer({ size: c.L * c.m * 4, usage: U.MAP_READ | U.COPY_DST });
    const mod = dev.createShaderModule({ code: WGSL }), modA = dev.createShaderModule({ code: WGSL_ATTN }), modB = dev.createShaderModule({ code: WGSL_ATTN2 });
    const pipe = (m, e) => dev.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: e } });
    this.P = { matvec: pipe(mod, "matvec"), embed: pipe(mod, "embed"), rmsnorm: pipe(mod, "rmsnorm"), relu2: pipe(mod, "relu2"), rope: pipe(modA, "rope"), attend: pipe(modB, "attend") };
    this.uni = []; this.reset();
  }
  reset() { this.t = 0; this.ds = 0; }
  _u(vals) {   // small uniform buffer per dispatch
    const b = this.dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dev.queue.writeBuffer(b, 0, new Uint32Array([vals.rows || 0, vals.cols || 0, vals.flags || 0, vals.t || 0, vals.ds || 0, vals.tok || 0, vals.n || 0, 0]));
    this.uni.push(b); return b;
  }
  _disp(pass, pipe, bufs, uni, wx, wy = 1) {
    // layout "auto" only contains the bindings an entry point uses, so unused slots are passed as null and skipped
    const entries = bufs.map((b, i) => (b ? { binding: i, resource: { buffer: b } } : null)).filter(Boolean); entries.push({ binding: bufs.length, resource: { buffer: uni } });
    pass.setPipeline(pipe); pass.setBindGroup(0, this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries })); pass.dispatchWorkgroups(wx, wy);
  }
  _mv(pass, name, X, Y, rows, cols, flags, bias) {
    const wg = Math.ceil(rows / 64), wx = Math.min(wg, 65535), wy = Math.ceil(wg / 65535);
    this._disp(pass, this.P.matvec, [this.B[name], this.B[name + ".scale"], X, Y, bias || this.dummy], this._u({ rows, cols, flags }), wx, wy);
  }
  async step(tok, eot, wantLogits = true) {
    const c = this.c, d = c.d, t = this.t; if (tok === eot) this.ds = t;
    const enc = this.dev.createCommandEncoder(), pass = enc.beginComputePass();
    this._disp(pass, this.P.embed, [this.B.wte, this.B["wte.scale"], null, this.x, null], this._u({ cols: d, tok }), Math.ceil(d / 64));
    for (let l = 0; l < c.L; l++) {
      this._disp(pass, this.P.rmsnorm, [null, null, this.x, this.ln, null], this._u({ cols: d }), 1);
      this._mv(pass, `wqkv${l}`, this.ln, this.qkv, 3 * d, d, 0);
      this._disp(pass, this.P.rope, [this.qkv, this.cs, this.K[l], this.V[l], this.q], this._u({ rows: c.H, cols: d, t, n: c.T }), Math.ceil(c.H * this.hd / 2 / 64));
      this._disp(pass, this.P.attend, [this.q, this.K[l], this.V[l], this.y], this._u({ rows: c.H, cols: d, t, ds: this.ds }), c.H);
      this._mv(pass, `wo${l}`, this.y, this.x, d, d, 2);
      this._disp(pass, this.P.rmsnorm, [null, null, this.x, this.ln, null], this._u({ cols: d }), 1);
      this._mv(pass, `w1_${l}`, this.ln, this.a[l], c.m, d, 1, this.B[`b1_${l}`]);
      this._disp(pass, this.P.relu2, [null, null, this.a[l], this.h, null], this._u({ cols: c.m }), Math.ceil(c.m / 64));
      this._mv(pass, `w2_${l}`, this.h, this.x, d, c.m, 2);
    }
    if (wantLogits) {
      this._disp(pass, this.P.rmsnorm, [null, null, this.x, this.ln, null], this._u({ cols: d }), 1);
      this._mv(pass, "lm_head", this.ln, this.logits, c.Vp, d, 1, this.B.lm_bias);
    }
    pass.end();
    if (wantLogits) enc.copyBufferToBuffer(this.logits, 0, this.readLogits, 0, c.Vp * 4);
    for (let l = 0; l < c.L; l++) enc.copyBufferToBuffer(this.a[l], 0, this.readActs, l * c.m * 4, c.m * 4);
    this.dev.queue.submit([enc.finish()]);
    let logits = null;
    if (wantLogits) { await this.readLogits.mapAsync(GPUMapMode.READ); logits = new Float32Array(this.readLogits.getMappedRange().slice(0, c.V * 4)); this.readLogits.unmap(); }
    await this.readActs.mapAsync(GPUMapMode.READ); const all = new Float32Array(this.readActs.getMappedRange().slice(0)); this.readActs.unmap();
    for (const b of this.uni) b.destroy(); this.uni = [];
    const acts = Array.from({ length: c.L }, (_, l) => all.subarray(l * c.m, (l + 1) * c.m));
    this.t++; return { logits, acts };
  }
}

export function sample(logits, temperature = 0.8, topk = 40, rng = Math.random) {
  if (temperature <= 0) { let b = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i; return b; }
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topk);
  const mx = logits[idx[0]]; const p = idx.map(i => Math.exp((logits[i] - mx) / temperature)); const z = p.reduce((a, b) => a + b, 0);
  let r = rng() * z; for (let k = 0; k < idx.length; k++) { r -= p[k]; if (r <= 0) return idx[k]; } return idx[idx.length - 1];
}
