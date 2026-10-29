// Node test: browser CPU engine (int8) vs NumPy fp64 reference logits; BPE tokenizer vs tiktoken.
import fs from "fs";
import { CPUModel } from "./js/engine.js";
import { BPE } from "./js/bpe.js";
const dir = "web/models/test", meta = JSON.parse(fs.readFileSync(dir + "/meta.json"));
const buf = new Uint8Array(meta.total_bytes); let off = 0;
for (const c of meta.chunks) { const b = fs.readFileSync(dir + "/" + c); buf.set(b, off); off += b.length; }
const T = {}; for (const t of meta.tensors) T[t.name] = t.kind === "int8" ? new Int8Array(buf.buffer, t.offset, t.bytes) : new Float32Array(buf.buffer, t.offset, t.bytes / 4);
const model = new CPUModel({ meta, cfg: meta.cfg, T, tensors: meta.tensors });
const ref = JSON.parse(fs.readFileSync("runs/ref_test.json"));
let maxd = 0, agree = 0, t0 = Date.now(), actd = 0;
for (let i = 0; i < ref.tokens.length; i++) {
  const { logits, acts } = await model.step(ref.tokens[i], 50256);
  let am = 0, bm = 0; for (let j = 0; j < 50257; j++) { maxd = Math.max(maxd, Math.abs(logits[j] - ref.logits[i][j])); if (logits[j] > logits[am]) am = j; if (ref.logits[i][j] > ref.logits[i][bm]) bm = j; }
  agree += am === bm;
  if (i < ref.acts0.length) for (let j = 0; j < acts[0].length; j++) actd = Math.max(actd, Math.abs(acts[0][j] - ref.acts0[i][j]));
}
console.log(`CPU engine vs fp64 reference: max |dlogit| = ${maxd.toFixed(4)}, argmax agreement ${agree}/${ref.tokens.length}, max |d act0| = ${actd.toFixed(4)}, ${((Date.now() - t0) / ref.tokens.length).toFixed(0)} ms/token`);
const bpe = new BPE(JSON.parse(fs.readFileSync("web/assets/gpt2_bpe.json")));
for (const t of JSON.parse(fs.readFileSync("web/assets/bpe_tests.json"))) {
  const ids = bpe.encode(t.text); const ok = JSON.stringify(ids) === JSON.stringify(t.ids);
  console.log(`BPE ${ok ? "OK  " : "FAIL"} ${JSON.stringify(t.text).slice(0, 40)} ${ok ? "" : JSON.stringify(ids) + " vs " + JSON.stringify(t.ids)}`);
}
