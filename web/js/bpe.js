// GPT-2 byte-level BPE (tiktoken algorithm: repeatedly merge the adjacent pair with the lowest rank).
export class BPE {
  constructor(data) {
    this.eot = data.eot;
    this.bytes = data.tokens.map(b => b ? Uint8Array.from(atob(b), c => c.charCodeAt(0)) : new Uint8Array(0));
    this.rank = new Map();
    this.bytes.forEach((b, i) => { if (b.length) this.rank.set(latin1(b), i); });
    // original GPT-2 pre-tokenizer (tiktoken's possessive quantifiers are not valid in JS regex)
    this.re = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.te = new TextEncoder(); this.td = new TextDecoder();
  }
  encode(text) {
    const out = [];
    for (const m of text.matchAll(this.re)) {
      const s = latin1(this.te.encode(m[0]));
      const whole = this.rank.get(s);
      if (whole !== undefined) { out.push(whole); continue; }
      let parts = Array.from(s);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.rank.get(parts[i] + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        parts.splice(best, 2, parts[best] + parts[best + 1]);
      }
      for (const p of parts) out.push(this.rank.get(p));
    }
    return out;
  }
  decodeBytes(ids) {
    let n = 0; for (const i of ids) n += this.bytes[i].length;
    const buf = new Uint8Array(n); let o = 0;
    for (const i of ids) { buf.set(this.bytes[i], o); o += this.bytes[i].length; }
    return buf;
  }
  decode(ids) { return this.td.decode(this.decodeBytes(ids)); }
  piece(id) { return id === this.eot ? "⟨eot⟩" : this.td.decode(this.bytes[id]); }
}
function latin1(bytes) { let s = ""; for (const b of bytes) s += String.fromCharCode(b); return s; }
