# Convert ClimbMix tokenized parquet shards -> doc-shuffled uint16 token streams for the CUDA trainer.
# Layout: each doc is emitted as [EOT, tok0, tok1, ...]; docs.int64 holds the start offset of each doc (its EOT).
import numpy as np, pyarrow.parquet as pq, json, sys, time
EOT = 50256
shards = ["data/climbmix_shard_0.tokenized.parquet", "data/climbmix_shard_1.tokenized.parquet"]
VAL_TOKENS = 4_000_000
t0 = time.time()
toks, lens = [], []
for s in shards:
    col = pq.read_table(s, columns=["tokens"]).column("tokens").combine_chunks()
    flat = col.values.to_numpy()
    assert flat.max() < 65536 and flat.min() >= 0
    n_eot = int((flat == EOT).sum())
    toks.append(flat.astype(np.uint16)); lens.append(np.diff(col.offsets.to_numpy()))
    print(f"loaded {s} docs={len(lens[-1])} toks={len(flat)} max={flat.max()} eot_in_data={n_eot} t={time.time()-t0:.0f}s", flush=True)
    del col, flat
tok = np.concatenate(toks); del toks
lens = np.concatenate(lens).astype(np.int64)
starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
perm = np.random.default_rng(1234).permutation(len(lens))
cum = np.cumsum(lens[perm] + 1)
n_val = int(np.searchsorted(cum, VAL_TOKENS)) + 1
def write(name, docs):
    out_lens = lens[docs] + 1
    out_starts = np.concatenate([[0], np.cumsum(out_lens)[:-1]])
    out = np.empty(int(out_lens.sum()), dtype=np.uint16)
    for d, o in zip(docs, out_starts):
        out[o] = EOT
        out[o + 1:o + 1 + lens[d]] = tok[starts[d]:starts[d] + lens[d]]
    out.tofile(f"data_bin/{name}.u16"); out_starts.astype(np.int64).tofile(f"data_bin/{name}_docs.i64")
    print(f"wrote {name}: tokens={len(out)} docs={len(docs)} t={time.time()-t0:.0f}s", flush=True)
    return len(out), len(docs)
nv = write("val", perm[:n_val]); nt = write("train", perm[n_val:])
json.dump({"tokenizer": "gpt2", "eot": EOT, "vocab": 50257, "seed": 1234, "shards": shards,
           "val_tokens": nv[0], "val_docs": nv[1], "train_tokens": nt[0], "train_docs": nt[1]},
          open("data_bin/meta.json", "w"), indent=1)
print("DONE", flush=True)
