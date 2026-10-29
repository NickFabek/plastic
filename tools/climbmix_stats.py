import sys, pyarrow.parquet as pq, pyarrow.compute as pc, tiktoken
enc = tiktoken.get_encoding("gpt2")
for path in sys.argv[1:]:
    t = pq.read_table(path, columns=["token_count"])
    tc = t.column("token_count")
    print(path, "docs", t.num_rows, "tokens", pc.sum(tc).as_py(), "mean", round(pc.mean(tc).as_py(),1), "max", pc.max(tc).as_py())
t = pq.ParquetFile(sys.argv[1]).read_row_group(0, columns=["tokens"]).slice(0, 3)
for row in t.column("tokens").to_pylist():
    print("----", len(row)); print(enc.decode(row)[:600])
