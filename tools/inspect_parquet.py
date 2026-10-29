import sys, pyarrow.parquet as pq
for path in sys.argv[1:]:
    f = pq.ParquetFile(path)
    print("==", path)
    print("schema:", f.schema_arrow)
    print("rows:", f.metadata.num_rows, "row_groups:", f.metadata.num_row_groups)
    t = f.read_row_group(0)
    print("rg0 rows:", t.num_rows)
    for name in t.column_names:
        col = t.column(name)
        v = col[0].as_py()
        if isinstance(v, (bytes, bytearray)): s = f"<bytes len={len(v)}> {v[:16]!r}"
        elif isinstance(v, dict): s = {k: (f"<bytes {len(x)}>" if isinstance(x,(bytes,bytearray)) else x) for k,x in v.items()}
        elif isinstance(v, list): s = f"list len={len(v)} first={v[:20]}"
        else: s = repr(v)[:300]
        print(f"  {name}: {s}")
