import json, sys, numpy as np; sys.path.insert(0, "tools"); from jev_label import label, TOPICS, FORMATS; from apiclient import spend
P = json.load(open("labels/passages.json", encoding="utf-8"))[:20]
A = [label(p["text"], "desc") for p in P]; B = [label(p["text"], "nodesc") for p in P]
for v, R in (("desc", A), ("nodesc", B)):
    print(v, "mean input tokens", np.mean([r[4] for r in R]), "mean conf topic", np.mean([r[2] for r in R]).round(3), "fmt", np.mean([r[3] for r in R]).round(3))
agree_t = np.mean([a[0].argmax() == b[0].argmax() for a, b in zip(A, B)]); agree_f = np.mean([a[1].argmax() == b[1].argmax() for a, b in zip(A, B)])
print("argmax agreement desc vs nodesc: topic", agree_t, "format", agree_f)
for p, a in list(zip(P, A))[:8]:
    print("---", p["text"][:150].replace("\n", " ")); print("   topic:", TOPICS[a[0].argmax()]["id"], round(float(a[0].max()), 2), "| format:", FORMATS[a[1].argmax()]["id"], round(float(a[1].max()), 2))
print(json.dumps(spend()))
