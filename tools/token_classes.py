"""Token-level abstractions (multilevel binning): gpt-6-luna proposes 64 token classes for this corpus's GPT-2 BPE
tokens; Jev bins each of the top-N tokens (state = token + 2 example contexts) into one class.
Rare tokens inherit the class of their nearest frequent token in the distributional (bigram-SVD) space.
usage: token_classes.py propose | bin <N> | finalize"""
import json, sys, os, numpy as np, tiktoken
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "tools"); from apiclient import openai_chat, jev, spend
enc = tiktoken.get_encoding("gpt2"); cnt = np.load("stats/unigram.npy").astype(float); order = np.argsort(-cnt)
tok = np.memmap("data_bin/train.u16", dtype=np.uint16, mode="r")
def show(t): return json.dumps(enc.decode([int(t)]))   # JSON-quoted so spaces / newlines are visible
def contexts(ids, n=2, span=8, scan=30_000_000):
    want = {int(t): [] for t in ids}; s = np.asarray(tok[:scan])
    for t in want:
        pos = np.flatnonzero(s == t)[:200]
        for p in pos[::max(1, len(pos) // n)][:n]:
            a, b = max(0, p - span), min(len(s), p + span + 1)
            want[t].append(enc.decode(s[a:p].tolist()) + "⟦" + enc.decode([t]) + "⟧" + enc.decode(s[p + 1:b].tolist()))
    return want
cmd = sys.argv[1]
if cmd == "propose":
    rng = np.random.default_rng(0)
    pick = np.concatenate([order[:300], order[300:3000][rng.choice(2700, 250, replace=False)], order[3000:16000][rng.choice(13000, 250, replace=False)]])
    ctx = contexts(pick[::4], n=1)
    lines = [f"{show(t)}  rank={int(np.flatnonzero(order == t)[0])}" + (f"  e.g. {json.dumps(ctx[int(t)][0])}" if int(t) in ctx and ctx[int(t)] else "") for t in pick]
    prompt = f"""We are constructing the first layer of a small language model by hand. Each GPT-2 BPE token of an LLM-pretraining
web corpus (educational web text, QA, math, code) will be binned into exactly ONE of 64 token classes, and the model
starts as a class-based bigram model:  p(next | token) = p(class(next) | class(token)) * p(next | class(next)).
Design the 64 classes so that this model predicts well: a token's class should capture its distributional role
(what tends to come before and after it), e.g. word-initial vs word-continuation pieces, part of speech, number and
digit pieces, punctuation and brackets by role, code syntax, math notation, whitespace/newlines, capitalised
sentence starts, names, units, etc. Classes must be mutually exclusive, jointly exhaustive, judgeable from the token
string plus a couple of example contexts, and reasonably balanced by token occurrence. Include a catch-all.
Each class: "id" (snake_case, <=4 words, self-explanatory) and "desc" (<=12 words).
Return JSON: {{"classes": [{{"id": "...", "desc": "..."}}, ...]}} with exactly 64 entries.

Sample of tokens (JSON-quoted, so a leading space is visible), with frequency rank and an example context
(the token is marked with ⟦ ⟧):
""" + "\n".join(lines)
    r = openai_chat([{"role": "system", "content": "You are an expert computational linguist. Output only JSON."}, {"role": "user", "content": prompt}],
                    model="gpt-6-luna", response_format={"type": "json_object"})
    cl = json.loads(r["choices"][0]["message"]["content"])["classes"]
    json.dump(cl, open("labels/token_classes.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(cl), "classes:", ", ".join(c["id"] for c in cl)); print("usage", r["usage"]); print(json.dumps(spend()))
elif cmd == "bin":
    N = int(sys.argv[2]); cl = json.load(open("labels/token_classes.json", encoding="utf-8")); ids = order[:N]
    ctx = contexts(ids, n=2)
    q = {"cls": {"type": "choice", "instructions": "Which class best describes the role of this GPT-2 token (marked ⟦ ⟧ in the examples)?",
                 "criteria": {c["id"]: None for c in cl}}}
    def one(t):
        r = jev({"token": enc.decode([int(t)]), "examples": ctx[int(t)]}, q); pr = r["answers"]["cls"]["probabilities"]
        return np.array([pr.get(c["id"], 0.0) for c in cl], dtype=np.float32), r["usage"]["input_tokens"]
    if len(sys.argv) > 3:   # cost estimate on a sample
        res = [one(t) for t in ids[:int(sys.argv[3])]]
        print("mean input tokens", np.mean([x[1] for x in res]), "est usd for N", np.mean([x[1] for x in res]) * N * 0.04e-6); sys.exit()
    with ThreadPoolExecutor(16) as ex: res = list(ex.map(one, ids))
    P = np.stack([x[0] for x in res]); np.savez("labels/token_classes_jev.npz", ids=ids, P=P)
    print("DONE binned", N, "tokens; mean input tokens", np.mean([x[1] for x in res]), json.dumps(spend()))
elif cmd == "finalize":   # full-vocab class array: Jev argmax for top-N, nearest frequent neighbour otherwise
    z = np.load("labels/token_classes_jev.npz"); ids, P = z["ids"], z["P"]
    sv = np.load("stats/bigram_svd.npz"); F = np.concatenate([sv["L"][:, :64], sv["R"][:, :64]], 1); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    cls = np.full(50257, -1); cls[ids] = P.argmax(1)
    rest = np.flatnonzero(cls < 0); nn = (F[rest] @ F[ids].T).argmax(1); cls[rest] = cls[ids][nn]
    np.save("labels/token_classes_jev_full.npy", cls); print("classes used", len(np.unique(cls)), "sizes", np.bincount(cls).tolist()[:64])
