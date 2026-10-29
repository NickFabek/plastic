"""Ask gpt-6-luna to propose the neuron basis (abstractions) from a sample of the training data."""
import json, sys; sys.path.insert(0, "tools"); from apiclient import openai_chat, spend
sample = json.load(open("labels/taxonomy_sample.json", encoding="utf-8"))
blob = "\n".join(f"[{i}] " + s["text"][:420].replace("\n", " ") for i, s in enumerate(sample))
SYSTEM = "You are an expert in corpus linguistics and neural-network interpretability. Output only JSON."
USER = f"""We are building a small transformer language model whose MLP neurons will each be assigned ONE abstraction
(concept). A separate classifier will bin every ~120-token passage of the training data into exactly one concept per
facet, and the neuron for that concept is then constructed from the statistics of its bin (which words occur, which
tokens come next). Good concepts therefore (1) are mutually exclusive within a facet and jointly exhaustive for THIS
corpus, (2) change which tokens are likely when you know them (high mutual information with the next token),
(3) are roughly balanced in frequency: merge anything rarer than ~0.2% of passages into a broader concept, split
anything above ~4%, (4) are judgeable from a 120-token passage alone.

Design two facets:
- "topic": the subject matter. Exactly 160 fine concepts, organised into exactly 16 coarse groups of 10. The coarse
  groups should themselves be good abstractions for a model that only has 16 neurons.
- "format": genre / document type / register / structure (e.g. exam question with options, news report, forum reply,
  product listing, step-by-step instructions, source code, ...). Exactly 40 concepts.
Each concept: "id" (snake_case, <=4 words, self-explanatory) and "desc" (<=12 words, what distinguishes it).
Include one catch-all concept per facet. Base the taxonomy on the actual distribution in the sample below (the
corpus is an LLM-pretraining web mixture: educational web text, QA, code, etc.).

Return JSON: {{"topic": [{{"group": "...", "group_desc": "...", "concepts": [{{"id": "...", "desc": "..."}}, ...]}}, ...],
"format": [{{"id": "...", "desc": "..."}}, ...]}}

SAMPLE ({len(sample)} random passages, truncated):
{blob}"""
r = openai_chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
                model="gpt-6-luna", response_format={"type": "json_object"})
tax = json.loads(r["choices"][0]["message"]["content"])
json.dump(tax, open("labels/taxonomy_raw.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
nt = sum(len(g["concepts"]) for g in tax["topic"]); print("groups", len(tax["topic"]), "topic concepts", nt, "format", len(tax["format"]))
print("usage", r.get("usage")); print(json.dumps(spend()))
for g in tax["topic"]: print("-", g["group"], ":", ", ".join(c["id"] for c in g["concepts"]))
print("FORMAT:", ", ".join(c["id"] for c in tax["format"]))
