import json, sys; sys.path.insert(0, "tools"); from apiclient import jev, spend
st = "def add(a, b):\n    return a + b  # simple helper used by the billing module"
q3 = {"type": "choice", "instructions": "What kind of text is this?", "criteria": {"code": "source code", "prose": "natural language prose", "math": "mathematical notation"}}
r1 = jev(st, {"kind": q3}); print("1q:", r1["usage"], r1["answers"]["kind"]["choice"], r1["answers"]["kind"]["probabilities"])
r2 = jev(st, {"kind": q3, "lang": {"type": "noul", "instructions": "Is this Python code?"}}); print("2q:", r2["usage"], r2["answers"]["lang"])
opts = {f"c{i}": None for i in range(200)}; opts["c7"] = "source code in a programming language"
r3 = jev(st, {"big": {"type": "choice", "instructions": "Which concept applies?", "criteria": opts}}); a = r3["answers"]["big"]
print("200opt-null:", r3["usage"], a["choice"], round(a["confidence"], 3))
print(json.dumps(spend()))
