import json, glob, collections, numpy as np
rows = []
for f in glob.glob("results/e4w/*_n165_s*.json"):
    rows += json.load(open(f, encoding="utf-8"))
rows = [r for r in rows if r["n_fit"] == 165 and r["codes_from"] == r["model"] and not r.get("detok")]
asym = {"on", "under", "left", "right", "above", "behind", "in", "near"}
by = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["layer"], r["code"], r["system"])].append(r)
models = sorted({r["model"] for r in rows})
print("rels:", sorted({r["rel"] for r in rows}))
print(f"{'model':28s} {'layer':6s} {'code':7s} rec1  rec2  delta_fr inter  | concat rec1/2 | delta_fr(concat)")
for m in models:
    cands = [(k, v) for k, v in by.items() if k[0] == m and k[3] == "model"]
    def score(v): return np.mean([(x["rec_val_role1"] + x["rec_val_role2"]) / 2 for x in v if x["rel"] != "and"])
    k, v = max(cands, key=lambda kv: score(kv[1]))
    a = [x for x in v if x["rel"] != "and"]
    ref = [x for x in by[(m, k[1], k[2], "ref_concat")] if x["rel"] != "and"]
    mean = lambda L, key: np.mean([x[key] for x in L]) if L else float("nan")
    print(f"{m:28s} {k[1]:6s} {k[2]:7s} {mean(a,'rec_fresh_role1'):.2f}  {mean(a,'rec_fresh_role2'):.2f}  {mean(a,'delta_fresh'):.2f}     {mean(a,'inter'):.3f}  | {mean(ref,'rec_fresh_role1'):.2f}/{mean(ref,'rec_fresh_role2'):.2f} | {mean(ref,'delta_fresh'):.2f}")
