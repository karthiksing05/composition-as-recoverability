import json, glob, sys, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "results/e14"
rows = [r for f in glob.glob(f"{d}/*.json") for r in json.load(open(f, encoding="utf-8"))]
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
short = lambda m: m.split("/")[-1]
df["model"] = df.model.map(short)
# layer per (model, seed, emb) chosen on validation pairs, mean over relations
v = df[df.split == "val"].groupby(["model", "seed", "emb", "layer"]).pair.mean().reset_index()
best = v.loc[v.groupby(["model", "seed", "emb"]).pair.idxmax(), ["model", "seed", "emb", "layer"]]
f = df[df.split == "fresh"].merge(best, on=["model", "seed", "emb", "layer"])
cols = ["pair", "top1", "beats_r1", "beats_r2"]
print(f.groupby(["model", "emb"])[cols].mean().round(3).to_string())
print("\nby relation (pair):")
print(f.groupby(["model", "emb", "rel"]).pair.mean().unstack("rel").round(2).to_string())
print("\nchosen layers:", best.groupby(["model", "emb"]).layer.agg(lambda s: ",".join(s)).to_string())
print("\nseeds:", df.groupby("model").seed.nunique().to_dict())
