import json, glob, pandas as pd
rows = []
for f in glob.glob("results/e3/*.json"):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
r = df[df.model == "raam"].groupby(["V", "w", "depth"])[["fam_id", "fresh_id", "fresh_r2", "bound"]].mean().round(2)
print(r.unstack("depth").to_string())
h = df[df.model == "hrr"].groupby(["V", "w", "split", "depth"])[["r2", "id"]].mean().round(2)
print(h.unstack("depth").to_string())
