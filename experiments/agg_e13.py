import json, glob, sys, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "results/e13"
rows = [r for f in glob.glob(f"{d}/*.json") for r in json.load(open(f, encoding="utf-8"))]
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("files", len(glob.glob(f"{d}/*.json")))
for col in ("fresh_r2", "pop_r2", "fam_r2"):
    print(f"\n== {col} (mean over seeds; n per cell in brackets)")
    t = df.groupby(["w", "cond", "depth"])[col].mean().unstack("depth").round(2)
    print(t.to_string())
print("\nn seeds:", df.groupby(["w", "cond"]).seed.nunique().unstack().to_string())
