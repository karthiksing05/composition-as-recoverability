import json, glob, sys, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "results/e15"
rows = [r for f in glob.glob(f"{d}/*.json") for r in json.load(open(f, encoding="utf-8"))]
df = pd.DataFrame(rows)
df["base"] = df.get("base", pd.Series(index=df.index, dtype=object)).fillna("e2")
pd.set_option("display.width", 250)
print("files", len(glob.glob(f"{d}/*.json")))
for col in ("role1", "role2", "newnew", "heldout"):
    print(f"\n== {col}")
    print(df.groupby(["base", "method", "k"])[col].mean().unstack("k").round(2).to_string())
print("\nby arch at k=32:")
print(df[df.k == 32].groupby(["base", "arch", "method"])[["role1", "role2", "newnew", "heldout"]].mean().round(2).to_string())
