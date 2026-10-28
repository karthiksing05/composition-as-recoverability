"""E1 robustness: fresh-part R^2 by part distribution, readout, and V (results/e1rob, results/e1)."""
import json, glob, pandas as pd
rows = []
for d in ("results/e1", "results/e1rob"):
    for f in glob.glob(f"{d}/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
df = pd.DataFrame(rows)
df["dist"] = df.get("dist", pd.Series(["gauss"] * len(df))).fillna("gauss")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
d = df[(df.readout != "nn_table") & df.V.isin([8, 32, 128])]
print(d.pivot_table(index=["op"], columns=["dist", "readout", "V"], values="fresh_r2").round(2).to_string())
