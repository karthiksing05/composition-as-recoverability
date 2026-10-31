"""Aggregate e2_redecode: fresh-glyph output R^2 with the encoder frozen and the decoder retrained."""
import glob, json, numpy as np, pandas as pd
K = ["r00", "r01", "r10", "r11"]
rows = []
for f in glob.glob("results/e2redec/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    for c in ("orig", "redec_trained", "redec_init", "redec_same"):
        rows.append(dict(arch=r["arch"], seed=r["seed"], cond=c, all=np.mean([r[c][k] for k in K]), over_in=np.mean([r[c]["r10"], r[c]["r11"]])))
for f in glob.glob("results/e2v2/*_m64_V64_c1_s*.json") + glob.glob("results/e2v2/*_m64_V8_c1_s*.json"):
    r = json.load(open(f, encoding="utf-8"))
    rows.append(dict(arch=r["arch"], seed=r["seed"], cond=f"e2e_V{r['V']}", all=np.mean([r[f"beh_fresh_{k}"] for k in K]),
                     over_in=np.mean([r["beh_fresh_r10"], r["beh_fresh_r11"]])))
d = pd.DataFrame(rows)
order = ["e2e_V8", "orig", "redec_same", "redec_trained", "redec_init", "e2e_V64"]
for key in ("all", "over_in"):
    t = d.groupby(["cond", "arch"])[key].agg(["mean", "sem"]).unstack("arch")
    print(f"\n== fresh-glyph output R^2 ({key})"); print(t.reindex(order).round(3).to_string())
    a = d[d.arch != "tf"].groupby(["cond", "seed"])[key].mean().groupby("cond").agg(["mean", "sem"])
    print("mlp+cnn+slot:"); print(a.reindex(order).round(3).to_string())
