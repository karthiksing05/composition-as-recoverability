"""Aggregate E11: role readouts from token-level states (mean, last, nouns, query)."""
import json, glob
import numpy as np, pandas as pd


def short(df):
    for pre in ("EleutherAI/", "allenai/", "microsoft/", "Qwen/"):
        df["model"] = df.model.str.replace(pre, "")
    return df

rows = []
for f in glob.glob("results/e11/*.json"):
    rows += json.load(open(f, encoding="utf-8"))
d = short(pd.DataFrame(rows))
key = ["model", "seed", "rel", "layer", "code"]
ref = d[d.system != "model"].pivot_table(index=key, columns="system", values=["sec_rec_fresh", "pos_rec_fresh"])
ref.columns = [f"{a}|{b}" for a, b in ref.columns]
M = d[d.system == "model"].set_index(key).join(ref).reset_index()
M["kappa2"] = (M.sec_rec_fresh - M["sec_rec_fresh|ref_sum"]) / (M["sec_rec_fresh|ref_pos"] - M["sec_rec_fresh|ref_sum"])
M["kappa1"] = (M.pos_rec_fresh - M["pos_rec_fresh|ref_sum"]) / (M["pos_rec_fresh|ref_pos"] - M["pos_rec_fresh|ref_sum"])
M["sel"] = M.sem_rec_val + M.xfer_rec_val
out = []
for (m, st), g in M.groupby(["model", "state"]):
    lc = g.groupby(["layer", "code"]).sel.mean().idxmax()
    gg = g.set_index(["layer", "code"]).loc[lc]
    out.append(dict(model=m, state=st, layer=lc[0], code=lc[1], seeds=gg.seed.nunique(), kappa1=gg.kappa1.mean(), kappa2=gg.kappa2.mean(),
                    sem_rec=gg.sem_rec_fresh.mean(), sem_role_conv=gg.sem_role_conv.mean(),
                    xfer_role_can=gg.xfer_role_can.mean(), xfer_role_conv=gg.xfer_role_conv.mean(), xfer_rec_conv=gg.xfer_rec_conv.mean()))
pd.set_option("display.width", 220)
print(pd.DataFrame(out).set_index(["model", "state"]).round(3).to_string())
