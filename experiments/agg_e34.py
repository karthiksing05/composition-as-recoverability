"""Aggregate E34.  usage: agg_e34.py [results dir]
Per (system, condition, readout): the (layer[, timestep], code) with the best validation R2 per seed; kappa on fresh photographs there."""
import glob, json, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
d = sys.argv[1] if len(sys.argv) > 1 else "results/e34"
rows, meta = [], []
for f in sorted(glob.glob(f"{d}/*_s[0-9].json")):
    for r in json.load(open(f, encoding="utf-8")):
        (meta if r["cond"] == "meta" else rows).append(r)
df = pd.DataFrame(rows)
df["rec_val"] = df[["rec_val_role1", "rec_val_role2"]].mean(1)
df["role_acc"] = df[["role_fresh_role1", "role_fresh_role2"]].mean(1)
df["id_acc"] = df[["id_fresh_role1", "id_fresh_role2"]].mean(1)
out = []
for (m, s, c, rd), g in df.groupby(["model", "seed", "cond", "readout"]):
    b = g.loc[g.rec_val.idxmax()]
    out.append(dict(model=m, seed=s, cond=c, readout=rd, layer=b.layer, code=b.code, dim=b.dim, kappa=b.kappa_mean,
                    kappa_fig=b.kappa_role1, kappa_gnd=b.kappa_role2, role_acc=b.role_acc, id_acc=b.id_acc, rec_val=b.rec_val))
S = pd.DataFrame(out)
S.to_csv(f"{d}/agg_per_seed.csv", index=False)
T = S.groupby(["model", "cond", "readout"]).agg(kappa=("kappa", "mean"), kappa_sd=("kappa", "std"), kappa_fig=("kappa_fig", "mean"),
        kappa_gnd=("kappa_gnd", "mean"), role_acc=("role_acc", "mean"), id_acc=("id_acc", "mean"), n=("seed", "nunique"),
        layer=("layer", lambda x: ",".join(sorted(set(x))))).reset_index()
T.to_csv(f"{d}/agg_kappa.csv", index=False)
print(T.round(3).to_string(index=False))
print("=== kappa table (mean over seeds; sd in brackets)")
for cond in ("rand", "fixed"):
    P = T[T.cond == cond]
    print(cond)
    print(P.assign(v=P.kappa.round(2).astype(str) + " (" + P.kappa_sd.round(2).astype(str) + ")").pivot(index="model", columns="readout", values="v").to_string())
print("=== role accuracy (chance .5)")
for cond in ("rand", "fixed"):
    print(cond); print(T[T.cond == cond].pivot(index="model", columns="readout", values="role_acc").round(3).to_string())
M = pd.DataFrame(meta)
print(M.groupby("model")[["loc_only_role_acc", "mean_gap"]].mean())
