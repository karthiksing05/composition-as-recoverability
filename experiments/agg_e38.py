"""Aggregate E38.  usage: agg_e38.py [results dir] [e34 dir]
kappa: per (model, cond, readout) the code with best validation R2 per seed (E34's rule); mean and SEM over seeds.
Prints kappa tables (controls, POOL next to E34's G2/Q4/M) and image-only matching; writes summary.json."""
import glob, json, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
d = sys.argv[1] if len(sys.argv) > 1 else "results/e38"
d34 = sys.argv[2] if len(sys.argv) > 2 else "results/e34"
rows, match = [], []
for f in sorted(glob.glob(f"{d}/*_s[0-9].json")):
    for r in json.load(open(f, encoding="utf-8")):
        if r["cond"] == "match":
            for c, v in r["match"].items():
                match.append(dict(model=r["model"], seed=r["seed"], cond=c, **v))
        else:
            rows.append(r)
df = pd.DataFrame(rows)
df["rec_val"] = df[["rec_val_role1", "rec_val_role2"]].mean(1)
out = []
for (m, s, c, rd), g in df.groupby(["model", "seed", "cond", "readout"]):
    b = g.loc[g.rec_val.idxmax()]
    out.append(dict(model=m, seed=s, cond=c, readout=rd, code=b.code, kappa=b.kappa_mean, kappa_fig=b.kappa_role1, kappa_gnd=b.kappa_role2))
S = pd.DataFrame(out)
sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
T = S.groupby(["model", "cond", "readout"]).agg(kappa=("kappa", "mean"), sem=("kappa", sem), kappa_fig=("kappa_fig", "mean"),
                                                  kappa_gnd=("kappa_gnd", "mean"), n=("seed", "nunique")).reset_index()
# E34 reference values (same seeds, same selection rule), recomputed with SEM from its per-seed table
ref = pd.read_csv(f"{d34}/agg_per_seed.csv")
ref = ref[ref.readout.isin(["G2", "Q4", "M"]) & ref.model.isin(T.model.unique())]
R = ref.groupby(["model", "cond", "readout"]).agg(kappa=("kappa", "mean"), sem=("kappa", sem), kappa_fig=("kappa_fig", "mean"),
                                                   kappa_gnd=("kappa_gnd", "mean"), n=("seed", "nunique")).reset_index()
A = pd.concat([T, R]).reset_index(drop=True)
A["cond"] = A.cond.replace({"rand": "moved"})
A["v"] = A.kappa.round(3).astype(str) + " +-" + A["sem"].round(3).astype(str)
print("=== kappa (mean +- SEM over seeds; n seeds =", sorted(A.n.unique()), ")")
for cond in ("fixed", "moved"):
    print(cond); print(A[A.cond == cond].pivot(index="model", columns="readout", values="v").to_string())
print("=== POOL figure / ground kappa")
print(A[A.readout == "POOL"].round(3)[["model", "cond", "kappa_fig", "kappa_gnd"]].to_string(index=False))
M = pd.DataFrame(match)
MT = M.groupby(["model", "cond"]).agg(role=("role_acc", "mean"), role_sem=("role_acc", sem), id=("id_acc", "mean"), id_sem=("id_acc", sem),
                                      n_seeds=("seed", "nunique"), n_canvases=("n", "first")).reset_index()
print("=== image-only matching (chance .5 for roles)")
print(MT.round(3).to_string(index=False))
json.dump(dict(kappa=A.drop(columns="v").to_dict("records"), per_seed=S.to_dict("records"), match=MT.to_dict("records"),
               match_per_seed=M.to_dict("records")), open(f"{d}/summary.json", "w", encoding="utf-8"), indent=1)
