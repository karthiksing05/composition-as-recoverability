"""Aggregate E30: kappa on fresh photographs (layer, timestep and code chosen on validation parts, as agg_r7.e4style)
and the diffusion classifier.  usage: agg_e30.py [results dir]"""
import glob, json, sys
import numpy as np, pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
d = sys.argv[1] if len(sys.argv) > 1 else "results/e30"
rows = []
for f in glob.glob(f"{d}/*_trained_s*_*.json") + glob.glob(f"{d}/*_random_s*_*.json"):
    rows += json.load(open(f, encoding="utf-8"))
df = pd.DataFrame(rows)
if len(df):
    df["name"] = df.model.str.split("|").str[0]
    for s in ("val", "fresh"):
        df[f"rec_{s}"] = df[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    df["pool"] = df.layer.str[0]
    df["lay"] = df.layer.str[1:]
    refs = df[df.system != "model"].copy()
    refs["pool"] = "ref"
    R = refs.pivot_table(index=["name", "source", "seed", "rel", "lay", "code"], columns="system", values="rec_fresh").reset_index()
    M = df[df.system == "model"].copy()
    # the code layer a model row was read against: its own layer, or the trained U-Net's layer named in code_layer
    M["clay"] = M.code_layer.str[1:]
    M["rsrc"] = np.where(M.codesrc == "own", M.source, "trained")
    M = M.merge(R.rename(columns={"source": "rsrc", "lay": "clay"}), on=["name", "rsrc", "seed", "rel", "clay", "code"], how="left")
    M["kappa"] = (M.rec_fresh - M.ref_sum) / (M.ref_concat - M.ref_sum)
    M["ckey"] = M.code + "@" + M.clay
    out = []
    for (name, source, codesrc, pool), g in M.groupby(["name", "source", "codesrc", "pool"]):
        if g.ref_concat.isna().all():
            continue
        sel = g.groupby(["lay", "ckey"]).rec_val.mean()
        lay, ck = sel.idxmax()
        b = g[(g.lay == lay) & (g.ckey == ck)]
        ks = b.groupby("seed").kappa.mean()
        for rel, r in b.groupby("rel"):
            out.append(dict(name=name, source=source, codes=codesrc, pool=pool, layer=lay, code=ck, rel=rel,
                            kappa=r.kappa.mean(), rec_fresh=r.rec_fresh.mean(), ref_concat=r.ref_concat.mean(), ref_sum=r.ref_sum.mean(),
                            rec_val=r.rec_val.mean(), n_seeds=r.seed.nunique(), alpha_max=r.alpha_role1.max()))
        out.append(dict(name=name, source=source, codes=codesrc, pool=pool, layer=lay, code=ck, rel="ALL", kappa=ks.mean(),
                        kappa_sem=ks.sem() if len(ks) > 1 else np.nan, rec_fresh=b.rec_fresh.mean(), ref_concat=b.ref_concat.mean(),
                        ref_sum=b.ref_sum.mean(), rec_val=b.rec_val.mean(), n_seeds=b.seed.nunique(), alpha_max=b.alpha_role1.max()))
    T = pd.DataFrame(out)
    print("=== kappa on fresh photographs (selection on validation parts)\n", T.round(3).to_string(index=False))
    # sanity: references against themselves
    rr = refs.groupby(["name", "source", "system"]).rec_fresh.mean()
    print("=== reference R2 (fresh)\n", rr.round(3).to_string())
    T.to_csv(f"{d}/agg_kappa.csv", index=False)
cl = []
for f in glob.glob(f"{d}/*_classifier_s*_*.json"):
    cl.append(json.load(open(f, encoding="utf-8")))
if cl:
    c = pd.DataFrame(cl)
    g = c.groupby(["model", "rel"])[["roles", "roles_first", "roles_second", "identity"]].agg(["mean", "sem"])
    print("=== diffusion classifier (n seeds:", c.groupby(["model", "rel"]).seed.nunique().to_dict(), ")\n", g.round(3).to_string())
    for m, r in c.groupby("model"):
        print(m, "roles by timestep (mean over rel, seeds):", np.round(np.mean(np.stack(r.roles_by_t), 0), 2).tolist(),
              "\n   t grid", r.t_grid.iloc[0], "\n   identity by t", np.round(np.mean(np.stack(r.identity_by_t), 0), 2).tolist())
