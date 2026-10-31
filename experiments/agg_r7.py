"""Aggregate round 7: large LMs and the Pythia suite (e6, E4 pipeline), semantic vs positional roles (e7),
and image encoders (e8).  Layer and part code are chosen per model on validation parts."""
import json, glob, sys
import numpy as np, pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
which = (sys.argv[1:] or ["e6", "e7", "e8"]) if __name__ == "__main__" else []
ASYM = ["on", "left", "under", "inside", "behind"]


def load(d):
    rows = []
    for f in glob.glob(f"results/{d}/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
    return pd.DataFrame(rows)


def e4style(df, rels, tag):
    for s in ("pairs", "val", "fresh"):
        df[f"rec_{s}"] = df[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    for s in ("val", "fresh"):
        df[f"id_{s}"] = df[[f"id_{s}_role1", f"id_{s}_role2"]].mean(1)
        df[f"role_{s}"] = df[[f"role_{s}_role1", f"role_{s}_role2"]].mean(1)
    A = df[df.rel.isin(rels)]
    sel = A[A.system == "model"].groupby(["model", "layer", "code"]).rec_val.mean().reset_index()
    best = sel.loc[sel.groupby("model").rec_val.idxmax()][["model", "layer", "code"]]
    B = A.merge(best, on=["model", "layer", "code"])
    K = ["rec_pairs", "rec_fresh", "rec_fresh_role1", "rec_fresh_role2", "id_fresh", "role_fresh", "rev_sim", "inter"]
    t = B.groupby(["model", "system"])[K].mean().unstack("system")
    out = pd.DataFrame({"layer": best.set_index("model").layer, "code": best.set_index("model").code,
                        "n_seeds": B[B.system == "model"].groupby("model").seed.nunique()})
    for k in K:
        out[k] = t[(k, "model")]
    out["ref_concat"], out["ref_sum"] = t[("rec_fresh", "ref_concat")], t[("rec_fresh", "ref_sum")]
    out["kappa"] = (out.rec_fresh - out.ref_sum) / (out.ref_concat - out.ref_sum)
    out["id_sum"] = t[("id_fresh", "ref_sum")]
    per = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values="rec_fresh")
    ms = B[B.system == "model"].groupby(["model", "seed"]).rec_fresh.mean()
    k_seed = (ms - per.ref_sum) / (per.ref_concat - per.ref_sum)
    out["kappa_sem"] = k_seed.groupby("model").sem()
    print(f"=== {tag}: selected layer/code, means over seeds and relations {rels}\n", out.round(3).to_string())
    return out, B


if "e6" in which:
    df = load("e6")
    df["model"] = df.model.str.replace("EleutherAI/", "").str.replace("allenai/", "").str.replace("google/", "")
    out, B = e4style(df, ASYM, "e6")
    C = df[df.rel == "and"].merge(out[["layer", "code"]].reset_index(), on=["model", "layer", "code"])
    for s in ("fresh",):
        C[f"rec_{s}"] = C[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    print("'and' control (model):\n", C[C.system == "model"].groupby("model").rec_fresh.mean().round(3).to_string())
    print("role1 vs role2 (causal asymmetry):\n", out[["rec_fresh_role1", "rec_fresh_role2"]].round(3).to_string())
    out.to_csv("results/agg_r7_e6.csv")

if "e7" in which:
    df = load("e7")
    df["model"] = df.model.str.replace("EleutherAI/", "").str.replace("allenai/", "").str.replace("google/", "")
    rows = []
    for m, g in df.groupby("model"):
        mm = g[g.system == "model"]  # layer and code chosen on validation nouns
        sc = mm.groupby(["layer", "code"])[["pos_rec_val", "sem_rec_val"]].apply(lambda x: (x.pos_rec_val + x.sem_rec_val).mean())
        lay, code = sc.idxmax()
        s = g[(g.layer == lay) & (g.code == code)]
        for sysn, h in s.groupby("system"):
            r = h[[c for c in h.columns if c.startswith(("pos_", "sem_", "xfer_"))]].mean()
            r["model"], r["system"], r["layer"], r["code"], r["n_seeds"] = m, sysn, lay, code, h.seed.nunique()
            rows.append(r)
    t = pd.DataFrame(rows).set_index(["model", "system"])
    cols = ["layer", "code", "n_seeds", "pos_rec_fresh", "sem_rec_fresh", "pos_role_fresh", "sem_role_fresh", "sem_role_conv", "xfer_role_can", "xfer_role_conv", "xfer_rec_conv"]
    print("=== e7 (layer/code chosen on validation nouns)\n", t[cols].round(3).to_string())
    print(df[df.system == "model"].groupby(["model", "rel"]).xfer_role_conv.mean().unstack().round(2).to_string())

if "e8" in which:
    df = load("e8")
    out, B = e4style(df, ["beside", "above", "overlap"], "e8")
    for rel in ("beside", "above", "overlap"):
        _, _ = e4style(df.copy(), [rel], f"e8 {rel}")
    out.to_csv("results/agg_r7_e8.csv")
