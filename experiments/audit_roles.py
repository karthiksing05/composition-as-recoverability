"""Audit: kappa for each role separately, at the layer and code the figures choose.

In a causal model the tokens of the first noun never see the second, so role 1 can
be recoverable with no training.  This script reports kappa for role 1 and role 2,
for trained and random weights, and the E7 role tests for each model."""
import numpy as np, pandas as pd
from agg_r7 import load, e4style, ASYM


def short(df):
    for pre in ("EleutherAI/", "allenai/", "microsoft/", "Qwen/"):
        df["model"] = df.model.str.replace(pre, "")
    return df


def per_role(df, rels):
    _, B = e4style(df, rels, "audit")
    t = B.groupby(["model", "seed", "system"])[["rec_fresh_role1", "rec_fresh_role2"]].mean().unstack("system")
    out = {}
    for r in ("1", "2"):
        k = f"rec_fresh_role{r}"
        out[f"kappa{r}"] = (t[(k, "model")] - t[(k, "ref_sum")]) / (t[(k, "ref_concat")] - t[(k, "ref_sum")])
        out[f"rec{r}"] = t[(k, "model")]
    out = pd.DataFrame(out).groupby("model").mean()
    best = B[B.system == "model"].groupby("model")[["layer", "code"]].first()
    return out.join(best)


w = load("e4w")
w = w[(w.n_fit == 165) & (w.codes_from == w.model) & (~w.detok.fillna(False).astype(bool))]
tables = {"text encoders (e4w)": per_role(w, ASYM), "language models (e6)": per_role(short(load("e6")), ASYM),
          "image encoders (e8)": per_role(load("e8"), ["beside", "above", "overlap"])}
pd.set_option("display.width", 200)
for name, t in tables.items():
    print(f"\n== {name}")
    print(t[["layer", "code", "rec1", "rec2", "kappa1", "kappa2"]].round(3).to_string())

e7 = short(load("e7"))
e7 = e7[e7.system == "model"] if "system" in e7 else e7
cols = ["pos_role_fresh", "sem_role_fresh", "xfer_role_conv", "pos_rec_fresh", "sem_rec_fresh"]
print("\n== E7 role tests (first-mentioned noun, figure fit on both, figure fit canonical -> converse)")
print(e7.groupby("model")[cols].mean().round(3).to_string())
