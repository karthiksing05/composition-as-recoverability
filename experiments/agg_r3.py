"""Aggregate round-3 results: E1 with a fixed fresh pool, E3 population readouts, E5c
interchange and forgetting controls, E2 references and the raw-input vocabulary readout."""
import json, glob
import numpy as np, pandas as pd
from agg_r2 import load as load_dir, e2cols, within_corr

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 500)


def rows(pattern):
    out = []
    for f in glob.glob(pattern):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return pd.DataFrame(out)


# ---- E1
e1 = rows("results/e1v2/*.json")
e1["dist"] = e1["dist"].fillna("gauss")
g = e1[(e1.dist == "gauss") & (e1.readout != "nn_table")]
print("=== E1 gauss fresh_r2 (population denominator, 2000-part pool)")
print(g.pivot_table(index="op", columns=["readout", "V"], values="fresh_r2").round(2).to_string())
d, Vs = 16, sorted(g.V.unique())
cf = {V: (min(V - 1, d) / d - max(d - V + 1, 0) / (V * d)) for V in Vs}
print("closed form concat (min-norm, unstandardized):", {V: round(c, 3) for V, c in cf.items()})
print(g[g.op == "concat"].groupby(["readout", "V"]).fresh_r2.agg(["mean", "sem"]).round(3).unstack(0).to_string())
print("=== E1 robustness (linear, V=128)")
r = e1[(e1.readout.isin(["linear", "ols_raw"])) & (e1.V.isin([32, 128]))]
print(r.pivot_table(index="op", columns=["dist", "readout", "V"], values="fresh_r2").round(2).to_string())
k = rows("results/e1v2k/*.json")
if len(k):
    print("=== E1 kernel vs linear (gauss)")
    print(k[k.readout.isin(["linear", "kernel", "mlp"])].pivot_table(index="op", columns=["readout", "V"], values="fresh_r2").round(2).to_string())
print("=== E1 lookup fam_id (V=32)")
print(e1[(e1.readout == "nn_table") & (e1.V == 32) & (e1.dist == "gauss")].groupby("op").fam_id.mean().round(2).to_string())

# ---- E3
e3 = rows("results/e3v2/*.json")
ra = e3[e3.model == "raam"]
print("=== E3 RAAM: own decomposer fresh_r2 vs population readout pop_r2 (and bound)")
print(ra.groupby(["V", "w", "depth"])[["fresh_r2", "pop_r2", "bound", "fam_id"]].mean().round(3).unstack("depth").to_string())
h = pd.DataFrame(json.load(open("results/e3hrr3/hrr_unitary.json", encoding="utf-8")))
hp = h[h.split == "pool"]
print("=== HRR: population readout, correlation unbinding, crosstalk prediction")
print(hp.groupby(["w", "depth"])[["pop_r2", "r2", "crosstalk_pred"]].mean().round(3).unstack("depth").to_string())
hf = h[h.split == "fam"]
print(hf.groupby(["w", "depth"])["id"].mean().round(3).unstack("depth").to_string())

# ---- E5c
e5 = rows("results/e5c/*.json")
e5 = e5[e5.columns[e5.notna().any()]]
if "newpart_role1_bothroles" in e5:
    print("rows with bothroles:", e5.newpart_role1_bothroles.notna().sum(), "of", len(e5))
    e5 = e5[e5.newpart_role1_bothroles.notna()]
for k_ in ("pred", "true", "noswap"):
    e5[f"rev_out_{k_}"] = e5[[f"rev_out_{k_}_r1{o}" for o in "01"]].mean(1)
for t in ("swap", "rand", "randB1", "swapB2", "natural"):
    e5[f"ctl_{t}_ratio_r1"] = e5[f"ctl_{t}_dright_r1"] / e5[f"ctl_{t}_dleft_r1"]
c5 = ["delta_fresh_r1", "rev_state_r1", "rev_out_pred", "rev_out_true", "rev_out_noswap", "newrole_out_pred_r1", "newrole_out_net_r1",
      "swap_left_base_r1", "swap_right_base_r1"]
for t in ("swap", "rand", "randB1", "swapB2", "natural"):
    c5 += [f"ctl_{t}_left_orig_r1", f"ctl_{t}_right_orig_r1", f"ctl_{t}_left_donor_r1", f"ctl_{t}_right_donor_r1",
           f"ctl_{t}_dleft_r1", f"ctl_{t}_dright_r1", f"ctl_{t}_ratio_r1"]
c5 += [f"newpart_{a}_{t}" for t in ("before", "replay0", "replay1", "oldctl", "bothroles") for a in ("role1", "role2", "heldout")]
for sec in ("over", "occ"):
    d5 = e5[e5.second == sec]
    print(f"=== E5c {sec}: n per V", d5.groupby("V").size().to_dict())
    print(d5.groupby("V")[c5].mean().round(3).T.to_string())
    print(d5[d5.arch != "tf"].groupby("V")[["ctl_swap_ratio_r1", "delta_fresh_r1"]].mean().round(3).to_string())

# ---- E2 references
rf = pd.DataFrame(json.load(open("results/e2refs/refs.json", encoding="utf-8")))
print("=== E2 refs (mean over seeds)")
print(rf.groupby("V").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
e2 = e2cols(load_dir("e2v2"))
m = e2.merge(rf[["V", "seed", "input_vocab_over_r1", "input_pool_over_r1"]], on=["V", "seed"])
for x in ("vocab_fresh_1", "input_vocab_over_r1", "pool_fresh_1"):
    w, p, n = within_corr(m, x, "beh_fresh_1", ["arch", "V", "m"])
    print(f"corr({x}, beh_fresh_1): within={w:.3f} pooled={p:.3f} n={n}")
# partial: does the trained vocab readout predict behavior beyond the raw-input vocab readout?
d_ = m[["vocab_fresh_1", "input_vocab_over_r1", "beh_fresh_1", "arch", "V", "m"]].dropna()
for c in ("vocab_fresh_1", "input_vocab_over_r1", "beh_fresh_1"):
    d_[c + "_w"] = d_[c] - d_.groupby(["arch", "V", "m"])[c].transform("mean")
X = np.c_[d_["vocab_fresh_1_w"], d_["input_vocab_over_r1_w"]]
beta = np.linalg.lstsq(X, d_["beh_fresh_1_w"], rcond=None)[0]
print("within-condition regression beh ~ vocab_trained + vocab_input:", beta.round(3))
# bootstrap CI of difference of within-condition correlations (vocab - pool)
rng = np.random.default_rng(0); diffs = []
for _ in range(1000):
    s = e2.iloc[rng.integers(0, len(e2), len(e2))]
    diffs.append(within_corr(s, "vocab_fresh_1", "beh_fresh_1", ["arch", "V", "m"])[0] - within_corr(s, "pool_fresh_1", "beh_fresh_1", ["arch", "V", "m"])[0])
print("within-corr difference vocab - pool: CI", np.percentile(diffs, [2.5, 97.5]).round(3))
