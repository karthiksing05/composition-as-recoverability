"""Aggregate the round-3 review experiments: E2 penalty selection (e2h), E5 controls (e5d),
and E4 with a wider penalty grid, delta, and shared part codes for random weights (e4d)."""
import json, glob, sys
import numpy as np, pandas as pd
from agg_r2 import load as load_dir, e2cols, within_corr

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
which = sys.argv[1:] or ["e2h", "e5d", "e4d"]


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x))


if "e2h" in which:
    h = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2h/*.json")])
    old = e2cols(load_dir("e2v2"))
    m = h.merge(old[["arch", "m", "V", "seed", "beh_fresh_1", "beh_fresh_r11", "vocab_fresh_1", "pool_fresh_1"]],
                on=["arch", "m", "V", "seed"], suffixes=("", "_old"))
    print("=== E2h: runs", len(h), "matched", len(m), "max |beh_fresh_r11 - old|", float((m.beh_fresh_r11 - m.beh_fresh_r11_old).abs().max()))
    tags = []
    for sys_ in ("vocab", "pool", "init", "inputvocab"):
        for rule in ("cv", "glyph", "fresh"):
            m[f"{sys_}_{rule}"] = m[[f"{sys_}_{rule}_role1", f"{sys_}_{rule}_role2"]].mean(1)
            tags.append(f"{sys_}_{rule}")
    print("max |vocab_cv - old vocab readout|", float((m.vocab_cv - m.vocab_fresh_1).abs().max()))
    print(m.groupby("V")[tags].mean().round(3).T.to_string())
    print(m[m.m == 64].groupby(["arch", "V"])[["vocab_cv", "vocab_glyph", "vocab_fresh", "inputvocab_cv", "inputvocab_glyph", "inputvocab_fresh"]].mean().round(3).to_string())
    print("per-seed spread of inputvocab at V=32:", m[m.V == 32].groupby("seed").inputvocab_cv.mean().round(2).tolist(),
          m[m.V == 32].groupby("seed").inputvocab_glyph.mean().round(2).tolist())
    keys = ["arch", "V", "m"]
    for x in tags:
        w, p, n = within_corr(m, x, "beh_fresh_1", keys)
        print(f"corr({x}, beh): within={w:.3f} pooled={p:.3f} n={n}")
    for x in ("vocab_glyph_alpha_role1", "vocab_fresh_alpha_role1", "inputvocab_glyph_alpha_role1", "pool_fresh_alpha_role1"):
        print(x, m[x].describe()[["min", "50%", "max"]].to_dict())
    # kappa per V, from the population readout and the e2refs references at m = 64
    rf = pd.DataFrame(json.load(open("results/e2refs/refs.json", encoding="utf-8")))
    cc, ss = rf.concat_m64.mean(), rf.sum_m64.mean()
    k = m[(m.m == 64) & (m.arch != "tf")].groupby("V")[["pool_cv", "pool_fresh"]].mean()
    print("kappa (m=64, non-tf) from pool readout:\n", ((k - ss) / (cc - ss)).round(3).to_string(), "\nrefs", round(cc, 3), round(ss, 3))

if "e5d" in which:
    e5 = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5d/*.json")])
    print("=== E5d runs", len(e5), e5.groupby(["second", "V"]).size().to_dict())
    ctl = ("swap", "swapB2", "rand", "randB1", "randB1cov", "natural", "natural1")
    for t in ctl:
        num, den = (f"ctl_{t}_dleft_r1", f"ctl_{t}_dright_r1") if t == "swapB2" else (f"ctl_{t}_dright_r1", f"ctl_{t}_dleft_r1")
        e5[f"spill_{t}"] = e5[num] / e5[den]
    for sec in ("over", "occ"):
        d = e5[e5.second == sec]
        g = d.groupby("V")
        print(f"--- {sec}: spill, mean of per-run ratios (sem)")
        print(pd.concat({t: g[f"spill_{t}"].mean().round(3).astype(str) + " (" + g[f"spill_{t}"].agg(sem).round(3).astype(str) + ")" for t in ctl}, axis=1).to_string())
        print("    ratio of means:", {t: g.apply(lambda x: x[f"ctl_{t}_dright_r1"].mean() / x[f"ctl_{t}_dleft_r1"].mean()).round(3).to_dict() for t in ctl if t != "swapB2"})
        np_ = ["newpart_role1_before", "newpart_role2_before", "newpart_heldout_before", "newpart_role1_replay0", "newpart_role2_replay0",
               "newpart_heldout_replay0", "newpart_role1_before_old", "newpart_role2_before_old", "newpart_role1_oldctl", "newpart_role2_oldctl",
               "newpart_heldout_oldctl", "newpart_role1_bothroles", "newpart_role2_bothroles", "newpart_heldout_bothroles",
               "newpart_role1_replay1", "newpart_role2_replay1", "newpart_heldout_replay1", "newpart_role2_addcode", "newpart_role2_addcode_avg"]
        print(g[np_].mean().round(3).T.to_string())
        # paired: drop in role 2 vs drop on held-out pairs after one-role fine-tuning, per run
        d = d.assign(dr2=d.newpart_role2_before - d.newpart_role2_replay0, dho=d.newpart_heldout_before - d.newpart_heldout_replay0,
                     dr2o=d.newpart_role2_before_old - d.newpart_role2_oldctl, dho_o=d.newpart_heldout_before - d.newpart_heldout_oldctl,
                     add_vs_ft=d.newpart_role2_addcode - d.newpart_role2_replay0, add_vs_before=d.newpart_role2_addcode - d.newpart_role2_before)
        d["diff"] = d.dr2 - d.dho
        d["diff_o"] = d.dr2o - d.dho_o
        print(d.groupby("V")[["dr2", "dho", "diff", "dr2o", "dho_o", "diff_o", "add_vs_ft", "add_vs_before"]].agg(["mean", sem]).round(3).T.to_string())

for E4DIR in [w for w in which if w.startswith("e4") and w not in ("e4mix", "e4wn")]:
    print(f"########## {E4DIR}")
    rows = []
    for f in glob.glob(f"results/{E4DIR}/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
    df = pd.DataFrame(rows)
    df["key"] = np.where(df.codes_from != df.model, df.model + "@" + df.codes_from, df.model)
    for s in ("pairs", "val", "fresh"):
        df[f"rec_{s}"] = df[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    for s in ("val", "fresh"):
        df[f"id_{s}"] = df[[f"id_{s}_role1", f"id_{s}_role2"]].mean(1)
        df[f"role_{s}"] = df[[f"role_{s}_role1", f"role_{s}_role2"]].mean(1)
    asym = ["on", "left", "under", "inside", "behind"]
    A = df[df.rel.isin(asym)]
    sel = A[A.system == "model"].groupby(["key", "layer", "code"]).rec_val.mean().reset_index()
    best = sel.loc[sel.groupby("key").rec_val.idxmax()].set_index("key")
    print("=== E4d selected on validation nouns:\n", best.to_string())
    B = A.merge(best[["layer", "code"]].reset_index(), on=["key", "layer", "code"])
    K = ["rec_pairs", "rec_fresh", "rec_fresh_role1", "rec_fresh_role2", "id_fresh", "role_fresh", "rev_sim", "inter", "delta_pairs", "delta_fresh"]
    t = B.groupby(["key", "system"])[K].mean().round(3)
    print(t.to_string())
    B.groupby(["key", "system"])[K].agg(["mean", "sem"]).to_csv(f"results/{E4DIR}_summary.csv")
    p = t["rec_fresh"].unstack("system")
    p["kappa"] = (p["model"] - p["ref_sum"]) / (p["ref_concat"] - p["ref_sum"])
    print("\nkappa:\n", p.round(3).to_string())
    mm = B[B.system == "model"]
    print("\nfraction of selected readouts at the top of the grid:", mm.groupby("key").apply(
        lambda x: float(((x.alpha_role1 >= x.alpha_max) | (x.alpha_role2 >= x.alpha_max)).mean())).round(2).to_dict())
    print("median alpha role1:", mm.groupby("key").alpha_role1.median().to_dict())
    # compare to e4c (narrow grid) for trained models
    old = []
    for f in glob.glob("results/e4c/*_n165_s*.json"):
        old += json.load(open(f, encoding="utf-8"))
    o = pd.DataFrame(old)
    o["rec_fresh"] = o[["rec_fresh_role1", "rec_fresh_role2"]].mean(1)
    o = o[o.rel.isin(asym) & (o.system == "model")].merge(best[["layer", "code"]].reset_index().rename(columns={"key": "model"}), on=["model", "layer", "code"])
    print("\ne4c (narrow grid) rec_fresh at the e4d-selected layer/code:", o.groupby("model").rec_fresh.mean().round(3).to_dict())
    C = df[df.rel == "and"].merge(best[["layer", "code"]].reset_index(), on=["key", "layer", "code"])
    print("\n'and' control:\n", C[C.system == "model"].groupby("key")[["rec_fresh", "role_fresh", "rev_sim", "delta_fresh"]].mean().round(3).to_string())

if "e4wn" in which:  # pool-size curve with whitened codes, at the e4w-selected layer and code
    rows = []
    for f in glob.glob("results/e4wn/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
    n = pd.DataFrame(rows)
    for s in ("val", "fresh"):
        n[f"rec_{s}"] = n[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    n = n[n.rel.isin(["on", "left", "under", "inside", "behind"])]
    sel = n[(n.system == "model") & (n.n_fit == 165)].groupby(["model", "layer", "code"]).rec_val.mean().reset_index()
    best = sel.loc[sel.groupby("model").rec_val.idxmax()][["model", "layer", "code"]]
    B = n.merge(best, on=["model", "layer", "code"])
    print("=== e4wn: rec_fresh by n_fit (whitened)\n", B.groupby(["model", "system", "n_fit"]).rec_fresh.mean().unstack("n_fit").round(3).to_string())

if "e4mix" in which:  # kappa against a role-filler reference with non-orthogonal role maps (review A, M5)
    mx = pd.DataFrame([r for f in glob.glob("results/e4mix/*.json") for r in json.load(open(f, encoding="utf-8"))])
    mx["rec_fresh"] = mx[["rec_fresh_role1", "rec_fresh_role2"]].mean(1)
    rows = []
    for f in glob.glob("results/e4w/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
    w = pd.DataFrame(rows)
    w = w[w.codes_from == w.model]
    w["rec_fresh"] = w[["rec_fresh_role1", "rec_fresh_role2"]].mean(1)
    w = w.merge(mx[["model", "layer", "code"]].drop_duplicates(), on=["model", "layer", "code"])
    w = w[w.rel.isin(["on", "left", "under", "inside", "behind"])]
    t = w.groupby(["model", "system"]).rec_fresh.mean().unstack()
    t["ref_mix"] = mx[mx.system == "ref_mix"].groupby("model").rec_fresh.mean()
    t["ref_concat_check"] = mx[mx.system == "ref_concat"].groupby("model").rec_fresh.mean()
    t["kappa_concat"] = (t.model - t.ref_sum) / (t.ref_concat - t.ref_sum)
    t["kappa_mix"] = (t.model - t.ref_sum) / (t.ref_mix - t.ref_sum)
    print("=== e4mix (n runs", len(mx), ")\n", t.round(3).to_string())
