"""Audit: are the language-model state numbers in main.tex sec:behavior selected on validation or on test parts?
Reads results only (e11, e16, e9 csv, e19/e26, e21/e21b, e23, e33v2); writes nothing but stdout (redirect to e35/audit_lm_selection.txt).
Run from experiments/:  python audit_lm_selection.py
Rules compared per (model, state):
  POOLED  = agg_e11.py / e33_agg.py rule: argmax over (layer, code) of validation score averaged over seeds and relations
  PERSEED = argmax per seed of validation score (mean over relations), then that seed's test value
  TESTBEST= argmax over (layer, code) of mean test transfer (oracle; optimistic)
  FIXED   = layer/code hard-coded in fig_story.py E11 dict / e21_intervene PICK
Validation score: e11 = sem_rec_val + xfer_rec_val (agg_e11.py); e33v2 = val_rec + sem_val_rec (e33_agg.py); e16 = xfer_rec_val only.
+- is SEM over seeds (std ddof=1 / sqrt(n)); e33_agg.py's summary.txt prints the SD, not the SEM."""
import json, glob
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
short = lambda s: s.split("/")[-1]


def load(d, pat="*.json"):
    rows = []
    for f in sorted(glob.glob(f"results/{d}/{pat}")):
        r = json.load(open(f, encoding="utf-8"))
        rows += r if isinstance(r, list) else [r]
    D = pd.DataFrame(rows)
    D["model"] = D.model.astype(str).map(short)
    return D


def ms(per_seed):
    a = np.asarray(per_seed, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan)


def fmt(t):
    return f"{t[0]:.3f}+-{t[1]:.3f}"


def three_rules(g, valcol, testcol, fixed=None):
    """g: rows of one (model,state,target) with columns seed, rel, layer, code. Returns dict of (mean, sem, layer/code)."""
    key = ["layer", "code"]
    pooled = g.groupby(key)[valcol].mean().idxmax()
    gp = g[(g.layer == pooled[0]) & (g.code == pooled[1])]
    r = {"POOLED": (ms(gp.groupby("seed")[testcol].mean()), pooled)}
    ps = []
    for s, gs in g.groupby("seed"):
        lc = gs.groupby(key)[valcol].mean().idxmax()
        ps.append(gs[(gs.layer == lc[0]) & (gs.code == lc[1])][testcol].mean())
    r["PERSEED"] = (ms(ps), None)
    tb = g.groupby(key)[testcol].mean().idxmax()
    gt = g[(g.layer == tb[0]) & (g.code == tb[1])]
    r["TESTBEST"] = (ms(gt.groupby("seed")[testcol].mean()), tb)
    if fixed is not None:
        gf = g[(g.layer == fixed[0]) & (g.code == fixed[1])]
        r["FIXED"] = (ms(gf.groupby("seed")[testcol].mean()), fixed)
    return r


def line(name, r):
    s = f"{name:34s}"
    for k in ("FIXED", "POOLED", "PERSEED", "TESTBEST"):
        if k in r:
            (m, e), lc = r[k][0], r[k][1]
            s += f" | {k} {m:.3f}+-{e:.3f}" + (f" [{lc[0]}/{lc[1]}]" if lc is not None else "")
    print(s)


E11 = {"phi-4": ("L30", "phrase"), "Qwen3-8B": ("L27", "carrier"), "OLMo-2-1124-13B": ("L30", "phrase"),
       "OLMo-2-1124-7B": ("L16", "carrier"), "gpt2-large": ("L18", "carrier"), "pythia-1.4b": ("L12", "carrier"),
       "pythia-12b": ("L18", "carrier")}
ORDER = ["phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B", "gpt2-large", "pythia-1.4b", "pythia-12b"]
MEAN = ORDER[:4]

# ---------------------------------------------------------------- 1. e11 main-prompt question state (source of fig_both b main dots, body 0.98/0.87/0.73/0.20)
print("=" * 30, "E11 query state, xfer_role_conv (figure probe fitted on canonical, scored on converse); val = sem_rec_val+xfer_rec_val")
d11 = load("e11"); d11 = d11[d11.system == "model"].copy()
d11["val"] = d11.sem_rec_val + d11.xfer_rec_val
e11_main = {}
for m in ORDER:
    g = d11[(d11.model == m) & (d11.state == "query")]
    if g.empty:
        print(m, "no e11 rows"); continue
    r = three_rules(g, "val", "xfer_role_conv", E11[m])
    line(m, r); e11_main[m] = r
    # also val-only on the xfer criterion, and the canonical-only val criterion
    r2 = three_rules(g.assign(val=g.xfer_rec_val), "val", "xfer_role_conv")
    line("  (val = xfer_rec_val only)", r2)

# ---------------------------------------------------------------- 2. e16 prompts + correlation
print("\n" + "=" * 30, "E16 prompt variants (answers = choice; transfer = xfer_role_conv)")
d16 = load("e16")
e9 = pd.read_csv("results/agg_r8_e9.csv", index_col=0)
pts = {}  # (model, variant) -> dict(ans, fixed, pooledval, testbest)
for (m, v), g in d16.groupby(["model", "variant"]):
    gg = g[(g.layer == E11[m][0]) & (g.code == E11[m][1])]
    ans = gg.choice.mean()
    gv = g.assign(val=g.xfer_rec_val)
    r = three_rules(gv, "val", "xfer_role_conv", E11[m])
    pts[m, v] = dict(ans=ans, FIXED=r["FIXED"][0][0], POOLED=r["POOLED"][0][0], PERSEED=r["PERSEED"][0][0], TESTBEST=r["TESTBEST"][0][0])
    line(f"{m}/{v}  ans {ans:.3f}", r)
for m in ORDER:
    if m in e11_main:
        r = e11_main[m]
        pts[m, "main"] = dict(ans=e9.loc[m, "choice"], FIXED=r["FIXED"][0][0], POOLED=r["POOLED"][0][0], PERSEED=r["PERSEED"][0][0], TESTBEST=r["TESTBEST"][0][0])

# e33v2 (newer, 12 layers x 2 codes, 5 relations) val-selected main-prompt transfer, for the four meaning models
d33 = load("e33v2")
d33["val"] = d33.val_rec + d33.sem_val_rec
e33 = {}
for m in MEAN:
    g = d33[(d33.model == m) & (d33.state == "qf") & (d33.target == "fig")]
    e33[m] = three_rules(g, "val", "role_conv")
print("\ne33v2 qf->fig:")
for m in MEAN:
    line(m, e33[m])

df = pd.DataFrame(pts).T
df.index.names = ["model", "variant"]


def corr(sub, col, label):
    a, t = sub.ans.values.astype(float), sub[col].values.astype(float)
    if len(a) < 3:
        print(f"  {label:28s} n={len(a)} too small"); return
    pr, pp = pearsonr(a, t); sr, sp = spearmanr(a, t)
    print(f"  {label:28s} n={len(a):2d} pearson {pr:.3f} (p={pp:.3g})  spearman {sr:.3f} (p={sp:.3g})")


idx = df.index.get_level_values(0)
hi, lo = df[idx.isin(MEAN)], df[~idx.isin(MEAN)]
print("\nCorrelation answer accuracy vs transfer, 7 models x 4 prompts (fig_both b). Transfer column per rule:")
for col in ("FIXED", "POOLED", "PERSEED", "TESTBEST"):
    print(f" transfer = {col}")
    corr(df, col, "all points"); corr(hi, col, "within meaning cluster"); corr(lo, col, "within word-order cluster")
    mn = df.xs("main", level=1); corr(mn, col, "main prompt only (7 models)")
# main prompt replaced by e33v2 pooled val-selected for the four meaning models
df2 = df.copy()
for m in MEAN:
    df2.loc[(m, "main"), "POOLED"] = e33[m]["POOLED"][0][0]
    df2.loc[(m, "main"), "FIXED"] = e33[m]["POOLED"][0][0]
print(" transfer = e33v2 pooled val-selected on main prompt (others as E16 FIXED)")
idx2 = df2.index.get_level_values(0)
corr(df2, "FIXED", "all points"); corr(df2[idx2.isin(MEAN)], "FIXED", "within meaning cluster"); corr(df2[~idx2.isin(MEAN)], "FIXED", "within word-order cluster")
corr(df2.xs("main", level=1), "FIXED", "main prompt only")
print("\nPoints table:\n", df.round(3).to_string())

# ---------------------------------------------------------------- 3. e33v2 all states
print("\n" + "=" * 30, "E33v2 (val = val_rec+sem_val_rec; role_conv)")
for m in MEAN:
    print("---", m)
    for st, tg in (("qf", "fig"), ("qf", "gnd"), ("qg", "gnd"), ("qg", "fig"), ("period", "fig"), ("period", "gnd"), ("last", "fig"), ("mean", "fig")):
        g = d33[(d33.model == m) & (d33.state == st) & (d33.target == tg)]
        r = three_rules(g, "val", "role_conv")
        line(f"{st}->{tg}", r)
        if (st, tg) in (("period", "fig"), ("mean", "fig")):
            lc = r["POOLED"][1]; gp = g[(g.layer == lc[0]) & (g.code == lc[1])]
            print(f"     role_can at pooled-selected layer: {gp.groupby('seed').role_can.mean().mean():.3f}")
    # unasked role at the asked-role layer (summary rule)
    for st, asked, un in (("qf", "fig", "gnd"), ("qg", "gnd", "fig")):
        ga = d33[(d33.model == m) & (d33.state == st) & (d33.target == asked)]
        lc = ga.groupby(["layer", "code"]).val.mean().idxmax()
        gu = d33[(d33.model == m) & (d33.state == st) & (d33.target == un) & (d33.layer == lc[0]) & (d33.code == lc[1])]
        print(f"  {st} unasked {un} at asked layer {lc}: conv {fmt(ms(gu.groupby('seed').role_conv.mean()))} can {gu.role_can.mean():.3f}")
# ranges quoted in text
print("\nSeed-level check that the same layer is chosen across seeds (e33v2 qf->fig):")
for m in MEAN:
    g = d33[(d33.model == m) & (d33.state == "qf") & (d33.target == "fig")]
    print(m, {int(s): tuple(gs.groupby(["layer", "code"]).val.mean().idxmax()) for s, gs in g.groupby("seed")})

# ---------------------------------------------------------------- 4. probes: ridge / kernel / mlp (OLMo-2 7B)
print("\n" + "=" * 30, "OLMo-2 7B query probes (e19 L16, e26L24, e26L32): hyperparameters from validation vs oracle on test")
for d in ("e19", "e26L24", "e26L32"):
    D = load(d); D = D[(D.model == "OLMo-2-1124-7B") & (D.state == "query")]
    if D.empty:
        continue
    cols = [c for c in D.columns if c.endswith("_conv")]
    print(d, {c: round(float(D.groupby("seed")[c].mean().mean()), 3) for c in cols}, "layers", sorted(D.layer.unique()), "codes", sorted(D.code.unique()))
# any layer:
# ---------------------------------------------------------------- 5. patching
print("\n" + "=" * 30, "Patching: e21 to_answer (source=reversed, whole) and e21b flip (whole) by layer; 'picked' = E11/PICK layer")
d21 = load("e21"); d21b = load("e21b")
for m in ("phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B"):
    g = d21[(d21.model == m) & (d21.source == "reversed") & (d21.kind == "whole")]
    L = sorted(g.layer.unique())
    pk = sorted(g[g.picked].layer.unique())
    print(m, "e21 to_answer by layer", {int(l): round(float(g[g.layer == l].groupby("seed").to_answer.mean().mean()), 3) for l in L}, "picked", pk)
    gb = d21b[(d21b.model == m) & (d21b.kind == "whole")]
    if len(gb):
        print("   e21b flip by layer", {int(l): round(float(gb[gb.layer == l].groupby("seed").flip.mean().mean()), 3) for l in sorted(gb.layer.unique())}, "seeds", gb.seed.nunique())
    # subspace kinds at picked layer
    gk = d21b[(d21b.model == m)]
    if len(gk):
        l = gk.layer.max() if False else None
print("e21b subspace flips at each model's layer (mean over models/seeds), kinds:")
for k in sorted(d21b.kind.unique()):
    g = d21b[d21b.kind == k]
    print("  ", k, round(float(g.flip.mean()), 3), end=";")
print()

# ---------------------------------------------------------------- 6. invented words (e23)
print("\n" + "=" * 30, "E23 invented words")
d23 = load("e23")
print("columns:", list(d23.columns))
num = [c for c in d23.columns if d23[c].dtype.kind == "f"]
for m in ("phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B", "gpt2-large", "pythia-1.4b", "pythia-12b"):
    g = d23[(d23.model == m) & (d23.picked)]
    if g.empty:
        continue
    print(m, "picked", sorted({(l, c) for l, c in zip(g.layer, g.code)}), {t: {c: round(float(gt[c].mean()), 3) for c in num if c in ("choice", "xfer_role_conv", "xfer_conv", "answers", "transfer", "role_conv", "kappa")} for t, gt in g.groupby("test")})
g = d23[(d23.model == "phi-4")]
tcols = [c for c in num if "conv" in c or c in ("choice",)]
print("phi-4 all layers/codes by test (cols", tcols, ")")
print(g.groupby(["test", "layer", "code"])[tcols].mean().round(3).to_string())
