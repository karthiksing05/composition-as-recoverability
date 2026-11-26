"""Aggregate E40 (attribute binding).  Mirrors e33_agg.py.  Selection: layer x code with the highest
val_rec + sem_val_rec (E33/E11 rule), mean over seeds and colour pairs.  Prints per model: answer accuracy
(canonical / converse), selected layer, transfer on converse and canonical for qa/qb/period/mean."""
import json, glob, sys, os
import numpy as np, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e40"
rows, beh = [], []
for f in sorted(glob.glob(f"{d}/*_s?.json")):
    j = json.load(open(f, encoding="utf-8"))
    rows += j["rows"]; beh += j["behavior"]
D = pd.DataFrame(rows); B = pd.DataFrame(beh)
D["model"] = D.model.str.split("/").str[-1]; B["model"] = B.model.str.split("/").str[-1]
D["sel"] = D.val_rec + D.sem_val_rec if "sem_val_rec" in D else D.val_rec
out = {}
def stat(g, col):
    per_seed = g.groupby("seed")[col].mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"), len(per_seed)
for m, gm in D.groupby("model"):
    print(f"\n=== {m}  (seeds {sorted(gm.seed.unique())}, pairs {gm.rel.nunique()}, layers {sorted(gm.layer.unique())})")
    gb = B[B.model == m]
    acc = {ph: stat(gb[gb.phrasing == ph], "choice") for ph in ("canonical", "converse")}
    out[f"{m}|answer_acc"] = acc
    print(f"answer acc: canonical {acc['canonical'][0]:.3f}+-{acc['canonical'][1]:.3f}  converse {acc['converse'][0]:.3f}+-{acc['converse'][1]:.3f}")
    sel_layer = {}
    for (st, tg), g in gm.groupby(["state", "target"]):
        sel_layer[(st, tg)] = g.groupby(["layer", "code"]).sel.mean().idxmax()
    for (st, tg), g in gm.groupby(["state", "target"]):
        sel = sel_layer[(st, tg)]
        gs = g[(g.layer == sel[0]) & (g.code == sel[1])]
        bl = g.groupby(["layer", "code"]).role_conv.mean().idxmax()
        gbst = g[(g.layer == bl[0]) & (g.code == bl[1])]
        rec = dict(sel_layer=int(sel[0]), sel_code=sel[1], conv=stat(gs, "role_conv"), can=stat(gs, "role_can"), conv_rec=stat(gs, "rec_conv")[0],
                   val_rec=stat(gs, "val_rec")[0], best_layer=int(bl[0]), best_code=bl[1], best_conv=stat(gbst, "role_conv"))
        out[f"{m}|{st}|{tg}"] = rec
        print(f"{st:7s} -> {tg}: sel L{rec['sel_layer']}/{rec['sel_code']:7s} val R2 {rec['val_rec']:.3f} conv {rec['conv'][0]:.3f}+-{rec['conv'][1]:.3f} can {rec['can'][0]:.3f} | best L{rec['best_layer']}/{rec['best_code']} conv {rec['best_conv'][0]:.3f}")
    for st, asked, un in (("qa", "c1", "c2"), ("qb", "c2", "c1")):
        lc = sel_layer[(st, asked)]
        gs = gm[(gm.state == st) & (gm.target == un) & (gm.layer == lc[0]) & (gm.code == lc[1])]
        s = stat(gs, "role_conv"); c = stat(gs, "role_can")
        out[f"{m}|{st}|{un}|at_asked_layer"] = dict(layer=int(lc[0]), code=lc[1], conv=s, can=c)
        print(f"{st} unasked {un} at asked-role layer L{lc[0]}/{lc[1]}: conv {s[0]:.3f}+-{s[1]:.3f} can {c[0]:.3f}")
    for st in ("period", "mean", "qa"):
        g = gm[(gm.state == st) & (gm.target == "c1")]
        pl = g.groupby(["layer", "code"]).role_conv.mean().groupby("layer").max()
        out[f"{m}|{st}|c1|per_layer_best_code_conv"] = {int(k): float(v) for k, v in pl.items()}
        print(f"per-layer {st}->c1 conv (best code):", " ".join(f"L{int(k)}:{v:.2f}" for k, v in pl.items()))
    out[f"{m}|rel_breakdown"] = {}
    for st, tg in (("qa", "c1"), ("qa", "c2"), ("period", "c1"), ("qb", "c2"), ("qb", "c1")):
        sel = sel_layer[(st, tg)]
        g = gm[(gm.state == st) & (gm.target == tg) & (gm.layer == sel[0]) & (gm.code == sel[1])]
        out[f"{m}|rel_breakdown"][f"{st}->{tg}"] = g.groupby("rel").role_conv.mean().round(3).to_dict()
    out[f"{m}|answer_acc_by_pair"] = gb.groupby(["rel", "phrasing"]).choice.mean().round(3).unstack().to_dict()
os.makedirs(d, exist_ok=True)
json.dump(out, open(f"{d}/summary.json", "w", encoding="utf-8"), indent=1, default=float)
