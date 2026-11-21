"""Aggregate E33.  Selection: layer x code with the highest validation R2 (canonical new nouns), mean over seeds and relations.
'best' = layer x code with the highest mean converse transfer (oracle, optimistic)."""
import json, glob, sys
import numpy as np, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "results/e33"
rows = []
for f in sorted(glob.glob(f"{d}/*_s?.json")):
    rows += json.load(open(f, encoding="utf-8"))
D = pd.DataFrame(rows)
D["model"] = D.model.str.split("/").str[-1]
D["sel"] = D.val_rec + D.sem_val_rec if "sem_val_rec" in D else D.val_rec  # paper rule (E11): xfer val R2 + sem val R2
out = {}
lines = []
def stat(g, col):
    per_seed = g.groupby("seed")[col].mean()  # mean over relations, per seed
    return per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else float("nan"), len(per_seed)
for m, gm in D.groupby("model"):
    print(f"\n=== {m}  (seeds {sorted(gm.seed.unique())}, rels {gm.rel.nunique()}, layers {sorted(gm.layer.unique())})")
    sel_layer = {}
    for (st, tg), g in gm.groupby(["state", "target"]):
        lc = g.groupby(["layer", "code"]).sel.mean().idxmax()
        sel_layer[(st, tg)] = lc
    for (st, tg), g in gm.groupby(["state", "target"]):
        sel = sel_layer[(st, tg)]
        gs = g[(g.layer == sel[0]) & (g.code == sel[1])]
        bl = g.groupby(["layer", "code"]).role_conv.mean().idxmax()
        gb = g[(g.layer == bl[0]) & (g.code == bl[1])]
        # paper-style: layer of the asked-role (figure from qf; ground from qg) probe at the same state family
        rec = dict(sel_layer=int(sel[0]), sel_code=sel[1], conv=stat(gs, "role_conv"), can=stat(gs, "role_can"), conv_rec=stat(gs, "rec_conv")[0],
                   best_layer=int(bl[0]), best_code=bl[1], best_conv=stat(gb, "role_conv"))
        out[f"{m}|{st}|{tg}"] = rec
        print(f"{st:7s} -> {tg}: sel L{rec['sel_layer']}/{rec['sel_code']:7s} conv {rec['conv'][0]:.3f}+-{rec['conv'][1]:.3f} can {rec['can'][0]:.3f} | best L{rec['best_layer']}/{rec['best_code']} conv {rec['best_conv'][0]:.3f}+-{rec['best_conv'][1]:.3f}")
    # unasked role at the layer chosen for the asked role
    for st, asked, un in (("qf", "fig", "gnd"), ("qg", "gnd", "fig")):
        lc = sel_layer[(st, asked)]
        gs = gm[(gm.state == st) & (gm.target == un) & (gm.layer == lc[0]) & (gm.code == lc[1])]
        s = stat(gs, "role_conv"); c = stat(gs, "role_can")
        out[f"{m}|{st}|{un}|at_asked_layer"] = dict(layer=int(lc[0]), code=lc[1], conv=s, can=c)
        print(f"{st} unasked {un} at asked-role layer L{lc[0]}/{lc[1]}: conv {s[0]:.3f}+-{s[1]:.3f} can {c[0]:.3f}")
    # per-layer table for period / last / mean (figure target, phrase+carrier averaged by best code per layer)
    for st in ("period", "last", "mean", "qf"):
        g = gm[(gm.state == st) & (gm.target == "fig")]
        pl = g.groupby(["layer", "code"]).role_conv.mean().groupby("layer").max()
        out[f"{m}|{st}|fig|per_layer_best_code_conv"] = {int(k): float(v) for k, v in pl.items()}
        print(f"per-layer {st}->fig conv (best code):", " ".join(f"L{int(k)}:{v:.2f}" for k, v in pl.items()))
    out[f"{m}|rel_breakdown"] = {}
    for st, tg in (("qf", "fig"), ("qf", "gnd"), ("period", "fig"), ("qg", "gnd"), ("qg", "fig")):
        sel = sel_layer[(st, tg)]
        g = gm[(gm.state == st) & (gm.target == tg) & (gm.layer == sel[0]) & (gm.code == sel[1])]
        out[f"{m}|rel_breakdown"][f"{st}->{tg}"] = g.groupby("rel").role_conv.mean().round(3).to_dict()
json.dump(out, open(f"{d}/summary.json", "w", encoding="utf-8"), indent=1, default=float)
