"""Figure: calibration of analytic modes of combination (E1)."""
import json, glob
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from figstyle import *

rows = []
for f in glob.glob("results/e1v2/gauss_*.json"):  # fresh parts from a fixed 2,000-part pool
    rows += json.load(open(f))
df = pd.DataFrame(rows)
reg = df[df.readout != "nn_table"]
nn = df[df.readout == "nn_table"]

LABEL = {"concat": "concatenation", "tpr": "tensor product", "proj_m1.5": "projection 1.5×",
         "proj_m1": "projection 1×", "tanh_m2": "tanh layer 2×",
         "proj_m0.75": "projection 0.75×", "weighted": "$x + 0.5y$", "proj_m0.5": "projection 0.5×",
         "hrr": "HRR", "map_vsa": "MAP", "tanh_m1": "tanh layer 1×", "tanh_m0.5": "tanh layer 0.5×",
         "sum": "$x + y$", "hadamard": "$x \\odot y$"}
CAT = {"concat": BLUE, "tpr": BLUE, "proj_m1.5": BLUE, "proj_m1": BLUE, "tanh_m2": BLUE,
       "proj_m0.75": GREY, "proj_m0.5": GREY, "hrr": GREY, "map_vsa": GREY, "weighted": GREY, "tanh_m1": GREY, "tanh_m0.5": GREY,
       "sum": ORANGE, "hadamard": ORANGE}
ORDER = list(LABEL)
THEORY = {"concat": 1, "tpr": 1, "proj_m1.5": 1, "proj_m1": 1, "proj_m0.75": .75, "proj_m0.5": .5,
          "hrr": .5, "map_vsa": .5, "weighted": .8, "sum": .5}

fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.55), gridspec_kw=dict(wspace=0.42, width_ratios=[1.55, 1, 1]))

# a: familiar parts decode by lookup; fresh parts separate the operators
ax = axs[0]
fam = nn[nn.V == 32].groupby("op")["fam_id"].mean()
fr = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].mean()
ml = reg[(reg.V == 128) & (reg.readout == "mlp")].groupby("op")["fresh_r2"].mean()
y = np.arange(len(ORDER))[::-1]
for yy, op in zip(y, ORDER):
    c = CAT[op]
    ax.plot([fr[op], fam[op]], [yy, yy], color=c, lw=1.6, alpha=0.45, solid_capstyle="round")
    ax.plot(fam[op], yy, "o", mfc="white", mec=c, mew=1.1, ms=4.2, zorder=3)
    ax.plot(fr[op], yy, "o", color=c, ms=4.2, zorder=4)
    if op in ml and ml[op] > -0.05:
        ax.plot(ml[op], yy, "x", color=DGREY, ms=3.6, mew=0.9, zorder=5)
ax.set_yticks(y); ax.set_yticklabels([LABEL[o] for o in ORDER], fontsize=6.4)
for t, op in zip(ax.get_yticklabels(), ORDER):
    t.set_color(CAT[op] if CAT[op] != GREY else DGREY)
ax.set_xlim(-0.02, 1.04); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-0.7, len(ORDER) - 0.3)
ax.axvline(0.5, color=LGREY, lw=0.6, ls=":", zorder=0)
ax.set_xlabel("parts recovered"); ax.set_ylabel("operator")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=DGREY, label="familiar parts, lookup"),
                   Line2D([], [], marker="o", ls="", color=DGREY, label="fresh parts, affine readout"),
                   Line2D([], [], marker="x", ls="", color=DGREY, label="fresh parts, MLP readout")],
          loc="upper left", fontsize=5.8, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0))
panel(ax, "a", "A vocabulary hides blends")

# b: fresh-part recovery meets the rank ceiling
ax = axs[1]
g = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].agg(["mean", "sem"])
ax.plot([0, 1], [0, 1], color=LGREY, lw=0.8, zorder=0)
for op, th in THEORY.items():
    ax.errorbar(th, g.loc[op, "mean"], yerr=g.loc[op, "sem"], fmt="o", color=CAT[op], ms=4, elinewidth=0.6, mec="white", mew=0.5)
for (x0, y0, txt, ha, c) in ((0.97, 0.93, "role–filler", "right", BLUE), (0.83, 0.79, "$x + 0.5y$", "left", DGREY),
                             (0.78, 0.68, "0.75×", "left", DGREY), (0.54, 0.43, "HRR, MAP, 0.5×", "left", DGREY),
                             (0.47, 0.53, "$x + y$", "right", ORANGE)):
    ax.text(x0, y0, txt, fontsize=6.2, color=c, ha=ha, va="center")
ax.set_xlim(0.35, 1.05); ax.set_ylim(0.35, 1.05); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("proven ceiling (Prop. 3)"); ax.set_ylabel("fresh parts recovered ($R^2$)")
panel(ax, "b", "Recovery meets the ceiling")

# c: a readout must be learned from enough parts
ax = axs[2]
sub = reg[(reg.op == "concat") & (reg.readout == "ols_raw")].copy()
for split, ls, lab in (("fam_r2", "--", "familiar parts"), ("fresh_r2", "-", "fresh parts")):
    sub[split] = sub[split].clip(lower=-0.2)
    t = sub.groupby("V")[split].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], BLUE, ls=ls, marker="o" if split == "fresh_r2" else None, ms=3, label=lab)
Vg = np.array([4, 6, 8, 12, 16, 17, 24, 32, 64, 128]); dd = 16
cf = np.minimum(Vg - 1, dd) / dd - np.maximum(dd - Vg + 1, 0) / (Vg * dd)
ax.plot(Vg, cf, color=INK, ls=":", lw=1.0, label="closed form (Prop. 5)")
ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64]); ax.set_xticklabels(["4", "16", "64"])
ax.axvline(17, color=LGREY, lw=0.6, ls=":"); ax.text(19, 0.52, "$V = d + 1$", fontsize=6.2, color=DGREY)
ax.set_ylim(-0.2, 1.08); ax.set_xlabel("training parts $V$ ($d = 16$)"); ax.set_ylabel("recovery ($R^2$), concatenation")
ax.legend(loc="lower right", fontsize=5.8, handlelength=1.6)
panel(ax, "c", "Readouts need many parts")

fig.savefig(f"{FIGDIR}/fig_calibration.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_calibration.png", dpi=220)
print(pd.DataFrame({"fam": fam, "fresh": fr, "mlp": ml}).round(3).reindex(ORDER).to_string())
