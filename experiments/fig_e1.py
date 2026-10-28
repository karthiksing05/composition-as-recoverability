"""Figure: calibration of analytic modes of combination (E1)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

rows = []
for f in glob.glob("results/e1/*.json"):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
reg = df[df.readout != "nn_table"]
nn = df[df.readout == "nn_table"]

LABEL = {"concat": "concatenation", "tpr": "tensor product", "proj_m1.5": "projection, $m/kd$=1.5",
         "proj_m1": "projection, $m/kd$=1", "proj_m0.75": "projection, $m/kd$=0.75", "proj_m0.5": "projection, $m/kd$=0.5",
         "hrr": "HRR", "map_vsa": "MAP (VSA)", "weighted": "$x + 0.5y$", "tanh_m2": "tanh layer, $m/kd$=2",
         "tanh_m1": "tanh layer, $m/kd$=1", "tanh_m0.5": "tanh layer, $m/kd$=0.5", "sum": "$x + y$", "hadamard": "$x \\odot y$"}
CAT = {"concat": BLUE, "tpr": BLUE, "proj_m1.5": BLUE, "proj_m1": BLUE, "tanh_m2": BLUE,
       "proj_m0.75": GREY, "proj_m0.5": GREY, "hrr": GREY, "map_vsa": GREY, "weighted": GREY, "tanh_m1": GREY, "tanh_m0.5": GREY,
       "sum": ORANGE, "hadamard": ORANGE}
ORDER = list(LABEL)
# vocabulary-free ceiling for role 1 under the best linear readout (Prop. 3): rank-weighted
THEORY = {"concat": 1, "tpr": 1, "proj_m1.5": 1, "proj_m1": 1, "proj_m0.75": .75, "proj_m0.5": .5,
          "hrr": .5, "map_vsa": .5, "weighted": .8, "sum": .5}

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw=dict(wspace=0.55, hspace=0.75)); axs = axs.ravel()

# a: a lookup over the vocabulary identifies parts for every injective mode
ax = axs[0]
V = 32
s = nn[nn.V == V].groupby("op")["fresh_id"].agg(["mean", "sem"]).reindex(ORDER)
y = np.arange(len(ORDER))[::-1]
ax.barh(y, s["mean"], color=[CAT[o] for o in ORDER], height=0.7)
ax.set_yticks(y); ax.set_yticklabels([LABEL[o] for o in ORDER], fontsize=6.3)
ax.set_xlim(0, 1.02); ax.set_xticks([0, .5, 1]); ax.axvline(0.5, color=LGREY, lw=0.6, ls=":")
ax.set_xlabel("identification, given candidates")
panel(ax, "a", "Lookup decodes almost any mode")

# b: fresh-part recovery matches the rank ceiling
ax = axs[1]
g = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].agg(["mean", "sem"])
ax.plot([0, 1], [0, 1], color=LGREY, lw=0.8, zorder=0)
for op, th in THEORY.items():
    ax.errorbar(th, g.loc[op, "mean"], yerr=g.loc[op, "sem"], fmt="o", color=CAT[op], ms=4, elinewidth=0.6,
                mec="white", mew=0.5)
for (x0, y0, txt, ha) in ((0.98, 0.985, "concatenation, TPR,\nfull-rank projection", "right"),
                          (0.81, 0.79, "$x+0.5y$", "left"), (0.76, 0.665, "projection, $m/kd$=0.75", "left"),
                          (0.53, 0.43, "HRR, MAP,\nprojection $m/kd$=0.5", "left"), (0.47, 0.52, "$x+y$", "right")):
    ax.text(x0, y0, txt, fontsize=6.3, color=DGREY, ha=ha, va="center")
ax.set_xlim(0.35, 1.05); ax.set_ylim(0.35, 1.05)
ax.set_xlabel("theoretical linear ceiling"); ax.set_ylabel("recovery of fresh parts ($R^2$)")
panel(ax, "b", "Fresh parts expose the relation")

# c: a readout must be learned from enough parts
ax = axs[2]
for op, c, mk in (("concat", BLUE, "o"), ("proj_m0.5", GREY, "s"), ("sum", ORANGE, "^")):
    for split, ls in (("fam_r2", "--"), ("fresh_r2", "-")):
        sub = reg[(reg.op == op) & (reg.readout == "linear")].copy()
        sub[split] = sub[split].clip(lower=-0.2)
        t = sub.groupby("V")[split].agg(["mean", "sem"])
        band(ax, t.index, t["mean"], t["sem"], c, ls=ls, marker=mk if split == "fresh_r2" else None, ms=3)
ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64]); ax.set_xticklabels(["4", "16", "64"])
ax.axvline(16, color=LGREY, lw=0.6, ls=":"); ax.text(17, -0.15, "$V=d$", fontsize=6.3, color=DGREY)
for txt, c, yy in (("concatenation", BLUE, 0.93), ("projection, $m/kd$=0.5", GREY, 0.18), ("$x+y$", ORANGE, 0.62)):
    ax.text(150, yy, txt, color=c, fontsize=6.3, ha="right")
ax.set_ylim(-0.2, 1.05); ax.set_xlabel("vocabulary size $V$"); ax.set_ylabel("$R^2$ (dashed: familiar)")
panel(ax, "c", "Readouts need many parts")

# d: the readout class matters
ax = axs[3]
ops = ["concat", "tpr", "proj_m1", "tanh_m2", "proj_m0.5", "hrr", "weighted"]
t = reg[reg.V == 128].groupby(["op", "readout"])["fresh_r2"].mean().unstack()
x = np.arange(len(ops))
ax.bar(x - 0.2, t.loc[ops, "linear"], 0.38, color=BLUE, label="linear")
ax.bar(x + 0.2, t.loc[ops, "mlp"].clip(lower=0), 0.38, color=DGREY, label="MLP")
ax.set_xticks(x); ax.set_xticklabels(["concat", "TPR", "proj 1", "tanh 2", "proj .5", "HRR", "$x$+.5$y$"], rotation=55, ha="right", fontsize=6.3)
ax.set_ylabel("recovery of fresh parts ($R^2$)"); ax.set_ylim(0, 1.05); ax.legend(loc="upper right", handlelength=1)
panel(ax, "d", "Flexible readouts memorize")

fig.savefig(f"{FIGDIR}/fig_calibration.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_calibration.png", dpi=200)
print("ok")
