"""Figure: structural consequences of role-factorization in trained networks (E5b), with a case study."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

df = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5d/*.json")])
for k in ("pred", "true", "noswap"):
    df[f"rev_out_{k}"] = df[[f"rev_out_{k}_r1{o}" for o in "01"]].mean(1)
for t in ("swap", "rand", "randB1", "randB1cov", "natural", "natural1"):
    df[f"spill_{t}"] = df[f"ctl_{t}_dright_r1"] / df[f"ctl_{t}_dleft_r1"]
df["spill_swapB2"] = df["ctl_swapB2_dleft_r1"] / df["ctl_swapB2_dright_r1"]
d64 = df[df.V == 64]
case = np.load("results/case/glyph.npz")
NET, PRED = GREY, BLUE

fig = plt.figure(figsize=(6.5, 4.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1, 1], wspace=0.42, hspace=0.62)

# a: states predicted from the role maps decode as well as the network's own
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)
groups = [("rev_out_true", "rev_out_pred", "over"), ("rev_out_true", "rev_out_pred", "occ"),
          ("newrole_out_net_r1", "newrole_out_pred_r1", "over"), ("newrole_out_net_r1", "newrole_out_pred_r1", "occ")]
for g, (kn, kp, rel) in enumerate(groups):
    s = d64[d64.second == rel]
    for j, (k, c) in enumerate(((kn, NET), (kp, PRED))):
        ax.bar(g + (j - 0.5) * 0.38, s[k].mean(), 0.36, yerr=s[k].sem(), color=c, error_kw=dict(lw=0.5),
               label=(["network's own state", "predicted state"][j] if g == 0 else None))
ax.set_xticks(range(4)); ax.set_xticklabels(["over", "occlude", "over", "occlude"], fontsize=6.0)
for xg, t in ((0.5, "reversal"), (2.5, "new role")):
    ax.text(xg, -0.22, t, ha="center", va="top", fontsize=6.6, transform=ax.get_xaxis_transform())
ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("output $R^2$ ($V = 64$)")
ax.legend(loc="upper left", fontsize=5.4, handlelength=1.0, ncol=1, borderaxespad=0.1)
panel(ax, "a", "Predicted states decode well")

# b: case study of an interchange intervention
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("$\\bf{b}$  Case study: an interchange edit swaps one glyph and keeps the other", loc="left", pad=20)
COLS = [("swap_orig", "output before", DGREY), ("swap_donor", "donor", ORANGE), ("swap_edit", "after role-1 edit", BLUE),
        ("swap_target", "target", GREEN), ("swap_nat", "natural direction", RED)]
sub = gs[0, 1].subgridspec(3, 5, width_ratios=[2, 1, 2, 2, 2], wspace=0.12, hspace=0.12)
for c_, (key, lab, col) in enumerate(COLS):
    for r in range(3):
        iax = fig.add_subplot(sub[r, c_])
        iax.imshow(case[key][r], cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.1)
        if r == 0:
            iax.set_title(lab, fontsize=6.0, color=col, pad=3)
        if key in ("swap_edit", "swap_nat"):
            t_ = case["swap_target"][r]
            r2 = 1 - ((case[key][r] - t_) ** 2).sum() / ((case["swap_target"] - case["swap_target"].mean(0)) ** 2).sum((1, 2)).mean()
            iax.text(0.97, 0.04, f"{r2:.2f}", transform=iax.transAxes, ha="right", va="bottom", fontsize=5.4, color=col,
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))

# c: spill of each edit
ax = fig.add_subplot(gs[1, 0])
EDITS = [("swap", "role-1 edit", BLUE), ("swapB2", "role-2 edit", BLUE), ("natural1", "own response", INK),
         ("randB1cov", "$B_1$, code-like", "#7fb3d5"), ("randB1", "$B_1$, isotropic", GREY),
         ("natural", "natural direction", RED), ("rand", "random", GREY)]
for i, (k, lab, c) in enumerate(EDITS):
    v = d64[d64.second == "over"][f"spill_{k}"]
    o = d64[d64.second == "occ"][f"spill_{k}"].mean()
    ax.barh(-i, v.mean(), 0.62, xerr=v.sem(), color=c, error_kw=dict(lw=0.5))
    ax.plot(o, -i, "o", mfc="white", mec=c, ms=3.4, mew=0.9, zorder=3)
ax.set_yticks(-np.arange(len(EDITS))); ax.set_yticklabels([e[1] for e in EDITS], fontsize=6.0)
ax.axvline(1, color=LGREY, lw=0.6, ls=":")
ax.set_xlim(0, 1.6); ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_xlabel("spill into the other role")
ax.set_ylim(-6.6, 1.0)
ax.text(1.58, 0.55, "bars: over   rings: occlude", fontsize=5.6, color=DGREY, ha="right", va="center")
panel(ax, "c", "Edits stay local")

# d: learning a new glyph in role 1 only
ax = fig.add_subplot(gs[1, 1]); ygrid(ax)
conds = [("before", "before"), ("replay0", "one role"), ("replay1", "+ replay"), ("bothroles", "both roles"), ("addcode", "code added")]
for k_ in ("heldout", "role1"):
    df[f"newpart_{k_}_addcode"] = df[f"newpart_{k_}_before"]
s = df[(df.second == "over") & (df.V == 64)]
xs = np.arange(len(conds))
for j, (key, c, lab) in enumerate((("heldout", GREY, "old pairs"), ("role1", PRED, "role 1"), ("role2", ORANGE, "role 2"))):
    mu = [s[f"newpart_{key}_{t}"].mean() for t, _ in conds]; se = [s[f"newpart_{key}_{t}"].sem() for t, _ in conds]
    ax.bar(xs + (j - 1) * 0.27, mu, 0.26, yerr=se, color=c, label=lab, error_kw=dict(lw=0.5))
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in conds], fontsize=6.2)
ax.set_ylim(0.4, 1.17); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("output $R^2$")
ax.legend(fontsize=5.8, loc="upper center", handlelength=0.9, ncol=3, columnspacing=0.8, borderaxespad=0.1)
panel(ax, "d", "Fine-tuning on one role hurts the other; adding the code does not")

fig.savefig(f"{FIGDIR}/fig_structure.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_structure.png", dpi=220)
print("runs", len(df))
for k, *_ in EDITS:
    print(k, round(d64[d64.second == "over"][f"spill_{k}"].mean(), 3), round(d64[d64.second == "occ"][f"spill_{k}"].mean(), 3))
