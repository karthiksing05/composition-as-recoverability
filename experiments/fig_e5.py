"""Figure: structural consequences of role-factorization in trained networks (E5b)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

df = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5d/*.json")])
for k in ("pred", "true", "noswap"):
    df[f"rev_out_{k}"] = df[[f"rev_out_{k}_r1{o}" for o in "01"]].mean(1)
# spill: change in the untouched half of the output relative to the change in the edited half
for t in ("swap", "rand", "randB1", "randB1cov", "natural", "natural1"):
    df[f"spill_{t}"] = df[f"ctl_{t}_dright_r1"] / df[f"ctl_{t}_dleft_r1"]
df["spill_swapB2"] = df["ctl_swapB2_dleft_r1"] / df["ctl_swapB2_dright_r1"]
d64 = df[df.V == 64]
REL = (("over", "OVER"), ("occ", "OCCLUDE"))

fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.1), gridspec_kw=dict(wspace=0.72, width_ratios=[1, 1, 1.25, 1.35]))


def bars(ax, groups, series, colors, labels, ylab):
    n = len(series)
    for j, (key, c, lab) in enumerate(zip(series, colors, labels)):
        mu = [g[key].mean() for g in groups]; se = [g[key].sem() for g in groups]
        ax.bar(np.arange(len(groups)) + (j - (n - 1) / 2) * 0.8 / n, np.clip(mu, -0.5, None), 0.8 / n * 0.95,
               yerr=se, color=c, label=lab, error_kw=dict(lw=0.5))
    ax.axhline(0, color=DGREY, lw=0.4)
    ax.set_ylabel(ylab)


# a: reversal from a single composite
ax = axs[0]
bars(ax, [d64[d64.second == s] for s, _ in REL], ["rev_out_true", "rev_out_pred", "rev_out_noswap"],
     [GREY, BLUE, LGREY], ["true reversal", "predicted", "not reversed"], "output $R^2$")
ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in REL], fontsize=6.3); ax.set_ylim(-0.95, 1.5)
ax.legend(fontsize=5.6, loc="upper left", handlelength=0.9)
panel(ax, "a", "Reversal")

# b: a known glyph placed in a role it never occupied
ax = axs[1]
groups = [df[(df.second == s) & (df.V == V)] for s, _ in REL for V in (16, 64)]
bars(ax, groups, ["newrole_out_net_r1", "newrole_out_pred_r1"], [GREY, BLUE], ["network", "predicted"], "output $R^2$")
ax.set_xticks(range(4)); ax.set_xticklabels(["OV\n16", "OV\n64", "OC\n16", "OC\n64"], fontsize=6.0); ax.set_ylim(-0.1, 1.03)
ax.legend(fontsize=5.6, loc="upper left", handlelength=0.9)
panel(ax, "b", "New role")

# c: interchange intervention on role 1
ax = axs[2]
groups = [df[(df.second == s) & (df.V == V)] for s, _ in REL for V in (16, 64)]
bars(ax, groups, ["spill_swap", "spill_swapB2", "spill_natural1", "spill_randB1cov", "spill_natural", "spill_randB1", "spill_rand"],
     [BLUE, GREEN, PURPLE, "#7fb3d5", ORANGE, GREY, LGREY],
     ["role-1 swap", "role-2 swap", "own role-1 change", "$B_1$, code dist.", "natural", "$B_1$, isotropic", "random"],
     "spill into other role")
ax.set_xticks(range(4)); ax.set_xticklabels(["OV\n16", "OV\n64", "OC\n16", "OC\n64"], fontsize=6.0); ax.set_ylim(0, 2.45); ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.axhline(1, color=LGREY, lw=0.5, ls=":")
ax.legend(fontsize=4.8, loc="upper left", handlelength=0.8, ncol=2, columnspacing=0.4, labelspacing=0.25)
panel(ax, "c", "Interchange")

# d: learning a new glyph in role 1 only
ax = axs[3]
conds = [("before", "before"), ("replay0", "one\nrole"), ("replay1", "+replay"), ("bothroles", "both\nroles"), ("addcode", "code\nadded")]
# writing the code through B2 changes no weight: role 1 and old pairs keep their before values
for k_ in ("heldout", "role1"):
    s_ = df[f"newpart_{k_}_before"]; df[f"newpart_{k_}_addcode"] = s_
s = df[(df.second == "over") & (df.V == 64)]
xs = np.arange(len(conds))
for j, (key, c, lab) in enumerate((("heldout", GREY, "old pairs"), ("role1", BLUE, "role 1"), ("role2", ORANGE, "role 2"))):
    mu = [s[f"newpart_{key}_{t}"].mean() for t, _ in conds]; se = [s[f"newpart_{key}_{t}"].sem() for t, _ in conds]
    ax.bar(xs + (j - 1) * 0.27, mu, 0.26, yerr=se, color=c, label=lab, error_kw=dict(lw=0.5))
ax.set_xticks(xs); ax.set_xticklabels([l.replace("\n", " ") for _, l in conds], fontsize=5.4, rotation=40, ha="right"); ax.set_ylim(0.4, 1.2); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("output $R^2$")
ax.legend(fontsize=5.0, loc="upper center", handlelength=0.8, ncol=3, columnspacing=0.4)
panel(ax, "d", "Learning a glyph")

fig.savefig(f"{FIGDIR}/fig_structure.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_structure.png", dpi=200)
print("runs", len(df))
