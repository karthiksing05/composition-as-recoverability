"""Figure: depth (E3) and pretrained text encoders (E4)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

rows = []
for f in glob.glob("results/e3v2/*.json"):
    rows += json.load(open(f))
e3 = pd.DataFrame(rows)
raam = e3[e3.model == "raam"]
hrr = pd.DataFrame(json.load(open("results/e3hrr3/hrr_unitary.json")))  # one pair of unitary roles per height
hrr = hrr[hrr.split == "pool"]
# whitened part codes (each of the 16 components counts equally); see agg_r4.py e4w
e4all = pd.DataFrame([r for f in glob.glob("results/e4w/*.json") for r in json.load(open(f))])
e4all = e4all[e4all.n_fit == 165]
for k in ("rec_pairs", "rec_val", "rec_fresh", "id_fresh", "role_fresh"):
    e4all[k] = e4all[[f"{k}_role1", f"{k}_role2"]].mean(1)
own = e4all[e4all.codes_from == e4all.model]  # every model scored against its own part codes
asym, ctl = own[own.rel != "and"], own[own.rel == "and"]


def kappa(d, model, layer=None, code=None):
    """kappa at the layer and code chosen on validation nouns (asymmetric relations), per seed."""
    m = d[d.model == model]
    if layer is None:
        sel = m[m.system == "model"].groupby(["layer", "code"])["rec_val"].mean()
        layer, code = sel.idxmax()
    m = m[(m.layer == layer) & (m.code == code)]
    t = m.groupby(["seed", "system"])["rec_fresh"].mean().unstack()
    k = (t["model"] - t["ref_sum"]) / (t["ref_concat"] - t["ref_sum"])
    return k.mean(), k.sem(), layer, code


fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1.25, 1.25]))
fig.subplots_adjust(bottom=0.27)

# a: fresh-leaf recovery by depth for the learned recursive composer
ax = axs[0]
ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
for w, c in ((32, ORANGE), (64, GREY), (256, BLUE)):
    s = raam[(raam.V == 16) & (raam.w == w)]
    t = s.groupby("depth")["fresh_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="o", ms=3, label=f"$w$={w}")
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], c, ls="--")
    b = s.groupby("depth")["bound"].mean()
    ax.plot(b.index, b.values, color=c, ls=":", lw=0.8)
ax.set_ylim(-0.5, 1.05); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_ylabel("fresh leaves recovered ($R^2$)")
ax.text(1.5, -0.42, "trained", ha="center", fontsize=6.3, color=DGREY)
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.08))
panel(ax, "a", "Learned composer")

# b: HRR: correlation unbinding (solid) vs a population readout of the same state (dashed)
ax = axs[1]
for w, c in ((64, GREY), (256, BLUE), (1024, GREEN)):
    s = hrr[hrr.w == w]
    t = s.groupby("depth")["r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="s", ms=3, label=f"$w$={w}")
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], c, ls="--")
    t = s.groupby("depth")["crosstalk_pred"].mean()
    ax.plot(t.index, t.clip(lower=-0.5).values, color=c, ls=":", lw=0.8)
ax.set_ylim(-0.5, 1.05); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_yticklabels([])
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.0))
panel(ax, "b", "HRR unbinding")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=DGREY, ls="-", marker="o", ms=2.6, label="the system's decoder"),
                    Line2D([], [], color=DGREY, ls="--", label="population readout"),
                    Line2D([], [], color=DGREY, ls=":", lw=0.8, label="proven bound")],
           loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, fontsize=6.0, handlelength=2.0, frameon=False, columnspacing=1.2)

# c: text encoders, trained against random weights
NAMES = {"clip-b32": "CLIP-B", "sd15-clip-l": "CLIP-L", "clip-h14": "CLIP-H", "t5-small": "T5-s", "t5-base": "T5-b",
         "t5-large": "T5-l", "bert-base-uncased": "BERT-b", "bert-large-uncased": "BERT-l", "gpt2": "GPT2-s",
         "gpt2-medium": "GPT2-m", "gpt2-large": "GPT2-l"}
models = list(NAMES)
x = np.arange(len(models))
ax = axs[2]; ygrid(ax)
K = {}
for i, m in enumerate(models):
    K[m] = kappa(asym, m)
    kr = kappa(asym, "rand-" + m)
    ax.bar(i - 0.2, K[m][0], 0.38, yerr=K[m][1], color=BLUE, error_kw=dict(lw=0.5), label="trained" if i == 0 else None)
    ax.bar(i + 0.2, kr[0], 0.38, yerr=kr[1], color=LGREY, error_kw=dict(lw=0.5), label="random weights" if i == 0 else None)
ax.axhline(0, color=DGREY, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], rotation=60, fontsize=5.6)
ax.set_ylim(-0.25, 0.75); ax.set_ylabel(r"$\kappa$, fresh nouns")
ax.legend(loc="upper left", fontsize=5.6, handlelength=1, borderaxespad=0.1)
panel(ax, "c", "Training helps partway")

# d: the order-only control reaches the same kappa
ax = axs[3]; ygrid(ax)
for i, m in enumerate(models):
    _, _, layer, code = K[m]
    kc = kappa(ctl, m, layer, code)
    ax.plot([i, i], [K[m][0], kc[0]], color=LGREY, lw=1.2, zorder=1)
    ax.plot(i, K[m][0], "o", color=BLUE, ms=4, zorder=3, label="on, behind, ... (asymmetric)" if i == 0 else None)
    ax.plot(i, kc[0], "o", mfc="white", mec=DGREY, mew=1.0, ms=4, zorder=4, label="“a $x$ and a $y$” (order only)" if i == 0 else None)
ax.axhline(0, color=DGREY, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], rotation=60, fontsize=5.6)
ax.set_ylim(-0.25, 0.75); ax.set_ylabel(r"$\kappa$, fresh nouns")
ax.legend(loc="upper left", fontsize=5.6, handlelength=1, borderaxespad=0.1)
panel(ax, "d", "Order alone suffices")

fig.savefig(f"{FIGDIR}/fig_depth_pretrained.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_depth_pretrained.png", dpi=200)
print({m: [round(v, 3) for v in K[m][:2]] + list(K[m][2:]) for m in models})
