"""Figure: interventions on the question state (E21, E21b)."""
import glob, json
import numpy as np
from figstyle import *

FIGDIR = "../recoverability-theory/figures"
NOTES = "../notes/audit/figs"


def load(d):
    return [r for f in glob.glob(f"results/{d}/*.json") for r in json.load(open(f, encoding="utf-8"))]


def tag(ax, x, y, s, c, **kw):
    ax.text(x, y, s, color=c, fontsize=6.4, **kw)


e21, e21b = load("e21"), load("e21b")
MODELS = [("microsoft/phi-4", "phi-4", 30), ("Qwen/Qwen3-8B", "Qwen3-8B", 27), ("allenai/OLMo-2-1124-13B", "OLMo-2 13B", 30)]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.15), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1.15, 1]))

ax = axs[0]
SH = {"phi-4": ("o", FIX), "Qwen3-8B": ("s", FIX), "OLMo-2 13B": ("D", FIX)}
for m, n, _ in MODELS:
    rows = [r for r in e21 if r["model"] == m and r["source"] == "reversed" and r["kind"] == "whole"]
    L = sorted({r["layer"] for r in rows})
    y = [np.mean([r["to_answer"] for r in rows if r["layer"] == l]) for l in L]
    ax.plot([0.25, 0.5, 0.75, 1], y, marker=SH[n][0], color=FIX, ms=3.2, lw=1.0, label=n)
    print(n, "whole by layer", dict(zip(L, np.round(y, 3))), "samefig whole", round(float(np.mean([r["to_answer"] for r in e21 if r["model"] == m and r["source"] == "samefig" and r["kind"] == "whole" and r["picked"]])), 3))
ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["1/4", "1/2", "3/4", "1"])
ax.set_ylim(-0.03, 1.25); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(0.18, 1.07)
ax.set_xlabel("patched layer, share of depth"); ax.set_ylabel("answers flipped")
ax.legend(loc="upper left", fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0, ncol=1, labelspacing=0.2)
panel(ax, "a", "The answer arrives late")

ax = axs[1]


def sweep(prefix, sizes):
    out = []
    for k in sizes:
        v = [np.mean([r["flip"] for r in e21b if r["model"] == m and r["layer"] == l and r["kind"] == f"{prefix}_{k}"]) for m, _, l in MODELS]
        out.append(np.mean(v))
    return np.array(out)


for pre, sizes, c, name, yt, va in (("diff", (16, 64, 256), FIX, "source minus\ntarget", 0.93, "bottom"),
                                    ("readout", (16, 64, 128), REP, "readout", 0.12, "center"),
                                    ("pc", (16, 64, 256), CTRL, "top\ncomponents", 0.50, "top"),
                                    ("random", (16, 64, 256), LGREY, "random", 0.10, "center")):
    y = sweep(pre, sizes)
    print(pre, np.round(y, 3))
    ax.plot(sizes, y, "o-", color=c, ms=3, lw=1.0, zorder=3)
ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256]); ax.set_xticklabels(["16", "64", "256"]); ax.minorticks_off()
ax.set_xlim(12, 1100); ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1])
d, r_, p, q = sweep("diff", (16, 64, 256)), sweep("readout", (16, 64, 128)), sweep("pc", (16, 64, 256)), sweep("random", (16, 64, 256))
tag(ax, 300, d[2], "source minus\ntarget", FIX, va="center"); tag(ax, 148, r_[2] - 0.03, "readout", REP, va="center")
tag(ax, 300, p[2] - 0.14, "top\ncomponents", DGREY, va="center"); tag(ax, 300, q[2] + 0.02, "random", GREY, va="center")
ax.set_xlabel("dimensions patched"); ax.set_ylabel("answers flipped")
panel(ax, "b", "A subspace suffices")

ax = axs[2]
K = [("allbut_readout_16", "all but the\nreadout", REP), ("allbut_diff_64", "all but source\nminus target", FIX)]
V = [np.mean([np.mean([r["flip"] for r in e21b if r["model"] == m and r["layer"] == l and r["kind"] == k]) for m, _, l in MODELS]) for k, _, _ in K]
print("allbut", {k[0]: round(float(v), 3) for k, v in zip(K, V)})
for i, m_ in enumerate(MODELS):
    vv = [np.mean([r["flip"] for r in e21b if r["model"] == m_[0] and r["layer"] == m_[2] and r["kind"] == k]) for k, _, _ in K]
    print(m_[1], np.round(vv, 3), "samefig diff64", round(float(np.mean([r["flip"] for r in e21b if r["model"] == m_[0] and r["layer"] == m_[2] and r["kind"] == "samefig_diff_64"])), 3))
ax.bar([0, 1], V, color=[k[2] for k in K], width=0.62)
ax.set_xticks([0, 1]); ax.set_xticklabels([k[1] for k in K], fontsize=6.4)
ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("answers flipped"); ax.set_xlabel("patched from the source")
panel(ax, "c", "The readout is not needed")
fig.savefig(f"{FIGDIR}/fig_intervene.pdf"); fig.savefig(f"{NOTES}/fig_intervene.png", dpi=170)
print("ok")
