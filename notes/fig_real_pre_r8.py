"""Figure: the E4 protocol on large pretrained models (E6 language, E7 roles, E8 vision)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *
from agg_r7 import load, e4style, ASYM


def short(df):
    df["model"] = df.model.str.replace("EleutherAI/", "").str.replace("allenai/", "").str.replace("microsoft/", "").str.replace("Qwen/", "")
    return df


def kappa_seeds(df, rels):
    """kappa per (model, seed) at the layer and code chosen on validation parts."""
    out, B = e4style(df, rels, "fig")
    per = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values="rec_fresh")
    ms = B[B.system == "model"].groupby(["model", "seed"]).rec_fresh.mean()
    out = ((ms - per.ref_sum) / (per.ref_concat - per.ref_sum)).rename("kappa").to_frame()
    for r in "12":  # each role against references for the same role
        k = f"rec_fresh_role{r}"
        pr = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values=k)
        mr = B[B.system == "model"].groupby(["model", "seed"])[k].mean()
        out[f"kappa{r}"] = (mr - pr.ref_sum) / (pr.ref_concat - pr.ref_sum)
    return out.reset_index()


def msem(k, m, col="kappa"):
    v = k[k.model == m][col]
    return v.mean(), (v.sem() if len(v) > 1 else 0.0)


e6 = kappa_seeds(short(load("e6")), ASYM)
e8 = kappa_seeds(load("e8"), ["beside", "above", "overlap"])

fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.35), gridspec_kw=dict(wspace=0.5, width_ratios=[1.1, 0.9, 1.3, 1.2]))
fig.subplots_adjust(bottom=0.24, left=0.07, right=0.99)

# a: scale (Pythia suite) and other large language models
ax = axs[0]; ygrid(ax)
PY = [("pythia-160m", 0.16), ("pythia-410m", 0.41), ("pythia-1b", 1.0), ("pythia-1.4b", 1.4), ("pythia-2.8b", 2.8), ("pythia-6.9b", 6.9), ("pythia-12b", 12)]
PY = [(m, n) for m, n in PY if m in set(e6.model)]
mu, se = zip(*[msem(e6, m) for m, _ in PY])
band(ax, np.array([n for _, n in PY]), np.array(mu), np.array(se), BLUE, marker="o", ms=2.8)
mu2, se2 = zip(*[msem(e6, m, "kappa2") for m, _ in PY])
band(ax, np.array([n for _, n in PY]), np.array(mu2), np.array(se2), BLUE, ls="--", marker="o", ms=2.8, mfc="white")
ax.text(1.0, 0.6, "Pythia", fontsize=6.2, color=BLUE)
OTHER = [("Qwen3-8B", 8.2, "Qwen3"), ("OLMo-2-1124-7B", 7.3, "OLMo 7B"), ("OLMo-2-1124-13B", 13.7, "OLMo 13B"), ("phi-4", 14.7, "phi-4")]
for m, n, lab in OTHER:
    if m in set(e6.model):
        a, b = msem(e6, m)
        ax.errorbar(n, a, yerr=b, color=ORANGE, marker="s", ms=3, lw=0, elinewidth=0.6, zorder=4)
        a, b = msem(e6, m, "kappa2")
        ax.errorbar(n, a, yerr=b, color=ORANGE, marker="s", ms=3, mfc="white", lw=0, elinewidth=0.6, zorder=4)
ax.text(9.5, -0.13, "other LLMs", fontsize=6.2, color=ORANGE, ha="center")
ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"]); ax.minorticks_off()
ax.set_xlabel("parameters (billions)"); ax.set_ylabel(r"$\kappa$, fresh nouns")
ax.set_ylim(-0.45, 1.05); ax.set_yticks([0, 0.5, 1]); ax.axhline(0, color=DGREY, lw=0.5)
ax.axhline(1, color=INK, lw=0.8, ls=":"); ax.text(0.14, 0.94, "concatenation", fontsize=5.8, color=INK, va="top")
panel(ax, "a", "Scale stalls")

# b: training trajectory of Pythia-1.4b; step 0 is the random-initialization control
ax = axs[1]; ygrid(ax)
ST = [("pythia-1.4b@step0", 0), ("pythia-1.4b@step512", 512), ("pythia-1.4b@step2000", 2000), ("pythia-1.4b@step8000", 8000),
      ("pythia-1.4b@step32000", 32000), ("pythia-1.4b", 143000)]
ST = [(m, s) for m, s in ST if m in set(e6.model)]
mu, se = zip(*[msem(e6, m) for m, _ in ST])
x = np.array([max(s, 100) for _, s in ST])
band(ax, x, np.array(mu), np.array(se), BLUE, marker="o", ms=2.8, label="both nouns")
mu2, se2 = zip(*[msem(e6, m, "kappa2") for m, _ in ST])
band(ax, x, np.array(mu2), np.array(se2), BLUE, ls="--", marker="o", ms=2.8, mfc="white", label="second noun")
ax.legend(fontsize=5.4, loc="lower right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
ax.set_xscale("log"); ax.set_xticks([100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["0", "$10^3$", "$10^4$", "$10^5$"]); ax.minorticks_off()
ax.set_xlabel("training step (Pythia-1.4b)"); ax.set_ylim(-0.45, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels([])
ax.axhline(0, color=DGREY, lw=0.5); ax.axhline(1, color=INK, lw=0.8, ls=":")
panel(ax, "b", "Training peaks early")

# c: positional vs semantic roles (E7), one row per model
ax = axs[2]
d7 = short(load("e7"))
SEL = [("gpt2-large", "GPT-2 L"), ("bert-large-uncased", "BERT L"), ("t5-large", "T5 L"), ("clip-h14", "CLIP-H"),
       ("pythia-1.4b@step0", "Pythia, step 0"), ("pythia-1.4b", "Pythia-1.4b"), ("pythia-12b", "Pythia-12b"),
       ("Qwen3-8B", "Qwen3-8B"), ("OLMo-2-1124-13B", "OLMo-2 13B"), ("phi-4", "phi-4")]
SEL = [(m, l) for m, l in SEL if m in set(d7.model)]
rows = []
for m, _ in SEL:
    g = d7[d7.model == m]; mm = g[g.system == "model"]
    lay, code = mm.groupby(["layer", "code"])[["pos_rec_val", "sem_rec_val"]].apply(lambda x: (x.pos_rec_val + x.sem_rec_val).mean()).idxmax()
    rows.append(mm[(mm.layer == lay) & (mm.code == code)].groupby("seed")[["pos_role_fresh", "sem_role_fresh", "xfer_role_conv"]].mean().assign(model=m))
R = pd.concat(rows)
KEYS = [("pos_role_fresh", BLUE, "o", "first-mentioned noun"), ("sem_role_fresh", ORANGE, "s", "figure, fit on both phrasings"),
        ("xfer_role_conv", RED, "D", "figure, fit on one phrasing")]
yy = -np.arange(len(SEL))
ax.axvline(0.5, color=LGREY, lw=0.8, ls=":")
for i, (m, _) in enumerate(SEL):
    v = [R[R.model == m][k].mean() for k, *_ in KEYS]
    ax.plot([min(v), max(v)], [yy[i]] * 2, color=LGREY, lw=1.0, zorder=1)
for k, c, mk, lab in KEYS:
    ax.plot([R[R.model == m][k].mean() for m, _ in SEL], yy, mk, color=c, ms=3.4, ls="", label=lab, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in SEL], fontsize=5.8)
ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(yy[-1] - 0.7, 3.5)
ax.set_xlabel("role accuracy, fresh nouns")
ax.legend(fontsize=5.4, loc="upper left", handlelength=0.8, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
panel(ax, "c", "Roles follow word order")

# d: image encoders, trained vs random initialization (E8)
ax = axs[3]; ygrid(ax)
ENC = [("dinov2-s", "DINOv2 S"), ("dinov2-l", "DINOv2 L"), ("dinov2-g", "DINOv2 g"), ("clip-b32-img", "CLIP B/32"),
       ("clip-h14-img", "CLIP H/14"), ("siglip-so400m", "SigLIP")]
for j, (pre, c, lab) in enumerate([("", BLUE, "trained"), ("rand-", LGREY, "random weights")]):
    mu, se = zip(*[msem(e8, pre + m) for m, _ in ENC])
    ax.bar(np.arange(len(ENC)) + (j - 0.5) * 0.4, mu, 0.38, yerr=se, color=c, label=lab, error_kw=dict(lw=0.5))
ax.axhline(0, color=DGREY, lw=0.5)
ax.set_xticks(range(len(ENC))); ax.set_xticklabels([l for _, l in ENC], rotation=60, ha="right", fontsize=5.8)
ax.set_ylabel(r"$\kappa$, fresh photographs"); ax.set_ylim(-0.4, 0.85); ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
ax.legend(fontsize=5.6, loc="upper left", handlelength=0.9, borderaxespad=0.1)
panel(ax, "d", "Photographs agree")

fig.savefig(f"{FIGDIR}/fig_real.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_real.png", dpi=200)
print(e6.groupby("model")[["kappa", "kappa1", "kappa2"]].mean().round(3).to_string())
print(e8.groupby("model")[["kappa", "kappa1", "kappa2"]].mean().round(3).to_string())
print(R.groupby("model").mean().round(3).to_string())
