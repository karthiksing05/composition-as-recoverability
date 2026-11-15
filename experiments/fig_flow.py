"""The result figures in the order of the argument: behavior fails (toy, then pretrained models), then
recoverable states succeed (toy, role codes, pretrained recovery, roles).  One claim per figure, at most
four panels, direct labels in place of crowded legends."""
import json, glob, sys
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from figstyle import *
from agg_r2 import load as load2, e2cols
from agg_r7 import load as load7, e4style, ASYM

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
DEC, POP = BEH, REP
THREE = ["slot", "mlp", "cnn"]  # the transformer is reported in the text


def ygrid(ax):  # no background grid: nothing runs behind the labels
    pass


def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{NOTES}/{name}.png", dpi=170)


def jsons(d):
    return pd.DataFrame([r for f in glob.glob(f"results/{d}/*.json") for r in json.load(open(f, encoding="utf-8"))])


def short(df):
    for pre in ("EleutherAI/", "allenai/", "microsoft/", "Qwen/"):
        df["model"] = df.model.str.replace(pre, "")
    return df


def mean3(d, key, x):
    t = d[d.arch.isin(THREE)].groupby(["arch", x])[key].mean().unstack(0)
    return t.index.values, t.mean(1).values, t.std(1).values / np.sqrt(3)


def log2axis(ax, ticks):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()


def tag(ax, x, y, s, c, **kw):
    ax.text(x, y, s, fontsize=6.6, color=c, **kw)


# ------------------------------------------------------------------ data
e1 = pd.DataFrame([r for f in glob.glob("results/e1v2/gauss_*.json") for r in json.load(open(f))])
reg, nn = e1[e1.readout != "nn_table"], e1[e1.readout == "nn_table"]
e2 = e2cols(load2("e2v2"))
e3 = pd.DataFrame([r for f in glob.glob("results/e3v2/*.json") for r in json.load(open(f))])
raam = e3[(e3.model == "raam") & (e3.V == 16)]
e5 = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5d/*.json")])
case = np.load("results/case/glyph.npz")
e9 = pd.read_csv("results/agg_r8_e9.csv", index_col=0)
e16 = short(jsons("e16"))
E11 = {"phi-4": ("L30", "phrase"), "Qwen3-8B": ("L27", "carrier"), "OLMo-2-1124-13B": ("L30", "phrase"),
       "OLMo-2-1124-7B": ("L16", "carrier"), "gpt2-large": ("L18", "carrier"), "pythia-1.4b": ("L12", "carrier"),
       "pythia-12b": ("L18", "carrier")}
LMNAME = {"phi-4": "phi-4", "Qwen3-8B": "Qwen3-8B", "OLMo-2-1124-13B": "OLMo-2 13B", "OLMo-2-1124-7B": "OLMo-2 7B",
          "pythia-12b": "Pythia-12b", "pythia-1.4b": "Pythia-1.4b", "gpt2-large": "GPT-2 large"}
prompt = {}  # (model, variant) -> (answer accuracy, transfer) at the E11 layer and code
for (m, v), g in e16.groupby(["model", "variant"]):
    gg = g[(g.layer == E11[m][0]) & (g.code == E11[m][1])]
    prompt[m, v] = (gg.choice.mean(), gg.xfer_role_conv.mean())
e11 = short(jsons("e11"))
e11 = e11[(e11.system == "model") & (e11.state == "query")]
for m, (l, c) in E11.items():
    prompt[m, "main"] = (e9.loc[m, "choice"], e11[(e11.model == m) & (e11.layer == l) & (e11.code == c)].xfer_role_conv.mean())

mode = sys.argv[1] if len(sys.argv) > 1 else "all"

# ------------------------------------------------------------------ behavior fails, toy
if mode in ("all", "beh_toy"):
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.05), gridspec_kw=dict(wspace=0.5, width_ratios=[1.3, 1, 0.95, 1]))
    fig.subplots_adjust(bottom=0.2, left=0.13, right=0.99, top=0.86)
    ax = axs[0]
    OPS = [("concat", "concatenation", BLUE), ("tpr", "tensor product", BLUE), ("proj_m0.75", "projection 0.75×", GREY),
           ("weighted", "$x + 0.5y$", GREY), ("proj_m0.5", "projection 0.5×", GREY), ("hrr", "HRR", GREY), ("sum", "$x + y$", RED)]
    fam = nn[nn.V == 32].groupby("op")["fam_id"].mean()
    fr = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].mean()
    y = np.arange(len(OPS))[::-1]
    for yy, (op, _, c) in zip(y, OPS):
        ax.plot([fr[op], fam[op]], [yy, yy], color=c, lw=1.4, alpha=0.4)
        ax.plot(fam[op], yy, "o", mfc="white", mec=c, mew=1.0, ms=4.4, zorder=3)
        ax.plot(fr[op], yy, "o", color=c, ms=4.4, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([o[1] for o in OPS], fontsize=6.4)
    ax.set_xlim(0.2, 1.08); ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["", "0.5", "", "1"]); ax.set_ylim(-0.6, 7.7)
    ax.set_xlabel("parts recovered")
    ax.plot(0.26, 7.1, "o", mfc="white", mec=DGREY, ms=4); tag(ax, 0.3, 7.1, "training, lookup", DGREY, va="center")
    ax.plot(0.26, 6.45, "o", color=DGREY, ms=4)
    tag(ax, 0.3, 6.45, "held-out, readout", DGREY, va="center")
    panel(ax, "a", "A lookup hides blends")

    ax = axs[1]; ygrid(ax)
    d = e2[e2.m == 64]
    x, m_, e_ = mean3(d, "beh_heldout_1", "V"); band(ax, x, m_, e_, DEC, ls="--")
    x, f_, e_ = mean3(d, "beh_fresh_1", "V"); band(ax, x, f_, e_, DEC, marker="o", ms=2.8)
    tag(ax, 8, 1.1, "new pairs, training glyphs", DEC, ha="left", va="center")
    tag(ax, 13.5, 0.22, "held-out glyphs", DEC, ha="left", va="center")
    log2axis(ax, [8, 16, 32, 64]); ax.set_xlim(7, 72); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("training glyphs $V$"); ax.set_ylabel("network's output $R^2$")
    panel(ax, "b", "Held-out parts fail")

    ax = axs[2]; ygrid(ax)
    s = e5[(e5.second == "over") & (e5.V == 64)]
    for j, (k, c, lab) in enumerate((("heldout", GREY, "old pairs"), ("role2", ORANGE, "new glyph,\nother role"))):
        mu = [s[f"newpart_{k}_{t}"].mean() for t in ("before", "replay0")]; se = [s[f"newpart_{k}_{t}"].sem() for t in ("before", "replay0")]
        ax.bar(np.arange(2) + (j - 0.5) * 0.4, mu, 0.38, yerr=se, color=c, error_kw=dict(lw=0.5), label=lab)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "fine-tuned"], fontsize=6.6); ax.set_xlabel("adding a new glyph")
    ax.set_ylim(0.4, 1.3); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("network's output $R^2$")
    ax.legend(loc="upper center", ncol=2, fontsize=5.6, handlelength=0.9, columnspacing=0.7, borderaxespad=0, handletextpad=0.3)
    panel(ax, "c", "Adding parts harms")

    ax = axs[3]; ygrid(ax)
    ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
    for w, c in ((64, BEH2), (256, BEH)):
        t = raam[raam.w == w].groupby("depth")["fresh_r2"].agg(["mean", "sem"])
        band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="o", ms=2.8)
    tag(ax, 3.1, 0.92, "$w$ = 64", DGREY, ha="left", va="center"); tag(ax, 3.35, 0.55, "$w$ = 256", BEH, ha="left", va="center")
    ax.set_ylim(-0.5, 1.05); ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("tree depth $n$"); ax.set_ylabel("decomposer's $R^2$")
    panel(ax, "d", "Depth fails")
    save(fig, "fig_beh_toy")

# ------------------------------------------------------------------ behavior fails, pretrained models
if mode in ("all", "beh_real"):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.1), gridspec_kw=dict(wspace=0.5, width_ratios=[1.15, 0.85, 1.1]))
    fig.subplots_adjust(bottom=0.2, left=0.1, right=0.985, top=0.87)
    ax = axs[0]
    ORD = ["phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B", "pythia-12b", "pythia-1.4b", "gpt2-large"]
    y = np.arange(len(ORD))[::-1]
    ax.axvline(0.5, color=LGREY, lw=0.8, ls=":")
    for yy, m in zip(y, ORD):
        vs = [prompt[m, v][0] for v in ("reword", "zeroshot", "ground")] + [prompt[m, "main"][0]]
        ax.plot([min(vs), max(vs)], [yy, yy], color=LGREY, lw=2.2, solid_capstyle="round", zorder=1)
        ax.plot(prompt[m, "main"][0], yy, "o", mec=BEH, mew=0.9, mfc=BEH if prompt[m, "main"][0] > 0.7 else "white", ms=4.4, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LMNAME[m] for m in ORD], fontsize=6.4)
    ax.set_xlim(0.38, 1.03); ax.set_xticks([0.5, 0.75, 1]); ax.set_xticklabels(["0.5", "0.75", "1"]); ax.set_ylim(-0.6, 6.6)
    ax.set_xlabel("answer accuracy, held-out nouns")
    panel(ax, "a", "Answers about roles")

    enc = short(jsons("e17")).groupby("model")[["roles", "identity"]].mean()
    ax = axs[1]; ygrid(ax)
    EN = [("clip-b32", "CLIP\nB/32"), ("clip-h14", "CLIP\nH/14"), ("siglip-so400m", "SigLIP")]
    for j, (k, c) in enumerate((("identity", GREY), ("roles", ORANGE))):
        ax.bar(np.arange(3) + (j - 0.5) * 0.4, [enc.loc[m, k] for m, _ in EN], 0.38, color=c, label={"identity": "objects", "roles": "places"}[k])
    ax.axhline(0.5, color=INK, lw=0.7, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in EN], fontsize=6.2); ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("caption match accuracy")
    ax.legend(loc="upper center", ncol=2, fontsize=6, handlelength=0.9, borderaxespad=0, columnspacing=0.8, handletextpad=0.3)
    panel(ax, "b", "Image–text")

    txt = jsons("e18").groupby("model")[["roles", "identity"]].mean()
    ax = axs[2]; ygrid(ax)
    TX = [("clip-b32", "CLIP-B"), ("sd15-clip-l", "CLIP-L"), ("clip-h14", "CLIP-H"), ("t5-small", "T5-s"), ("t5-base", "T5-b"),
          ("t5-large", "T5-l"), ("bert-base-uncased", "BERT-b"), ("bert-large-uncased", "BERT-l")]
    Y8 = np.arange(8)[::-1]
    for j, (k, c) in enumerate((("identity", GREY), ("roles", ORANGE))):
        ax.barh(Y8 - (j - 0.5) * 0.4, [txt.loc[m, k] for m, _ in TX], 0.38, color=c, label={"identity": "nouns", "roles": "roles"}[k])
    ax.set_yticks(Y8); ax.set_yticklabels([l for _, l in TX], fontsize=6.2); ax.set_ylim(-0.6, 9.4); ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("paraphrase match accuracy")
    ax.legend(loc="upper center", ncol=2, fontsize=6, handlelength=0.9, borderaxespad=0, columnspacing=0.8, handletextpad=0.3)
    panel(ax, "c", "Text only")
    save(fig, "fig_beh_real")

# ------------------------------------------------------------------ recoverable states, toy
if mode in ("all", "rep_toy"):
    fig = plt.figure(figsize=(6.5, 3.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.42, bottom=0.11, top=0.95, left=0.07, right=0.99)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    panel(ax, "a", "One MLP: its state holds held-out glyphs that its decoder cannot render")
    ROWS = [("target", "target"), ("net", "network's output"), ("readout", "read from its state")]
    NK, TW, TH = 4, 0.092, 0.235
    YB = {"target": 0.655, "net": 0.345, "readout": 0.035}
    for b, V in enumerate((8, 64)):
        x0 = 0.15 + b * 0.45
        tgt = case[f"V{V}_target"][:NK]
        var = ((tgt - tgt.mean(0)) ** 2).sum((1, 2)).mean()
        for r, (key, lab) in enumerate(ROWS):
            im = case[f"V{V}_{key}"]; yb = YB[key]
            for k in range(NK):
                iax = ax.inset_axes([x0 + k * (TW + 0.006), yb, TW, TH])
                iax.imshow(im[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
                iax.set_xticks([]); iax.set_yticks([])
                col = {"net": DEC, "readout": POP}.get(key, LGREY)
                for sp in iax.spines.values():
                    sp.set_color(col); sp.set_linewidth(1.2 if key != "target" else 0.6)
                if key != "target":
                    r2 = 1 - ((im[k] - tgt[k]) ** 2).sum() / var
                    ax.text(x0 + k * (TW + 0.006) + TW / 2, yb - 0.012, f"{r2:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=col)
            if b == 0:
                ax.text(x0 - 0.012, yb + TH / 2, lab, ha="right", va="center", fontsize=6.8,
                        color={"net": DEC, "readout": POP}.get(key, DGREY), transform=ax.transAxes)
        ax.text(x0 + 2 * (TW + 0.006), 0.915, f"trained on $V = {V}$ glyphs", ha="center", va="bottom", fontsize=7, transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 0]); ygrid(ax)
    THEORY = {"concat": 1, "tpr": 1, "proj_m1.5": 1, "proj_m1": 1, "proj_m0.75": .75, "proj_m0.5": .5, "hrr": .5, "map_vsa": .5, "weighted": .8, "sum": .5}
    CAT = {"concat": BLUE, "tpr": BLUE, "proj_m1.5": BLUE, "proj_m1": BLUE, "sum": RED}
    g = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].agg(["mean", "sem"])
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.8, zorder=0)
    for op, th in THEORY.items():
        ax.plot(th, g.loc[op, "mean"], "o", color=CAT.get(op, GREY), ms=4.2, mec="white", mew=0.5)
    tag(ax, 0.88, 0.99, "role–filler", BLUE, ha="right"); tag(ax, 0.47, 0.57, "$x + y$", RED, ha="center")
    tag(ax, 0.84, 0.74, "$x + 0.5y$", DGREY, ha="left"); tag(ax, 0.55, 0.42, "HRR, MAP,\nhalf-rank", DGREY, ha="left", va="center", linespacing=1.05)
    ax.set_xlim(0.35, 1.08); ax.set_ylim(0.35, 1.05); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("proven ceiling"); ax.set_ylabel("held-out parts recovered ($R^2$)")
    panel(ax, "b", "Recovery meets the ceiling")

    ax = fig.add_subplot(gs[1, 1]); ygrid(ax)
    d = e2[e2.m == 64]
    x, m_, e_ = mean3(d, "pool_fresh_1", "V"); band(ax, x, m_, e_, POP, marker="o", ms=2.8)
    x, b_, e_ = mean3(d, "beh_fresh_1", "V"); band(ax, x, b_, e_, DEC, marker="o", ms=2.8)
    rd = pd.DataFrame([dict(arch=r["arch"], **{c: np.mean([r[c]["r10"], r[c]["r11"]]) for c in ("orig", "redec_trained")})
                       for r in (json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2redec/*.json"))])
    rd = rd[rd.arch.isin(THREE)].groupby("arch").mean().mean()
    ax.annotate("", xy=(8, rd.redec_trained), xytext=(8, rd.orig + 0.04), arrowprops=dict(arrowstyle="-|>", color=FIX, lw=0.7, mutation_scale=6))
    ax.plot([8], [rd.redec_trained], "s", color=FIX, mfc="white", ms=3.6, zorder=5)
    tag(ax, 9.2, rd.redec_trained, "new decoder", FIX, ha="left", va="center")
    tag(ax, 9, 0.93, "readout of the state", POP, ha="left"); tag(ax, 21, 0.45, "own decoder", DEC, ha="left")
    log2axis(ax, [8, 16, 32, 64]); ax.set_xlim(7, 72); ax.set_ylim(0, 1.05)
    ax.set_xlabel("training glyphs $V$"); ax.set_ylabel("$R^2$ on held-out glyphs")
    panel(ax, "c", "The state outlasts the decoder")

    ax = fig.add_subplot(gs[1, 2]); ygrid(ax)
    ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
    s = raam[raam.w == 256]
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"]); band(ax, t.index, t["mean"], t["sem"], POP, marker="o", ms=2.8)
    t = s.groupby("depth")["fresh_r2"].agg(["mean", "sem"]); band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], DEC, marker="o", ms=2.8)
    b = s.groupby("depth")["bound"].mean(); ax.plot(b.index, b.values, color=INK, ls=":", lw=0.9)
    tag(ax, 3.2, 0.85, "readout", POP, ha="left", va="top"); tag(ax, 3.1, 0.42, "own decomposer", DEC, ha="left", va="center")
    tag(ax, 3.9, 1.04, "bound", INK, ha="left", va="bottom")
    ax.set_ylim(-0.5, 1.2); ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("tree depth $n$ ($w = 256$)"); ax.set_ylabel("leaves recovered ($R^2$)")
    panel(ax, "d", "Depth, same state")
    save(fig, "fig_rep_toy")

# ------------------------------------------------------------------ the role codes
if mode in ("all", "rep_codes"):
    for k_ in ("pred", "true", "noswap"):
        e5[f"rev_out_{k_}"] = e5[[f"rev_out_{k_}_r1{o}" for o in "01"]].mean(1)
    for t in ("swap", "rand", "natural", "natural1"):
        e5[f"spill_{t}"] = e5[f"ctl_{t}_dright_r1"] / e5[f"ctl_{t}_dleft_r1"]
    d64 = e5[(e5.V == 64) & (e5.second == "over")]
    fig = plt.figure(figsize=(6.5, 3.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.05], width_ratios=[1, 1, 1.05], hspace=0.6, wspace=0.5, bottom=0.11, top=0.93, left=0.08, right=0.99)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    panel(ax, "a", "One MLP: an edit of the role-1 code swaps one glyph and keeps the other")
    COLS = [("swap_orig", "output before", DGREY), ("swap_donor", "donor glyph", DGREY), ("swap_edit", "after the edit", FIX),
            ("swap_target", "target", DGREY), ("swap_nat", "natural direction", CTRL)]
    W_ = 0.165
    for c_, (key, lab, col) in enumerate(COLS):
        for r in range(2):
            ww = W_ if key != "swap_donor" else W_ / 2
            x0 = 0.03 + sum((W_ if COLS[i][0] != "swap_donor" else W_ / 2) + 0.022 for i in range(c_))
            yb = 0.52 - r * 0.53
            iax = ax.inset_axes([x0, yb, ww, 0.36])
            iax.imshow(case[key][r], cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            iax.set_xticks([]); iax.set_yticks([])
            for sp in iax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.1)
            if r == 0:
                ax.text(x0 + ww / 2, yb + 0.385, lab, fontsize=6.6, color=col, ha="center", va="bottom", transform=ax.transAxes)
            if key in ("swap_edit", "swap_nat"):
                t_ = case["swap_target"][r]
                r2 = 1 - ((case[key][r] - t_) ** 2).sum() / ((case["swap_target"] - case["swap_target"].mean(0)) ** 2).sum((1, 2)).mean()
                ax.text(x0 + ww / 2, yb - 0.02, f"{r2:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=col)

    ax = fig.add_subplot(gs[1, 0]); ygrid(ax)
    groups = [("rev_out_true", "rev_out_pred"), ("newrole_out_net_r1", "newrole_out_pred_r1")]
    for g, (kn, kp) in enumerate(groups):
        for j, (k, c, lab) in enumerate(((kn, GREY, "own state"), (kp, FIX, "predicted"))):
            ax.bar(g + (j - 0.5) * 0.38, d64[k].mean(), 0.36, yerr=d64[k].sem(), color=c, error_kw=dict(lw=0.5), label=lab if g == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["reversal", "new role"], fontsize=6.8); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("decoder's output $R^2$")
    ax.legend(loc="upper center", ncol=2, fontsize=6, handlelength=0.9, borderaxespad=0, columnspacing=0.8, handletextpad=0.3)
    panel(ax, "b", "Predicted states decode")

    ax = fig.add_subplot(gs[1, 1])
    EDITS = [("swap", "role-1 edit", FIX), ("natural1", "own response", BEH), ("natural", "natural direction", GREY), ("rand", "random", LGREY)]
    for i, (k, lab, c) in enumerate(EDITS):
        v = d64[f"spill_{k}"]
        ax.barh(-i, v.mean(), 0.6, xerr=v.sem(), color=c, error_kw=dict(lw=0.5))
    ax.set_yticks(-np.arange(len(EDITS))); ax.set_yticklabels([e[1] for e in EDITS], fontsize=6.4)
    ax.set_xlim(0, 1.1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("spill into the other role")
    panel(ax, "c", "Edits stay local")

    ax = fig.add_subplot(gs[1, 2]); ygrid(ax)
    s = e5[(e5.second == "over") & (e5.V == 64)]
    conds = [("replay0", "fine-\ntuned"), ("replay1", "with\nreplay"), ("before", "code\nwritten")]
    for j, (k, c, lab) in enumerate((("heldout", GREY, "old pairs"), ("role2", ORANGE, "new glyph, other role"))):
        mu = [s[f"newpart_{k}_{t}"].mean() for t, _ in conds]; se = [s[f"newpart_{k}_{t}"].sem() for t, _ in conds]
        ax.bar(np.arange(3) + (j - 0.5) * 0.4, mu, 0.38, yerr=se, color=c, error_kw=dict(lw=0.5), label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in conds], fontsize=6.4, linespacing=1.0); ax.set_ylim(0.4, 1.3); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("decoder's output $R^2$")
    ax.legend(loc="upper center", ncol=1, fontsize=6, handlelength=0.9, borderaxespad=0, labelspacing=0.15, handletextpad=0.3)
    panel(ax, "d", "Writing a new code")
    save(fig, "fig_rep_codes")

# ------------------------------------------------------------------ recovery in pretrained models
if mode in ("all", "rep_real"):
    def kappa_seeds(df, rels):
        out, B = e4style(df, rels, "fig")
        per = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values="rec_fresh")
        ms = B[B.system == "model"].groupby(["model", "seed"]).rec_fresh.mean()
        out = ((ms - per.ref_sum) / (per.ref_concat - per.ref_sum)).rename("kappa").to_frame()
        k = "rec_fresh_role2"
        pr = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values=k)
        mr = B[B.system == "model"].groupby(["model", "seed"])[k].mean()
        out["kappa2"] = (mr - pr.ref_sum) / (pr.ref_concat - pr.ref_sum)
        return out.reset_index()

    def msem(k, m, col="kappa"):
        v = k[k.model == m][col]
        return v.mean(), (v.sem() if len(v) > 1 else 0.0)

    e6 = kappa_seeds(short(load7("e6")), ASYM); e10 = kappa_seeds(short(load7("e10")), ASYM)
    e6 = pd.concat([e6, e10[e10.model.str.startswith("pythia-1.4b")]])
    gr = pd.read_csv("results/agg_r8_e8grid.csv", index_col=[0, 1])
    e8 = kappa_seeds(load7("e8"), ["beside", "above", "overlap"])
    e4all = jsons("e4w"); e4all = e4all[e4all.n_fit == 165]
    for k in ("rec_val", "rec_fresh"):
        e4all[k] = e4all[[f"{k}_role1", f"{k}_role2"]].mean(1)
    own = e4all[(e4all.codes_from == e4all.model) & (e4all.rel != "and")]

    def kappa_txt(model):
        m = own[own.model == model]
        layer, code = m[m.system == "model"].groupby(["layer", "code"])["rec_val"].mean().idxmax()
        m = m[(m.layer == layer) & (m.code == code)]
        t = m.groupby(["seed", "system"])["rec_fresh"].mean().unstack()
        k = (t["model"] - t["ref_sum"]) / (t["ref_concat"] - t["ref_sum"])
        return k.mean(), k.sem()

    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.3), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1, 1, 1.15]))
    fig.subplots_adjust(bottom=0.2, left=0.075, right=0.99, top=0.88)
    ax = axs[0]; ygrid(ax)
    NM = {"clip-b32": "CLIP-B", "sd15-clip-l": "CLIP-L", "clip-h14": "CLIP-H", "t5-small": "T5-s", "t5-base": "T5-b", "t5-large": "T5-l",
          "bert-base-uncased": "BERT-b", "bert-large-uncased": "BERT-l", "gpt2": "GPT2-s", "gpt2-medium": "GPT2-m", "gpt2-large": "GPT2-l"}
    Y11 = np.arange(11)[::-1]
    for y_, m in zip(Y11, NM):
        a, b = kappa_txt(m), kappa_txt("rand-" + m)
        ax.barh(y_ + 0.2, a[0], 0.38, xerr=a[1], color=BLUE, error_kw=dict(lw=0.5), label="trained" if m == "clip-b32" else None)
        ax.barh(y_ - 0.2, b[0], 0.38, xerr=b[1], color=LGREY, error_kw=dict(lw=0.5), label="random" if m == "clip-b32" else None)
    ax.axvline(0, color=DGREY, lw=0.5, ymax=0.87); ax.axvline(1, color=INK, lw=0.8, ls=":", ymax=0.87)
    ax.set_yticks(Y11); ax.set_yticklabels(list(NM.values()), fontsize=5.8); ax.tick_params(axis="y", pad=1)
    ax.set_ylim(-0.6, 12.6); ax.set_xlim(-0.3, 1.05); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel(r"$\kappa$, held-out nouns")
    ax.legend(loc="upper center", fontsize=6, ncol=2, handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
    panel(ax, "a", "Text encoders")

    ax = axs[1]; ygrid(ax)
    PY = [("pythia-160m", 0.16), ("pythia-410m", 0.41), ("pythia-1b", 1.0), ("pythia-1.4b", 1.4), ("pythia-2.8b", 2.8), ("pythia-6.9b", 6.9), ("pythia-12b", 12)]
    PY = [(m, n) for m, n in PY if m in set(e6.model)]
    xx = np.array([n for _, n in PY])
    mu, se = zip(*[msem(e6, m) for m, _ in PY]); band(ax, xx, np.array(mu), np.array(se), BLUE, marker="o", ms=2.6)
    mu, se = zip(*[msem(e6, m, "kappa2") for m, _ in PY]); band(ax, xx, np.array(mu), np.array(se), BLUE, ls="--", marker="o", ms=2.6, mfc="white")
    for m, n in (("Qwen3-8B", 8.2), ("OLMo-2-1124-7B", 7.3), ("OLMo-2-1124-13B", 13.7), ("phi-4", 14.7)):
        a, _ = msem(e6, m); ax.plot(n, a, "s", color=REP2, ms=3.2, zorder=4)
        a, _ = msem(e6, m, "kappa2"); ax.plot(n, a, "s", color=REP2, mfc="white", ms=3.2, zorder=4)
    tag(ax, 0.13, 0.68, "both nouns", BLUE, ha="left"); tag(ax, 0.3, -0.13, "second noun", BLUE, ha="left"); tag(ax, 18, -0.02, "other\nmodels", REP2, ha="right", va="top", linespacing=1.05)
    ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"]); ax.minorticks_off()
    ax.set_xlabel("parameters (billions)"); ax.set_ylabel(r"$\kappa$, held-out nouns"); ax.set_ylim(-0.3, 1.05); ax.set_yticks([0, 0.5, 1]); ax.axhline(0, color=DGREY, lw=0.5); ax.axhline(1, color=INK, lw=0.8, ls=":")
    panel(ax, "b", "Scale stalls")

    ax = axs[2]; ygrid(ax)
    S410 = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 143000)
    k410 = np.array([[e10[e10.model == f"{r}@step{s}"].kappa.mean() for s in S410] for r in ["pythia-410m"] + [f"pythia-410m-seed{i}" for i in range(1, 6)]])
    xs = np.array([max(s, 100) for s in S410])
    ax.fill_between(xs, k410.min(0), k410.max(0), color=REP2, alpha=0.2, lw=0); ax.plot(xs, k410.mean(0), color=REP2, lw=1.0)
    ST = [(f"pythia-1.4b@step{s}", s) for s in (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000)] + [("pythia-1.4b", 143000)]
    ST = [(m, s) for m, s in ST if m in set(e6.model)]
    mu, se = zip(*[msem(e6, m) for m, _ in ST]); band(ax, np.array([max(s, 100) for _, s in ST]), np.array(mu), np.array(se), BLUE, marker="o", ms=2.6)
    tag(ax, 3000, 0.77, "1.4b", BLUE, ha="left"); tag(ax, 6000, 0.17, "410m, six runs", REP2, ha="center")
    ax.set_xscale("log"); ax.set_xticks([100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["0", "$10^3$", "$10^4$", "$10^5$"]); ax.minorticks_off()
    ax.set_xlabel("training step"); ax.set_ylabel(r"$\kappa$, held-out nouns"); ax.set_ylim(-0.3, 1.05); ax.set_yticks([0, 0.5, 1]); ax.axhline(0, color=DGREY, lw=0.5); ax.axhline(1, color=INK, lw=0.8, ls=":")
    panel(ax, "c", "Recovery peaks early")

    ax = axs[3]; ygrid(ax)
    ENC = [("dinov2-s", "DINOv2-S"), ("dinov2-l", "DINOv2-L"), ("dinov2-g", "DINOv2-g"), ("clip-b32-img", "CLIP-B"), ("clip-h14-img", "CLIP-H"), ("siglip-so400m", "SigLIP")]
    mu = [msem(e8, m)[0] for m, _ in ENC]; se = [msem(e8, m)[1] for m, _ in ENC]
    Y6 = np.arange(6)[::-1]
    ax.barh(Y6, mu, 0.6, xerr=se, color=BLUE, error_kw=dict(lw=0.5), label="average")
    ax.plot([gr.loc[(m, "grid"), "kappa"] for m, _ in ENC], Y6, "D", color=BLUE, mfc="white", ms=3.4, mew=0.9, ls="", label="quadrants", zorder=4)
    ax.axvline(0, color=DGREY, lw=0.5, ymax=0.84); ax.axvline(1, color=INK, lw=0.8, ls=":", ymax=0.84)
    ax.set_yticks(Y6); ax.set_yticklabels([l for _, l in ENC], fontsize=6); ax.tick_params(axis="y", pad=1)
    ax.set_ylim(-0.6, 6.9); ax.set_xlim(-0.05, 1.08); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel(r"$\kappa$, held-out photographs")
    ax.legend(loc="upper center", fontsize=6, ncol=2, handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
    panel(ax, "d", "Image encoders")
    save(fig, "fig_rep_real")

# ------------------------------------------------------------------ roles and the question state
if mode in ("all", "rep_roles"):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw=dict(wspace=0.42, width_ratios=[1.1, 1, 1.1]))
    fig.subplots_adjust(bottom=0.2, left=0.08, right=0.99, top=0.87)
    ax = axs[0]; ygrid(ax)
    e19 = short(jsons("e19"))
    M3 = [("phi-4", "phi-4"), ("Qwen3-8B", "Qwen3-8B"), ("OLMo-2-1124-13B", "OLMo-2 13B")]
    for s_, x0, c in (("mean", 0, LOST), ("query", 1, REP)):
        for off, (rd_, mk) in zip((-0.25, 0, 0.25), (("ridge", "o"), ("kernel", "s"), ("mlp", "D"))):
            for dm, (m, _) in zip((-0.06, 0, 0.06), M3):
                v = e19[(e19.model == m) & (e19.state == s_)][f"{rd_}_conv"].mean()
                ax.plot(x0 + off + dm, v, mk, color=c, ms=3.2, mec="white", mew=0.3)
    ax.axhline(0.5, color=LGREY, lw=0.8, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sentence\naverage", "end of\nthe question"], fontsize=6.6, linespacing=1.0); ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.04, 1.22); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("figure readout on converse")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color=DGREY, ms=3.2, label=l) for m, l in (("o", "ridge"), ("s", "kernel"), ("D", "MLP"))],
              loc="upper center", ncol=3, fontsize=6, handletextpad=0.1, columnspacing=0.5, borderaxespad=0)
    panel(ax, "a", "Where the role lives")

    ax = axs[1]
    for (m, v), (a, t) in prompt.items():
        c = REP if m in ("phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B") else LOST
        ax.plot(a, t, "o" if v == "main" else "o", color=c, ms=3.8 if v == "main" else 2.8, alpha=1 if v == "main" else 0.55, mec="white", mew=0.3)
    A = np.array([p[0] for p in prompt.values()]); T = np.array([p[1] for p in prompt.values()])
    r = np.corrcoef(A, T)[0, 1]
    tag(ax, 0.42, 1.0, f"$r$ = {r:.2f}", INK, ha="left", va="top"); tag(ax, 0.42, 0.88, "7 models, 4 prompts", DGREY, ha="left", va="top")
    ax.set_xlim(0.38, 1.03); ax.set_ylim(-0.04, 1.05); ax.set_xticks([0.5, 0.75, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("answer accuracy"); ax.set_ylabel("figure readout on converse")
    panel(ax, "b", "Answers follow the state")

    ax = axs[2]; ygrid(ax)
    d14 = pd.DataFrame([r for f in glob.glob("results/e14/*.json") for r in json.load(open(f, encoding="utf-8"))])
    d14["model"] = d14.model.map(lambda m: m.split("/")[-1])
    v = d14[d14.split == "val"].groupby(["model", "seed", "emb", "layer"]).pair.mean().reset_index()
    best = v.loc[v.groupby(["model", "seed", "emb"]).pair.idxmax(), ["model", "seed", "emb", "layer"]]
    f14 = d14[d14.split == "fresh"].merge(best, on=["model", "seed", "emb", "layer"]).groupby(["model", "emb"]).pair.mean()
    M5 = [("phi-4", "phi-4"), ("Qwen3-8B", "Qwen3"), ("OLMo-2-1124-13B", "OLMo\n13B"), ("pythia-1.4b", "Pythia\n1.4b"), ("gpt2-large", "GPT-2\nlarge")]
    for j, (e_, c, lab) in enumerate((("mean", LOST, "sentence average"), ("roleq", FIX, "two role questions"))):
        ax.bar(np.arange(5) + (j - 0.5) * 0.4, [f14[m, e_] for m, _ in M5], 0.38, color=c, label=lab)
    ax.axhline(1 / 3, color=INK, lw=0.7, ls=":")
    ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in M5], fontsize=6.0, linespacing=1.0); ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("paraphrase nearest, no readout")
    ax.legend(loc="upper center", ncol=1, fontsize=6, handlelength=0.9, borderaxespad=0, labelspacing=0.15, handletextpad=0.3)
    panel(ax, "c", "Embedding at the question")
    save(fig, "fig_rep_roles")
    print("r(answer, transfer)", round(r, 3), "n", len(A))
    print(f14.unstack().round(3).to_string())
    print(pd.Series(prompt).apply(lambda t: tuple(round(x, 2) for x in t)).to_string())
    print(e19.groupby(["model", "state"])[["ridge_conv", "kernel_conv", "mlp_conv"]].mean().round(3).to_string())
print("done", mode)
