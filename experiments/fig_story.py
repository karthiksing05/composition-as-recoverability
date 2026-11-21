"""Result figures in the order of the story (notes/PLAN_STORY.md).

One color per kind of measurement, in every figure:
  orange  weak composition: the system's own procedure (decoder, decomposer, answers, caption match)
  blue    strong composition: a declared readout of the state (R^2, kappa, transfer)
  green   decomposition put to use: procedures that read through the state (new decoder, edits, patches)
  grey    controls and references: input, random weights, chance, ceilings (light grey = random weights)
  red     only the blend endpoint and word-order outcomes
Filled markers are held-out parts and open markers training parts; a light shade is a second series
of the same kind."""
import json, glob, sys, collections
import numpy as np, pandas as pd
from figstyle import *
from agg_r2 import load as load2, e2cols
from agg_r7 import load as load7, e4style, ASYM

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
THREE = ["slot", "mlp", "cnn"]
FIX2 = "#a8d5b4"


def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{NOTES}/{name}.png", dpi=170)


def jsons(d):
    out = []
    for f in glob.glob(f"results/{d}/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return out


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
    kw.setdefault("fontsize", 6.6)
    ax.text(x, y, s, color=c, **kw)


def ref_lines(ax, lo=True, hi=True, axis="y"):
    f = ax.axhline if axis == "y" else ax.axvline
    if lo:
        f(0, color=DGREY, lw=0.5)
    if hi:
        f(1, color=INK, lw=0.8, ls=":")


mode = sys.argv[1] if len(sys.argv) > 1 else "all"
e1 = pd.DataFrame([r for f in glob.glob("results/e1v2/gauss_*.json") for r in json.load(open(f))])
reg, nn = e1[e1.readout != "nn_table"], e1[e1.readout == "nn_table"]
e2 = e2cols(load2("e2v2"))
e3 = pd.DataFrame([r for f in glob.glob("results/e3v2/*.json") for r in json.load(open(f))])
raam = e3[(e3.model == "raam") & (e3.V == 16)]
e5 = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5d/*.json")])
case = np.load("results/case/glyph.npz")
grid = pd.read_csv("results/agg_r8_e8grid.csv")

# ================================================================== beat 2: measuring recoverability
if mode in ("all", "measure"):
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.15), gridspec_kw=dict(wspace=0.55, width_ratios=[1.25, 1]))
    fig.subplots_adjust(bottom=0.2, left=0.16, right=0.97, top=0.86)
    ax = axs[0]
    OPS = [("concat", "concatenation"), ("tpr", "tensor product"), ("proj_m0.75", "projection 0.75×"),
           ("weighted", "$x + 0.5y$"), ("proj_m0.5", "projection 0.5×"), ("hrr", "HRR"), ("sum", "$x + y$")]
    fam = nn[nn.V == 32].groupby("op")["fam_id"].mean()
    fr = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].mean()
    y = np.arange(len(OPS))[::-1]
    for yy, (op, _) in zip(y, OPS):
        ax.plot([fr[op], fam[op]], [yy, yy], color=LGREY, lw=1.4, zorder=1)
        ax.plot(fam[op], yy, "o", mfc="white", mec=BEH, mew=1.0, ms=4.4, zorder=3)
        ax.plot(fr[op], yy, "o", color=RED if op == "sum" else REP, ms=4.4, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([o[1] for o in OPS], fontsize=6.4)
    ax.set_xlim(0.2, 1.08); ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["", "0.5", "", "1"]); ax.set_ylim(-0.6, 8.0)
    ax.set_xlabel("parts recovered (accuracy or $R^2$)")
    ax.plot(0.24, 7.45, "o", mfc="white", mec=BEH, ms=4); tag(ax, 0.28, 7.45, "lookup, training parts", BEH, va="center")
    ax.plot(0.24, 6.75, "o", color=REP, ms=4); tag(ax, 0.28, 6.75, "linear probe, new parts", REP, va="center")
    panel(ax, "a", "Training parts hide the blends")

    ax = axs[1]
    THEORY = {"concat": 1, "tpr": 1, "proj_m1.5": 1, "proj_m1": 1, "proj_m0.75": .75, "proj_m0.5": .5, "hrr": .5, "map_vsa": .5, "weighted": .8, "sum": .5}
    g = reg[(reg.V == 128) & (reg.readout == "linear")].groupby("op")["fresh_r2"].agg(["mean", "sem"])
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.8, zorder=0)
    for op, th in THEORY.items():
        ax.plot(th, g.loc[op, "mean"], "o", color=RED if op == "sum" else REP, ms=4.2, mec="white", mew=0.5)
    tag(ax, 0.88, 0.99, "role–filler", REP, ha="right"); tag(ax, 0.47, 0.57, "$x + y$", RED, ha="center")
    tag(ax, 0.84, 0.74, "$x + 0.5y$", REP, ha="left"); tag(ax, 0.55, 0.42, "HRR, MAP,\nhalf-rank", REP, ha="left", va="center", linespacing=1.05)
    ax.set_xlim(0.35, 1.08); ax.set_ylim(0.35, 1.05); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("proven ceiling"); ax.set_ylabel("new parts recovered ($R^2$)")
    panel(ax, "b", "Recovery meets the ceiling")
    save(fig, "fig_measure")

# ================================================================== beat 3: strong without weak
if mode in ("all", "strong_only"):
    fig = plt.figure(figsize=(6.5, 3.85))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.5, bottom=0.11, top=0.95, left=0.075, right=0.99)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    panel(ax, "a", "One MLP trained on 8 glyphs: its state holds new glyphs that its decoder cannot render")
    ROWS = [("target", "target"), ("net", "own decoder"), ("readout", "probe of the state")]
    NK, TW, TH = 4, 0.092, 0.235
    YB = {"target": 0.655, "net": 0.345, "readout": 0.035}
    for b, V in enumerate((8, 64)):
        x0 = 0.15 + b * 0.45
        tgt = case[f"V{V}_target"][:NK]
        var = ((tgt - tgt.mean(0)) ** 2).sum((1, 2)).mean()
        for key, lab in ROWS:
            im = case[f"V{V}_{key}"]; yb = YB[key]
            for k in range(NK):
                iax = ax.inset_axes([x0 + k * (TW + 0.006), yb, TW, TH])
                iax.imshow(im[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
                iax.set_xticks([]); iax.set_yticks([])
                col = {"net": BEH, "readout": REP}.get(key, LGREY)
                for sp in iax.spines.values():
                    sp.set_color(col); sp.set_linewidth(1.2 if key != "target" else 0.6)
                if key != "target":
                    r2 = 1 - ((im[k] - tgt[k]) ** 2).sum() / var
                    ax.text(x0 + k * (TW + 0.006) + TW / 2, yb - 0.012, f"{r2:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=col)
            if b == 0:
                ax.text(x0 - 0.012, yb + TH / 2, lab, ha="right", va="center", fontsize=6.8, color={"net": BEH, "readout": REP}.get(key, DGREY), transform=ax.transAxes)
        ax.text(x0 + 2 * (TW + 0.006), 0.915, f"trained on $V = {V}$ glyphs", ha="center", va="bottom", fontsize=7, transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 0])
    d = e2[e2.m == 64]
    x, m_, e_ = mean3(d, "pool_fresh_1", "V"); band(ax, x, m_, e_, REP, marker="o", ms=2.8)
    x, b_, e_ = mean3(d, "beh_fresh_1", "V"); band(ax, x, b_, e_, BEH, marker="o", ms=2.8)
    x, n_, e_ = mean3(d, "beh_heldout_1", "V"); band(ax, x, n_, e_, BEH, ls="--", marker="o", ms=2.8, mfc="white")
    tag(ax, 8, 1.13, "own decoder, new pairs of training glyphs", BEH, ha="left", va="center", fontsize=6.0)
    tag(ax, 66, 0.1, "probe of\nthe state", REP, ha="right", va="center", linespacing=1.0); tag(ax, 19, 0.42, "own decoder,\nnew glyphs", BEH, ha="left", va="center", linespacing=1.0)
    log2axis(ax, [8, 16, 32, 64]); ax.set_xlim(7, 72); ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("training glyphs $V$"); ax.set_ylabel("$R^2$ on wholes")
    panel(ax, "b", "The state outlasts the decoder")

    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
    s = raam[raam.w == 256]
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"]); band(ax, t.index, t["mean"], t["sem"], REP, marker="o", ms=2.8)
    t = s.groupby("depth")["fresh_r2"].agg(["mean", "sem"]); band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], BEH, marker="o", ms=2.8)
    bd = s.groupby("depth")["bound"].mean(); ax.plot(bd.index, bd.values, color=DGREY, ls=":", lw=0.9)
    tag(ax, 3.2, 0.85, "probe", REP, ha="left", va="top"); tag(ax, 3.1, 0.42, "own decomposer", BEH, ha="left", va="center")
    tag(ax, 3.9, 1.04, "bound", DGREY, ha="left", va="bottom"); tag(ax, 1.5, -0.42, "trained\ndepths", DGREY, ha="center", va="bottom", linespacing=1.0)
    ax.set_ylim(-0.5, 1.2); ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("tree depth $n$ ($w = 256$)"); ax.set_ylabel("new leaves ($R^2$)")
    panel(ax, "c", "Depth: the state keeps the leaves")

    ax = fig.add_subplot(gs[1, 2])
    enc = short(pd.DataFrame(jsons("e17"))).groupby("model")["roles"].mean()
    ENC = [("clip-b32-img", "clip-b32", "CLIP-B"), ("clip-h14-img", "clip-h14", "CLIP-H"), ("siglip-so400m", "siglip-so400m", "SigLIP"),
           ("dinov2-s", None, "DINOv2-S"), ("dinov2-l", None, "DINOv2-L"), ("dinov2-g", None, "DINOv2-g")]
    gq = grid[grid.pool == "grid"].set_index("model")
    Y6 = np.arange(6)[::-1]
    ax.axvline(0.5, color=LGREY, lw=0.8, ls=":", ymax=0.78)
    for yy, (gm, em, lab) in zip(Y6, ENC):
        ax.plot(gq.loc[gm, "kappa"], yy, "o", color=REP, ms=4.2, zorder=3)
        if em:
            ax.plot(enc[em], yy, "o", color=BEH, ms=4.2, zorder=3)
            ax.plot([enc[em], gq.loc[gm, "kappa"]], [yy, yy], color=LGREY, lw=1.2, zorder=1)
    ax.set_yticks(Y6); ax.set_yticklabels([e[2] for e in ENC], fontsize=6.2); ax.tick_params(axis="y", pad=1)
    ax.set_ylim(-0.6, 7.3); ax.set_xlim(0.3, 1.04); ax.set_xticks([0.5, 0.75, 1]); ax.set_xticklabels(["0.5", "0.75", "1"])
    ax.set_xlabel("score on new photographs")
    tag(ax, 0.32, 6.65, "caption places (chance 0.5)", BEH, va="center", fontsize=6.0); tag(ax, 0.32, 6.05, r"$\kappa$ of the quadrant state", REP, va="center", fontsize=6.0)
    panel(ax, "d", "Images: states hold places that matching misses")
    save(fig, "fig_strong_only")
    print("e17 roles", enc.round(3).to_dict())

# ================================================================== beat 4: decomposition at work
if mode in ("all", "decomp"):
    for t in ("swap", "rand", "natural", "natural1"):
        e5[f"spill_{t}"] = e5[f"ctl_{t}_dright_r1"] / e5[f"ctl_{t}_dleft_r1"]
    d64 = e5[(e5.V == 64) & (e5.second == "over")]
    fig = plt.figure(figsize=(6.5, 3.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.05], width_ratios=[1.05, 1, 1.05], hspace=0.62, wspace=0.55, bottom=0.11, top=0.93, left=0.08, right=0.99)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    panel(ax, "a", "One MLP: writing a donor's vector into role 1 swaps that glyph and keeps the other")
    COLS = [("swap_orig", "output before", DGREY), ("swap_donor", "donor glyph", DGREY), ("swap_edit", "after the edit", FIX),
            ("swap_target", "target", DGREY), ("swap_nat", "natural direction", GREY)]
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
                sp.set_color(col if key in ("swap_edit", "swap_nat") else LGREY); sp.set_linewidth(1.1)
            if r == 0:
                ax.text(x0 + ww / 2, yb + 0.385, lab, fontsize=6.6, color=col, ha="center", va="bottom", transform=ax.transAxes)
            if key in ("swap_edit", "swap_nat"):
                t_ = case["swap_target"][r]
                r2 = 1 - ((case[key][r] - t_) ** 2).sum() / ((case["swap_target"] - case["swap_target"].mean(0)) ** 2).sum((1, 2)).mean()
                ax.text(x0 + ww / 2, yb - 0.02, f"{r2:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=col)

    ax = fig.add_subplot(gs[1, 0])
    s = d64
    conds = [("replay0", "fine-\ntuned", BEH, BEH2), ("replay1", "with\nreplay", GREY, LGREY), ("before", "vector\nwritten", FIX, FIX2)]
    for i, (k, lab, c1, c2) in enumerate(conds):
        for j, (part, c) in enumerate((("heldout", c1), ("role2", c2))):
            v = s[f"newpart_{part}_{k}"]
            ax.bar(i + (j - 0.5) * 0.4, v.mean(), 0.38, yerr=v.sem(), color=c, error_kw=dict(lw=0.5))
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in conds], fontsize=6.4, linespacing=1.0)
    ax.set_ylim(0.4, 1.22); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("decoder's output $R^2$")
    tag(ax, -0.45, 1.16, "dark: old pairs", DGREY, va="center", fontsize=6.0); tag(ax, -0.45, 1.08, "light: new glyph, other role", DGREY, va="center", fontsize=6.0)
    panel(ax, "b", "A written vector adds a part")

    ax = fig.add_subplot(gs[1, 1])
    EDITS = [("swap", "role-1 edit", FIX), ("natural1", "own response", BEH), ("natural", "natural direction", GREY), ("rand", "random", LGREY)]
    for i, (k, lab, c) in enumerate(EDITS):
        v = d64[f"spill_{k}"]
        ax.barh(-i, v.mean(), 0.6, xerr=v.sem(), color=c, error_kw=dict(lw=0.5))
    ax.set_yticks(-np.arange(len(EDITS))); ax.set_yticklabels([e[1] for e in EDITS], fontsize=6.4)
    ax.set_xlim(0, 1.1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("change in the other role")
    panel(ax, "c", "Edits stay in their role")

    ax = fig.add_subplot(gs[1, 2])
    d14 = pd.DataFrame(jsons("e14")); d14["model"] = d14.model.map(lambda m: m.split("/")[-1])
    v = d14[d14.split == "val"].groupby(["model", "seed", "emb", "layer"]).pair.mean().reset_index()
    best = v.loc[v.groupby(["model", "seed", "emb"]).pair.idxmax(), ["model", "seed", "emb", "layer"]]
    f14 = d14[d14.split == "fresh"].merge(best, on=["model", "seed", "emb", "layer"]).groupby(["model", "emb"]).pair.mean()
    M5 = [("phi-4", "phi-4"), ("Qwen3-8B", "Qwen3"), ("OLMo-2-1124-13B", "OLMo\n13B"), ("pythia-1.4b", "Pythia\n1.4b"), ("gpt2-large", "GPT-2\nlarge")]
    for j, (e_, c, lab) in enumerate((("mean", RED, "sentence average"), ("roleq", FIX, "two role questions"))):
        ax.bar(np.arange(5) + (j - 0.5) * 0.4, [f14[m, e_] for m, _ in M5], 0.38, color=c, label=lab)
    ax.axhline(1 / 3, color=DGREY, lw=0.7, ls=":")
    ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in M5], fontsize=6.0, linespacing=1.0); ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("paraphrase nearest")
    ax.legend(loc="upper center", ncol=1, fontsize=6, handlelength=0.9, borderaxespad=0, labelspacing=0.15, handletextpad=0.3)
    panel(ax, "d", "The question state as an embedding")
    save(fig, "fig_decomp")

# ================================================================== beat 5: where recoverability comes from
if mode in ("all", "origins"):
    def kappa_seeds(df, rels):
        out, B = e4style(df, rels, "fig")
        per = B[B.system != "model"].pivot_table(index=["model", "seed"], columns="system", values="rec_fresh")
        ms = B[B.system == "model"].groupby(["model", "seed"]).rec_fresh.mean()
        return ((ms - per.ref_sum) / (per.ref_concat - per.ref_sum)).rename("kappa").to_frame().reset_index()

    def msem(k, m):
        v = k[k.model == m]["kappa"]
        return v.mean(), (v.sem() if len(v) > 1 else 0.0)

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.3), gridspec_kw=dict(wspace=0.62, hspace=0.62))
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.99, top=0.93)

    # a: glyph networks, plain and entangled inputs
    ax = axs[0, 0]
    r20 = pd.DataFrame(jsons("e20")); r20 = r20[(r20.arm == "base") & (r20.V == 8) & (r20.arch.isin(["mlp", "cnn", "slot"]))]
    for yy, (gain, lab) in zip((1, 0), ((0.0, "plain"), (4.0, "entangled"))):
        g = r20[r20.gain == gain]
        vals = [(g.read_input.mean(), GREY, 0.18), (g.read_untrained.mean(), LGREY, 0), (g.read_trained.mean(), REP, -0.18)]
        ax.plot([min(v[0] for v in vals), max(v[0] for v in vals)], [yy, yy], color=LGREY, lw=1.0, zorder=1)
        for v, c, dy in vals:
            ax.plot(v, yy + dy, "o", color=c, ms=4.6, zorder=3, mec=DGREY if c == LGREY else "white", mew=0.5)
        print("E20", lab, [round(v[0], 3) for v in vals])
    ax.set_yticks([1, 0]); ax.set_yticklabels(["plain", "entangled"], fontsize=6.6); ax.set_ylim(-0.6, 2.2)
    ax.set_xlim(0.2, 1.0); ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["", "0.5", "", "1"]); ax.set_xlabel("probe $R^2$, new glyphs")
    tag(ax, 0.22, 1.85, "input", GREY, va="center"); tag(ax, 0.42, 1.85, "untrained", DGREY, va="center"); tag(ax, 0.7, 1.85, "trained", REP, va="center")
    panel(ax, "a", "Glyphs: the input holds the parts")

    # b: language models against their input embeddings
    ax = axs[0, 1]
    r24 = jsons("e24"); rows = []
    for m in sorted({r["model"] for r in r24}):
        c = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in r24:
            if r["model"] == m:
                c[(r["layer"], r["code"])][r["system"]].append(r)
        best = max(c, key=lambda k: np.mean([(x["rec_val_role1"] + x["rec_val_role2"]) / 2 for x in c[k]["model"]]))
        rec = lambda s: float(np.mean([(x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2 for x in c[best][s]]))
        rc, rs = rec("ref_concat"), rec("ref_sum")
        rows.append((m.split("/")[-1], *[(rec(s) - rs) / (rc - rs) for s in ("input_avg", "input_concat", "model")]))
    NM = {"phi-4": "phi-4", "Qwen3-8B": "Qwen3-8B", "OLMo-2-1124-13B": "OLMo-2 13B", "OLMo-2-1124-7B": "OLMo-2 7B", "pythia-12b": "Pythia-12b",
          "pythia-1.4b": "Pythia-1.4b", "gpt2-large": "GPT-2 large"}
    rows = sorted(rows, key=lambda r: list(NM).index(r[0]) if r[0] in NM else 99)
    Y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=DGREY, lw=0.5)
    for yy, (m, ia, ic, st) in zip(Y, rows):
        ax.plot([ia, max(ic, st)], [yy, yy], color=LGREY, lw=1.0, zorder=1)
        ax.plot(ia, yy, "o", mfc="white", mec=GREY, ms=4.2, zorder=3); ax.plot(ic, yy, "o", color=GREY, ms=4.2, zorder=3); ax.plot(st, yy, "o", color=REP, ms=4.2, zorder=4)
    ax.set_yticks(Y); ax.set_yticklabels([NM.get(r[0], r[0]) for r in rows], fontsize=6.0); ax.tick_params(axis="y", pad=1)
    ax.set_xlim(-0.6, 1.0); ax.set_xticks([-0.5, 0, 0.5, 1]); ax.set_ylim(-0.6, len(rows) + 1.2); ax.set_xlabel(r"$\kappa$, new nouns")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=4, label="mean input"),
                       Line2D([], [], marker="o", ls="", color=GREY, ms=4, label="two inputs"),
                       Line2D([], [], marker="o", ls="", color=REP, ms=4, label="state")],
              loc="upper center", ncol=3, fontsize=5.6, handletextpad=0.05, columnspacing=0.5, borderaxespad=0)
    panel(ax, "b", "Language: the model builds them")
    print("E24", [(r[0], *np.round(r[1:], 2)) for r in rows])

    # c: images, pixels vs random vs trained quadrant states
    ax = axs[0, 2]
    p24 = jsons("e24b"); rec2 = lambda x: (x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2
    ENC = [("dinov2-s", "DINOv2-S"), ("dinov2-l", "DINOv2-L"), ("dinov2-g", "DINOv2-g"), ("clip-b32-img", "CLIP-B"), ("clip-h14-img", "CLIP-H"), ("siglip-so400m", "SigLIP")]
    gq = grid[grid.pool == "grid"].set_index("model")
    Y6 = np.arange(6)[::-1]; pix = {}
    for m, _ in ENC:
        lay, code = f"L{gq.loc[m, 'layer']}", gq.loc[m, "code"]
        rr = [r for r in p24 if r["model"] == m and r["layer"] == lay and r["code"] == code]
        if not rr:
            rr = [r for r in p24 if r["model"] == m]
        S = {s: np.mean([rec2(r) for r in rr if r["system"] == s]) for s in ("ref_sum", "ref_concat", "pixels")}
        pix[m] = (S["pixels"] - S["ref_sum"]) / (S["ref_concat"] - S["ref_sum"])
    ax.axvline(0, color=DGREY, lw=0.5)
    for yy, (m, lab) in zip(Y6, ENC):
        vals = (pix[m], gq.loc["rand-" + m, "kappa"], gq.loc[m, "kappa"])
        ax.plot([min(vals), max(vals)], [yy, yy], color=LGREY, lw=1.0, zorder=1)
        for v, c in zip(vals, (GREY, LGREY, REP)):
            ax.plot(v, yy, "o", color=c, ms=4.2, zorder=3, mec=DGREY if c == LGREY else "white", mew=0.5)
    ax.set_yticks(Y6); ax.set_yticklabels([e[1] for e in ENC], fontsize=6.0); ax.tick_params(axis="y", pad=1)
    ax.set_xlim(-1.0, 1.05); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["−1", "", "0", "", "1"]); ax.set_ylim(-0.6, 7.2); ax.set_xlabel(r"$\kappa$, new photographs")
    tag(ax, -0.98, 6.55, "pixels", GREY, va="center", fontsize=6.0); tag(ax, 0.06, 6.55, "random", DGREY, va="center", fontsize=6.0); tag(ax, 0.62, 6.55, "trained", REP, va="center", fontsize=6.0)
    panel(ax, "c", "Images: the encoder builds them")
    print("pixels", {k: round(v, 3) for k, v in pix.items()})

    # d: text encoders trained vs random
    ax = axs[1, 0]
    e4all = pd.DataFrame(jsons("e4w")); e4all = e4all[e4all.n_fit == 165]
    for k in ("rec_val", "rec_fresh"):
        e4all[k] = e4all[[f"{k}_role1", f"{k}_role2"]].mean(1)
    own = e4all[(e4all.codes_from == e4all.model) & (e4all.rel != "and")]

    def kappa_txt(model):
        m = own[own.model == model]
        layer, code = m[m.system == "model"].groupby(["layer", "code"])["rec_val"].mean().idxmax()
        m = m[(m.layer == layer) & (m.code == code)]
        t = m.groupby(["seed", "system"])["rec_fresh"].mean().unstack()
        k = (t["model"] - t["ref_sum"]) / (t["ref_concat"] - t["ref_sum"])
        return k.mean()
    TN = {"clip-b32": "CLIP-B", "sd15-clip-l": "CLIP-L", "clip-h14": "CLIP-H", "t5-small": "T5-s", "t5-base": "T5-b", "t5-large": "T5-l",
          "bert-base-uncased": "BERT-b", "bert-large-uncased": "BERT-l", "gpt2": "GPT2-s", "gpt2-medium": "GPT2-m", "gpt2-large": "GPT2-l"}
    Y11 = np.arange(11)[::-1]
    ax.axvline(0, color=DGREY, lw=0.5)
    for yy, m in zip(Y11, TN):
        a, b = kappa_txt(m), kappa_txt("rand-" + m)
        ax.plot([b, a], [yy, yy], color=LGREY, lw=1.0, zorder=1)
        ax.plot(b, yy, "o", color=LGREY, ms=4, zorder=3, mec=DGREY, mew=0.5); ax.plot(a, yy, "o", color=REP, ms=4, zorder=3, mec="white", mew=0.4)
    ax.set_yticks(Y11); ax.set_yticklabels(list(TN.values()), fontsize=5.6); ax.tick_params(axis="y", pad=1)
    ax.set_xlim(-0.3, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-0.6, 12.4); ax.set_xlabel(r"$\kappa$, new nouns")
    tag(ax, -0.28, 11.6, "random", DGREY, va="center", fontsize=6.0); tag(ax, 0.2, 11.6, "trained", REP, va="center", fontsize=6.0)
    panel(ax, "d", "Training raises it")

    e6 = kappa_seeds(short(load7("e6")), ASYM); e10 = kappa_seeds(short(load7("e10")), ASYM)
    e6 = pd.concat([e6, e10[e10.model.str.startswith("pythia-1.4b")]])
    ax = axs[1, 1]
    PY = [("pythia-160m", 0.16), ("pythia-410m", 0.41), ("pythia-1b", 1.0), ("pythia-1.4b", 1.4), ("pythia-2.8b", 2.8), ("pythia-6.9b", 6.9), ("pythia-12b", 12)]
    PY = [(m, n) for m, n in PY if m in set(e6.model)]
    mu, se = zip(*[msem(e6, m) for m, _ in PY]); band(ax, np.array([n for _, n in PY]), np.array(mu), np.array(se), REP, marker="o", ms=2.6)
    for m, n in (("Qwen3-8B", 8.2), ("OLMo-2-1124-7B", 7.3), ("OLMo-2-1124-13B", 13.7), ("phi-4", 14.7)):
        ax.plot(n, msem(e6, m)[0], "s", color=REP2, ms=3.2, zorder=4)
    tag(ax, 0.13, 0.66, "Pythia", REP, ha="left"); tag(ax, 18, 0.2, "other\nmodels", REP2, ha="right", va="top", linespacing=1.0)
    ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"]); ax.minorticks_off()
    ax.set_xlabel("parameters (billions)"); ax.set_ylabel(r"$\kappa$, new nouns"); ax.set_ylim(-0.3, 1.05); ax.set_yticks([0, 0.5, 1]); ref_lines(ax)
    panel(ax, "e", "Scale helps up to 7B")

    ax = axs[1, 2]
    S410 = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 143000)
    k410 = np.array([[e10[e10.model == f"{r}@step{s}"].kappa.mean() for s in S410] for r in ["pythia-410m"] + [f"pythia-410m-seed{i}" for i in range(1, 6)]])
    xs = np.array([max(s, 100) for s in S410])
    ax.fill_between(xs, k410.min(0), k410.max(0), color=REP2, alpha=0.2, lw=0); ax.plot(xs, k410.mean(0), color=REP2, lw=1.0)
    ST = [(f"pythia-1.4b@step{s}", s) for s in (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000)] + [("pythia-1.4b", 143000)]
    ST = [(m, s) for m, s in ST if m in set(e6.model)]
    mu, se = zip(*[msem(e6, m) for m, _ in ST]); band(ax, np.array([max(s, 100) for _, s in ST]), np.array(mu), np.array(se), REP, marker="o", ms=2.6)
    tag(ax, 3000, 0.77, "1.4b", REP, ha="left"); tag(ax, 6000, 0.17, "410m, six runs", REP2, ha="center")
    ax.set_xscale("log"); ax.set_xticks([100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["0", "$10^3$", "$10^4$", "$10^5$"]); ax.minorticks_off()
    ax.set_xlabel("training step"); ax.set_ylabel(r"$\kappa$, new nouns"); ax.set_ylim(-0.3, 1.05); ax.set_yticks([0, 0.5, 1]); ref_lines(ax)
    panel(ax, "f", "It peaks early in training")
    save(fig, "fig_origins")
# ================================================================== beat 3: both kinds, weak only, neither
E11 = {"phi-4": ("L30", "phrase"), "Qwen3-8B": ("L27", "carrier"), "OLMo-2-1124-13B": ("L30", "phrase"),
       "OLMo-2-1124-7B": ("L16", "carrier"), "gpt2-large": ("L18", "carrier"), "pythia-1.4b": ("L12", "carrier"),
       "pythia-12b": ("L18", "carrier")}
MEANING = ("phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B")


def prompts():
    e9 = pd.read_csv("results/agg_r8_e9.csv", index_col=0)
    e16 = short(pd.DataFrame(jsons("e16")))
    out = {}
    for (m, v), g in e16.groupby(["model", "variant"]):
        gg = g[(g.layer == E11[m][0]) & (g.code == E11[m][1])]
        out[m, v] = (gg.choice.mean(), gg.xfer_role_conv.mean())
    e11 = short(pd.DataFrame(jsons("e11"))); e11 = e11[(e11.system == "model") & (e11.state == "query")]
    for m, (l, c) in E11.items():
        out[m, "main"] = (e9.loc[m, "choice"], e11[(e11.model == m) & (e11.layer == l) & (e11.code == c)].xfer_role_conv.mean())
    return out


def case_panel(ax, which):
    import case_story
    case_story.draw(ax, which)


if mode in ("all", "both"):
    fig = plt.figure(figsize=(6.5, 3.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 1.0], hspace=0.5, wspace=0.5, bottom=0.1, top=0.95, left=0.08, right=0.99)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    case_panel(ax, "both")
    ax = fig.add_subplot(gs[1, 0])
    pr = prompts()
    for (m, v), (a, t) in pr.items():
        c = REP if m in MEANING else RED
        ax.plot(a, t, "o", color=c, ms=3.8 if v == "main" else 2.8, alpha=1 if v == "main" else 0.55, mec="white", mew=0.3)
    A = np.array([p[0] for p in pr.values()]); T = np.array([p[1] for p in pr.values()])
    r = np.corrcoef(A, T)[0, 1]
    tag(ax, 0.42, 1.0, f"$r$ = {r:.2f}", INK, ha="left", va="top"); tag(ax, 0.42, 0.88, "7 models, 4 prompts", DGREY, ha="left", va="top")
    tag(ax, 0.6, 0.02, "word order", RED, ha="left", va="bottom")
    ax.set_xlim(0.38, 1.03); ax.set_ylim(-0.04, 1.05); ax.set_xticks([0.5, 0.75, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("answer accuracy (weak)", color=BEH); ax.set_ylabel("role transfer (strong)", color=REP)
    panel(ax, "b", "Answers follow the state")
    print("r", round(r, 3))

    e21, e21b = jsons("e21"), jsons("e21b")
    MODELS = [("microsoft/phi-4", "phi-4", 30, "o"), ("Qwen/Qwen3-8B", "Qwen3-8B", 27, "s"), ("allenai/OLMo-2-1124-13B", "OLMo-2 13B", 30, "D")]
    ax = fig.add_subplot(gs[1, 1])
    for m, n, _, mk in MODELS:
        rows = [x for x in e21 if x["model"] == m and x["source"] == "reversed" and x["kind"] == "whole"]
        L = sorted({x["layer"] for x in rows})
        y = [np.mean([x["to_answer"] for x in rows if x["layer"] == l]) for l in L]
        ax.plot([0.25, 0.5, 0.75, 1], y, marker=mk, color=FIX, ms=3.2, lw=1.0, label=n)
    ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["1/4", "1/2", "3/4", "1"])
    ax.set_ylim(-0.03, 1.3); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(0.18, 1.07)
    ax.set_xlabel("patched layer, share of depth"); ax.set_ylabel("answers flipped")
    ax.legend(loc="upper left", fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0, labelspacing=0.2)
    panel(ax, "c", "Patching the state flips the answer")

    ax = fig.add_subplot(gs[1, 2])

    def sweep(prefix, sizes):
        return np.array([np.mean([np.mean([x["flip"] for x in e21b if x["model"] == m and x["layer"] == l and x["kind"] == f"{prefix}_{k}"]) for m, _, l, _ in MODELS]) for k in sizes])
    for pre, sizes, c in (("diff", (16, 64, 256), FIX), ("readout", (16, 64, 128), REP), ("pc", (16, 64, 256), GREY), ("random", (16, 64, 256), LGREY)):
        ax.plot(sizes, sweep(pre, sizes), "o-", color=c, ms=3, lw=1.0, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256]); ax.set_xticklabels(["16", "64", "256"]); ax.minorticks_off()
    ax.set_xlim(12, 1300); ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1])
    d, r_, p_, q = sweep("diff", (16, 64, 256)), sweep("readout", (16, 64, 128)), sweep("pc", (16, 64, 256)), sweep("random", (16, 64, 256))
    tag(ax, 300, d[2], "source minus\ntarget", FIX, va="center", linespacing=1.0); tag(ax, 148, r_[2] - 0.03, "probe", REP, va="center")
    tag(ax, 300, p_[2] - 0.15, "top\ncomponents", DGREY, va="center", linespacing=1.0); tag(ax, 300, q[2] + 0.03, "random", GREY, va="center")
    ax.set_xlabel("dimensions patched"); ax.set_ylabel("answers flipped")
    panel(ax, "d", "A subspace carries it")
    save(fig, "fig_both")

if mode in ("all", "weak"):
    fig = plt.figure(figsize=(6.5, 4.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.6, wspace=0.55, bottom=0.1, top=0.95, left=0.1, right=0.99)
    ax = fig.add_subplot(gs[0, 0])
    e9 = pd.read_csv("results/agg_r8_e9.csv", index_col=0)
    e21 = jsons("e21"); e21b = jsons("e21b")
    base = np.mean([x["to_answer"] for x in e21 if x["model"].endswith("OLMo-2-1124-7B") and x["source"] == "reversed" and x["kind"] == "whole" and x["picked"]])
    flips = {L: np.mean([x["flip"] for x in e21b if x["model"].endswith("OLMo-2-1124-7B") and x["layer"] == L and x["kind"] == "whole"]) for L in (24, 32)}
    q16 = [x for x in jsons("e19") if x["model"].endswith("OLMo-2-1124-7B") and x["state"] == "query"]
    lays = [q16] + [[x for x in jsons(d_) if x["model"].endswith("OLMo-2-1124-7B") and x["state"] == "query"] for d_ in ("e26L24", "e26L32")]
    best_late = max(np.mean([x[k] for x in q]) for q in lays for k in ("ridge_conv", "kernel_conv", "mlp_conv"))
    oracle = max(np.mean([x[k] for x in q]) for q in lays for k in ("kernel_oracle_conv", "mlp_oracle_conv"))
    BARS = [("answers", e9.loc["OLMo-2-1124-7B", "choice"], BEH), ("patch, layer 16", base, FIX2), ("patch, layer 32", max(flips.values()), FIX),
            ("ridge probe", np.mean([x["ridge_conv"] for x in q16]), REP), ("best probe, any layer", best_late, REP), ("tuned on test", oracle, GREY)]
    for i, (lab, v, c) in enumerate(BARS):
        ax.barh(-i, v, 0.62, color=c)
        ax.text(v + 0.02, -i, f"{v:.2f}", fontsize=5.8, va="center", color=DGREY)
    ax.set_yticks(-np.arange(len(BARS))); ax.set_yticklabels([b[0] for b in BARS], fontsize=6.0)
    ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("share of new pairs")
    panel(ax, "a", "OLMo-2 7B outruns its probes")
    print("OLMo-7B", [(b[0], round(float(b[1]), 3)) for b in BARS], flips)

    ax = fig.add_subplot(gs[0, 1:]); ax.axis("off")
    case_panel(ax, "neither")

    ax = fig.add_subplot(gs[1, 0:2])
    e19 = short(pd.DataFrame(jsons("e19"))); pr = prompts()
    M = [("phi-4", "phi-4"), ("Qwen3-8B", "Qwen3-8B"), ("OLMo-2-1124-13B", "OLMo-2 13B"), ("OLMo-2-1124-7B", "OLMo-2 7B")]
    x = np.arange(len(M))
    for j, (st, c, lab) in enumerate((("mean", RED, "sentence average"), ("query", REP, "question state"))):
        for off, (rd_, mk) in zip((-0.12, 0, 0.12), (("ridge", "o"), ("kernel", "s"), ("mlp", "D"))):
            v = [e19[(e19.model == m) & (e19.state == st)][f"{rd_}_conv"].mean() for m, _ in M]
            ax.plot(x + (j - 0.5) * 0.42 + off, v, mk, color=c, ms=3.4, ls="", mec="white", mew=0.3)
    for k, m in enumerate(("pythia-1.4b", "gpt2-large")):
        ax.plot(len(M) + k + 0.21, pr[m, "main"][1], "o", color=REP, ms=3.4, mec="white", mew=0.3)
    ax.axhline(0.5, color=LGREY, lw=0.8, ls=":")
    ax.set_xticks(list(x) + [len(M), len(M) + 1]); ax.set_xticklabels([l for _, l in M] + ["Pythia-1.4b", "GPT-2 large"], fontsize=6.2)
    ax.set_ylim(-0.04, 1.25); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("figure probe on converse")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=RED, ms=3.4, label="sentence average"), Line2D([], [], marker="o", ls="", color=REP, ms=3.4, label="question state"),
                       Line2D([], [], marker="o", ls="", color=DGREY, ms=3.2, label="ridge"), Line2D([], [], marker="s", ls="", color=DGREY, ms=3.2, label="kernel"),
                       Line2D([], [], marker="D", ls="", color=DGREY, ms=3.2, label="MLP")], loc="upper center", ncol=5, fontsize=6, handletextpad=0.1, columnspacing=0.8, borderaxespad=0)
    panel(ax, "c", "Sentence states follow word order in every model")

    ax = fig.add_subplot(gs[1, 2])
    txt = pd.DataFrame(jsons("e18")).groupby("model")[["roles", "identity"]].mean()
    TX = [("clip-b32", "CLIP-B"), ("sd15-clip-l", "CLIP-L"), ("clip-h14", "CLIP-H"), ("t5-small", "T5-s"), ("t5-base", "T5-b"),
          ("t5-large", "T5-l"), ("bert-base-uncased", "BERT-b"), ("bert-large-uncased", "BERT-l")]
    Y8 = np.arange(8)[::-1]
    ax.axvline(0.5, color=LGREY, lw=0.8, ls=":")
    for yy, (m, _) in zip(Y8, TX):
        ax.plot([txt.loc[m, "roles"], txt.loc[m, "identity"]], [yy, yy], color=LGREY, lw=1.0, zorder=1)
        ax.plot(txt.loc[m, "identity"], yy, "o", color=GREY, ms=3.8, zorder=3); ax.plot(txt.loc[m, "roles"], yy, "o", color=BEH, ms=3.8, zorder=3)
    ax.set_yticks(Y8); ax.set_yticklabels([l for _, l in TX], fontsize=6.0); ax.set_ylim(-0.6, 9.2); ax.set_xlim(-0.03, 1.05); ax.set_xticks([0, 0.5, 1])
    tag(ax, 0.0, 8.45, "roles", BEH, va="center", fontsize=6.0); tag(ax, 0.7, 8.45, "nouns", GREY, va="center", fontsize=6.0)
    ax.set_xlabel("paraphrase match accuracy")
    panel(ax, "d", "Text encoders lose roles")
    save(fig, "fig_weak")
# ================================================================== beat 1 and 3: the map
if mode in ("all", "map"):
    from matplotlib.patches import FancyBboxPatch
    import case_story
    fig = plt.figure(figsize=(6.5, 3.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    X0, X1, Y0, Y1, W, H = 0.115, 0.56, 0.47, 0.03, 0.435, 0.405
    ax.text(X0 + W / 2, 0.955, "own procedure succeeds", color=BEH, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(X1 + W / 2, 0.955, "own procedure fails", color=BEH, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.995, "weak composition: the system's own procedure on new parts", color=BEH, ha="center", va="top", fontsize=6.6)
    ax.text(0.035, Y0 + H / 2, "probe recovers\nthe parts", color=REP, ha="center", va="center", fontsize=8, fontweight="bold", rotation=90, linespacing=1.0)
    ax.text(0.035, Y1 + H / 2, "probe misses\nthe parts", color=REP, ha="center", va="center", fontsize=8, fontweight="bold", rotation=90, linespacing=1.0)
    ax.text(0.085, 0.5, "strong composition: a linear probe of the state", color=REP, ha="center", va="center", fontsize=6.6, rotation=90)
    CELLS = [(X0, Y0, "Both kinds", "phi-4"), (X1, Y0, "Strong without weak", "MLP trained on 8 glyphs"),
             (X0, Y1, "Weak without strong", "OLMo-2 7B"), (X1, Y1, "Neither", "Pythia-1.4b")]
    for x, y, t, sub in CELLS:
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.004,rounding_size=0.012", fc="#fafafa", ec=LGREY, lw=0.8))
        ax.text(x + 0.015, y + H - 0.04, t, fontsize=7.6, fontweight="bold", color=INK, va="center")
        ax.text(x + W - 0.015, y + H - 0.04, sub, fontsize=6.8, color=DGREY, va="center", ha="right")

    def cell_lm(x, y, model, sentence_pairs, foot):
        d = case_story.load(model)
        byS = {c["sentence"]: c for c in d["cases"]}
        ax.text(x + 0.015, y + H - 0.105, "Which object is on top?", fontsize=6.4, color=DGREY, va="center")
        ax.text(x + 0.245, y + H - 0.105, "answer", fontsize=6.4, color=BEH, va="center")
        ax.text(x + 0.325, y + H - 0.105, "question state", fontsize=6.4, color=REP, va="center")
        for i, snt in enumerate(sentence_pairs):
            c = byS[snt]; yy = y + H - 0.175 - i * 0.075
            ax.text(x + 0.015, yy, snt, fontsize=6.8, color=INK, va="center", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=LGREY, lw=0.6))
            for xx, key, col in ((x + 0.245, "answer", BEH), (x + 0.325, "readout_query", REP)):
                ret = c[key]["preferred"] if key == "answer" else c[key]["returned"]
                ok = ret == c["figure"]
                ax.text(xx, yy, ret, fontsize=7, color=col if ok else RED, fontweight="bold" if ok else "normal", va="center")
        ax.text(x + 0.015, y + 0.045, foot, fontsize=6.3, color=DGREY, va="center")

    cell_lm(X0, Y0, "phi-4", ["a cake on a plate", "a plate under a cake"], "new nouns: answers 0.99, role transfer 0.98")
    cell_lm(X0, Y1, "OLMo-2 7B", ["a cake on a plate", "a plate under a cake"], "answers 0.77, role transfer 0.20, patches flip 0.85")
    cell_lm(X1, Y1, "Pythia-1.4b", ["a cake on a plate", "a plate under a cake"], "answers 0.41–0.56, role transfer 0.07–0.10")
    # glyph cell
    tgt, net, rd = case["V8_target"], case["V8_net"], case["V8_readout"]
    var = ((tgt[:4] - tgt[:4].mean(0)) ** 2).sum((1, 2)).mean()
    for j, (im, lab, col) in enumerate(((tgt, "target", DGREY), (net, "own decoder", BEH), (rd, "probe", REP))):
        for k in range(2):
            ia = fig.add_axes([X1 + 0.02 + j * 0.138, Y0 + H - 0.2 - k * 0.1, 0.12, 0.085])
            ia.imshow(im[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto"); ia.set_xticks([]); ia.set_yticks([])
            for sp in ia.spines.values():
                sp.set_color(col if j else LGREY); sp.set_linewidth(1.1 if j else 0.6)
        ax.text(X1 + 0.08 + j * 0.138, Y0 + H - 0.1, lab, fontsize=6.4, color=col, ha="center", va="center")
    ax.text(X1 + 0.015, Y0 + 0.045, "new glyphs: own decoder 0.20–0.34, probe 0.83–0.86", fontsize=6.3, color=DGREY, va="center")
    save(fig, "fig_map")
print("done", mode)
