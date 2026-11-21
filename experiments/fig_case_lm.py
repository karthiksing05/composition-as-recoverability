"""Appendix case table: four sentence pairs x five language models (results/case_lm/*.json, E-case_lm.py).

Selection rule: the first four distinct (figure, ground) pairs in file order, each shown as canonical then converse
(book/bench, cake/plate, lamp/rug, leaf/rock). The files hold one relation ("on"/"under") and one question only, so no
second relation is available. Each cell: the model's answer (orange if it is the figure, red if the ground), the noun a
figure probe of the question state returns (blue / red), and the noun a probe of the token-averaged sentence state
returns (grey / red). Bold = figure noun, red = ground noun. Footer numbers are the saved 500-sentence aggregates."""
import json, os
from figstyle import *

ROOT = "/coc/flash10/ksingara3/compositionality-theory"
FILES = [("phi-4", "microsoft_phi-4"), ("Qwen3-8B", "Qwen_Qwen3-8B"), ("OLMo-2 7B", "allenai_OLMo-2-1124-7B"),
         ("Pythia-1.4b", "EleutherAI_pythia-1.4b"), ("GPT-2 large", "gpt2-large")]
D = {n: json.load(open(f"{ROOT}/experiments/results/case_lm/{f}.json", encoding="utf-8")) for n, f in FILES}
pairs = []
for c in D["phi-4"]["cases"]:
    if (c["figure"], c["ground"]) not in pairs:
        pairs.append((c["figure"], c["ground"]))
pairs = pairs[:4]

W, X0, SW = 6.75, 1.28, 1.09          # width, first model column, model column width
ROW, GAP, TOP = 0.185, 0.07, 0.62
H = TOP + 4 * (2 * ROW + GAP) + 0.72
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
rend = fig.canvas.get_renderer()


def segs(x, y, parts, fs=6.8):
    """draw text pieces left to right; parts = [(string, kwargs)]"""
    for s, kw in parts:
        t = ax.text(x, y, s.strip(), fontsize=fs, va="center", ha="left", **kw)
        x += t.get_window_extent(rend).width / fig.dpi + 0.05
    return x


ax.text(0.04, 0.17, "a", fontsize=9, fontweight="bold", va="center")
ax.text(0.17, 0.17, "Question asked of every sentence:  ", fontsize=7.5, va="center")
ax.text(2.18, 0.17, "Which object is on top?", fontsize=7.5, va="center", fontweight="bold")
ax.text(6.71, 0.17, "bold: figure noun    red: ground noun", fontsize=6.4, va="center", ha="right", color=DGREY)
SUB = [("answer", BEH, 0.0), ("question\nstate", REP, 0.37), ("sentence\naverage", DGREY, 0.73)]
for j, (n, _) in enumerate(FILES):
    x = X0 + j * SW
    ax.text(x, 0.33, n, fontsize=7.4, fontweight="bold", va="center")
    for s, c, dx in SUB:
        ax.text(x + dx, 0.51, s, fontsize=5.8, color=c, va="center", linespacing=0.95)
ax.plot([0.04, 6.71], [0.62 - 0.015, 0.62 - 0.015], color=LGREY, lw=0.5)

shown = []
y = TOP + 0.09
for fg, gd in pairs:
    for ph in ("canonical", "converse"):
        c0 = [c for c in D["phi-4"]["cases"] if (c["figure"], c["ground"], c["phrasing"]) == (fg, gd, ph)][0]
        toks = c0["sentence"].split()
        # sentence with figure bold and ground red
        parts = []
        for w in toks:
            kw = dict(fontweight="bold", color=INK) if w == fg else dict(color=LOST) if w == gd else dict(color=DGREY)
            parts.append((w + " ", kw))
        segs(0.08, y, parts)
        ax.text(1.2, y, "", fontsize=5)
        for j, (n, _) in enumerate(FILES):
            c = [c for c in D[n]["cases"] if (c["figure"], c["ground"], c["phrasing"]) == (fg, gd, ph)][0]
            x = X0 + j * SW
            for (key, col, dx) in (("answer", BEH, 0.0), ("readout_query", REP, 0.37), ("readout_mean", DGREY, 0.73)):
                w = c["answer"]["preferred"] if key == "answer" else c[key]["returned"]
                ok = w == fg
                ax.text(x + dx, y, w, fontsize=6.8, va="center", color=col if ok else LOST, fontweight="bold" if ok else "normal")
                shown.append((n, c["sentence"], key, w, ok))
        y += ROW
    y += GAP
    ax.plot([0.04, 6.71], [y - GAP / 2 - 0.005] * 2, color="#ececec", lw=0.5)
# footer: saved aggregates, share of 500 held-out sentences per phrasing for which each readout returns the figure
y += 0.03
ax.text(0.08, y, "share returning figure", fontsize=5.8, color=DGREY, va="center")
ax.text(0.08, y + 0.13, "(500 sentences each)", fontsize=5.8, color=DGREY, va="center")
for r, ph in enumerate(("can", "conv")):
    yy = y + 0.065 + 0.13 * r
    ax.text(X0 - 0.1, yy, "canon." if ph == "can" else "conv.", fontsize=5.4, color=DGREY, va="center", ha="right")
    for j, (n, _) in enumerate(FILES):
        a = D[n]["aggregates"]; x = X0 + j * SW
        for (k, col, dx) in (("ans_qprompt_", BEH, 0.0), ("xfer_query_", REP, 0.37), ("xfer_mean_", DGREY, 0.73)):
            ax.text(x + dx, yy, f"{a[k + ph]:.2f}", fontsize=6.4, color=col, va="center", fontweight="bold")
y += 0.13
H_used = y + 0.17
fig.set_size_inches(W, H_used)
ax.set_ylim(H_used, 0)
ax.set_position([0, 0, 1, 1])
os.makedirs(f"{ROOT}/notes/audit/figs", exist_ok=True)
fig.savefig(f"{FIGDIR}/fig_case_lm.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(f"{ROOT}/notes/audit/figs/fig_case_lm.png", dpi=170, bbox_inches=None, pad_inches=0)
print("pairs", pairs)
for n, _ in FILES: print(n, {k: D[n]["aggregates"][k] for k in ("ans_qprompt_can", "ans_qprompt_conv", "xfer_query_can", "xfer_query_conv", "xfer_mean_can", "xfer_mean_conv")}, D[n]["layer"], D[n]["code"])
