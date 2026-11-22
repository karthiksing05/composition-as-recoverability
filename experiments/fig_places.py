"""Body figure for the image systems and harnesses (sec:diffusion, sec:harness), read from the result files.
  a  quadrant-state kappa (E34 protocol, readout G2), fixed quadrants vs photographs moved along one row
  b  the system's own judgment: swapped-place caption vs wrong-object caption (E17 matching; E30 diffusion classifier,
     U-Nets at 512 px, transformer denoisers at their native 1,024 px), mean over seeds and relations
  c  E36 text harness: direct answer vs parse + lexicon, training relations and held-out relation phrases
  d  E37 layout harness: figure in its place, canonical vs converse captions, per planner (SD 1.5 = no planner)
usage: fig_places.py"""
import csv, glob, json, os
import numpy as np
from figstyle import *

R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results"
NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"

# a: E34 quadrant kappa
k34 = {(r["model"], r["cond"]): float(r["kappa"]) for r in csv.DictReader(open(f"{R}/e34/agg_kappa.csv", encoding="utf-8"))
       if r["readout"] == "G2"}
A = [("CLIP-B/32", "clip-b32-img"), ("DINOv2-L", "dinov2-l"), ("SigLIP", "siglip-so400m"),
     ("SD 1.5", "sd15"), ("PixArt-Σ", "pixart"), ("Lumina 2.0", "lumina"), ("random CLIP-B/32", "rand-clip-b32-img")]


# b: own judgment
def mean_of(pattern, key):
    v = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(pattern))]
    v = [x for d in v for x in (d if isinstance(d, list) else [d])]
    return float(np.mean([x[key] for x in v])), len(v)


B = [("CLIP-B/32", f"{R}/e17/clip-b32_s*.json"), ("CLIP-H/14", f"{R}/e17/clip-h14_s*.json"),
     ("SigLIP", f"{R}/e17/siglip-so400m_s*.json"), ("SD 1.5", f"{R}/e30/sd15_classifier_s*.json"),
     ("SDXL", f"{R}/e30/sdxl_classifier_s*.json"), ("PixArt-Σ", f"{R}/e30/pixart-1024_classifier_s*.json"),
     ("Lumina 2.0", f"{R}/e30/lumina-1024_classifier_s*.json")]
# c: E36
s36 = json.load(open(f"{R}/e36/summary.json", encoding="utf-8"))
C = [("Pythia-1.4B", "EleutherAI/pythia-1.4b"), ("GPT-2 L", "gpt2-large"), ("OLMo-2 7B", "allenai/OLMo-2-1124-7B"),
     ("Qwen3-8B", "Qwen/Qwen3-8B"), ("phi-4", "microsoft/phi-4")]
# d: E37
s37 = json.load(open(f"{R}/e37/summary.json", encoding="utf-8"))


def by_phr(d, phr):
    return float(np.mean([v["placement"] for k, v in d.items() if k.endswith("|" + phr)]))


D = [("SD 1.5, no plan", None), ("first noun", "position"), ("Pythia-1.4B", "EleutherAI/pythia-1.4b"),
     ("GPT-2 L", "gpt2-large"), ("OLMo-2 7B", "allenai/OLMo-2-1124-7B"), ("Qwen3-8B", "Qwen/Qwen3-8B"),
     ("phi-4", "microsoft/phi-4"), ("lexicon", "lexicon")]

fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0, 1.0], wspace=0.95))
FS = 6.4

# a
ax = axs[0]
for i, (lab, key) in enumerate(A):
    f, m = k34[(key, "fixed")], k34[(key, "rand")]
    c = CTRL if lab.startswith("random") else REP
    ax.plot([m, f], [i, i], color=LGREY, lw=1.6, zorder=1)
    ax.plot(f, i, "o", color=c, ms=4.2, zorder=3)
    ax.plot(m, i, "o", mfc="white", mec=c, mew=1.0, ms=4.2, zorder=3)
    print("a", lab, round(f, 2), round(m, 2))
ax.axvline(0, color=GREY, lw=0.6, ls="--")
ax.set_yticks(range(len(A))); ax.set_yticklabels([a[0] for a in A], fontsize=FS); ax.invert_yaxis()
ax.set_xlim(-0.8, 1.02); ax.set_xticks([-0.5, 0, 0.5, 1]); ax.set_xlabel("quadrant-state κ")
ax.plot([], [], "o", color=REP, ms=4, label="fixed places"); ax.plot([], [], "o", mfc="white", mec=REP, ms=4, label="moved")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.8, handletextpad=0.2, columnspacing=0.8)
ax.axhline(2.5, color=LGREY, lw=0.5)
panel(ax, "a", "Places in the state")

# b
ax = axs[1]
for i, (lab, pat) in enumerate(B):
    ro, n = mean_of(pat, "roles"); idn, _ = mean_of(pat, "identity")
    ax.plot([ro, idn], [i, i], color=LGREY, lw=1.6, zorder=1)
    ax.plot(idn, i, "s", color=CTRL, ms=3.8, zorder=3)
    ax.plot(ro, i, "o", color=BEH, ms=4.2, zorder=3)
    print("b", lab, round(ro, 3), round(idn, 3), n)
ax.axvline(0.5, color=GREY, lw=0.6, ls="--")
ax.set_yticks(range(len(B))); ax.set_yticklabels([b[0] for b in B], fontsize=FS); ax.invert_yaxis()
ax.set_xlim(0.4, 1.0); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel("own choice correct")
ax.plot([], [], "o", color=BEH, ms=4, label="swapped"); ax.plot([], [], "s", color=CTRL, ms=3.6, label="wrong object")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=5.8, handletextpad=0.2)
ax.axhline(2.5, color=LGREY, lw=0.5)
panel(ax, "b", "Own judgment")

# c
ax = axs[2]
for i, (lab, key) in enumerate(C):
    d, p, h = s36[f"{key}|train|all"]["direct"], s36[f"{key}|train|all"]["parse"], s36[f"{key}|new|all"]["parse"]
    ax.plot([d, p], [i, i], color=LGREY, lw=1.6, zorder=1)
    ax.plot(d, i, "o", color=BEH, ms=4.2, zorder=3)
    ax.plot(p, i, "o", color=FIX, ms=4.2, zorder=4)
    ax.plot(h, i + 0.28, "o", mfc="white", mec=FIX, mew=1.0, ms=3.4, zorder=3)
    print("c", lab, round(d, 3), round(p, 3), round(h, 3))
ax.axvline(0.5, color=GREY, lw=0.6, ls="--")
ax.set_yticks(range(len(C))); ax.set_yticklabels([c[0] for c in C], fontsize=FS); ax.invert_yaxis()
ax.set_xlim(0.3, 1.04); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel("role answer correct")
ax.plot([], [], "o", color=BEH, ms=4, label="direct"); ax.plot([], [], "o", color=FIX, ms=4, label="parse")
ax.plot([], [], "o", mfc="white", mec=FIX, ms=3.4, label="held-out")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=5.8, handletextpad=0.2)
panel(ax, "c", "Text harness")

# d
ax = axs[3]
for i, (lab, key) in enumerate(D):
    if key is None:
        d = s37["sd15_unplanned"]; ca, co = by_phr({k: v for k, v in d.items() if k != "ALL"}, "canon"), by_phr({k: v for k, v in d.items() if k != "ALL"}, "conv")
        c = BEH
    else:
        d = {k: v for k, v in s37["planners"][key].items() if k != "ALL"}; ca, co = by_phr(d, "canon"), by_phr(d, "conv")
        c = FIX
    ax.plot([ca, co], [i - 0.14, i + 0.14], color=LGREY, lw=1.6, zorder=1)
    ax.plot(ca, i - 0.14, "o", color=c, ms=4.2, zorder=3)
    ax.plot(co, i + 0.14, "o", mfc="white", mec=c, mew=1.0, ms=4.2, zorder=3)
    print("d", lab, round(ca, 3), round(co, 3))
ax.axvline(0.5, color=GREY, lw=0.6, ls="--")
ax.set_yticks(range(len(D))); ax.set_yticklabels([x[0] for x in D], fontsize=FS); ax.invert_yaxis()
ax.set_xlim(0, 1.02); ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel("figure in its place")
ax.plot([], [], "o", color=GREY, ms=4, label="canonical"); ax.plot([], [], "o", mfc="white", mec=GREY, ms=4, label="converse")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.8, handletextpad=0.2, columnspacing=0.8)
panel(ax, "d", "Layout harness")

os.makedirs(NOTES, exist_ok=True)
fig.savefig(f"{FIGDIR}/fig_places.pdf"); fig.savefig(f"{NOTES}/fig_places.png", dpi=170)
print("saved")
