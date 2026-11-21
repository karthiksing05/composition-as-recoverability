"""Results overview: behavior (own procedure) against the state (declared readout), both on held-out parts.

Every score is scaled so that 0 is a blend, chance or word order and 1 is exact recovery:
  R^2 for glyph and tree outputs and readouts; kappa for image quadrant states; transfer of the figure readout for
  language-model question states; answer and caption-matching accuracy a rescaled to 2a - 1.
Values are the summary numbers reported in main.tex (sources in comments); ranges give the spread over models,
architectures or relations, and the marker sits at the midpoint. Green arrows: procedures that read the state."""
from figstyle import *
from matplotlib.patches import FancyArrowPatch, Polygon

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
acc = lambda a: 2 * a - 1
# name, (state lo, hi), (behavior lo, hi), label offset (dx, dy), ha
PTS = [
    ("phi-4", (0.98, 0.98), (acc(0.99), acc(0.99)), (0.035, 0.0), "left"),         # sec:behavior: answers .99, transfer .98
    ("Qwen3-8B", (0.87, 0.87), (acc(0.99), acc(0.99)), (-0.035, 0.0), "right"),       # answers .99, transfer .87
    ("OLMo-2 13B", (0.73, 0.73), (acc(0.92), acc(0.92)), (-0.035, 0.0), "right"),      # answers .92, transfer .73
    ("OLMo-2 7B", (0.20, 0.20), (acc(0.77), acc(0.77)), (0.03, 0.04), "left"),          # answers .77, transfer .20
    ("Pythia, GPT-2", (0.07, 0.10), (acc(0.41), acc(0.56)), (0.04, -0.07), "left"),   # sec:neither
    ("glyphs, V = 64", (0.89, 0.89), (0.85, 0.91), (0.025, -0.08), "left"),  # sec:states
    ("glyphs, V = 8", (0.83, 0.86), (0.20, 0.34), (-0.035, 0.0), "right"),
    ("tree, depth 3", (0.99, 0.99), (0.26, 0.26), (0.0, -0.1), "right"),
    ("image encoders, places", (0.81, 0.94), (acc(0.50), acc(0.51)), (0.0, -0.075), "center"),  # swapped-caption .50-.51, kappa .81-.94
    ("diffusion models, places", (0.41, 0.71), (acc(0.49), acc(0.50)), (0.0, 0.065), "center"),  # e30: classifier roles .49-.50, grid kappa SDXL .41, SD1.5 .71
]
REPAIRS = [  # (from point name, to behavior value, label)  sec:success
    ("glyphs, V = 8", 0.75, "new decoder"),
]

fig, ax = plt.subplots(figsize=(4.3, 3.6))
lo, hi = -0.25, 1.1
ax.add_patch(Polygon([[lo, lo], [hi, hi], [lo, hi]], closed=True, fc="#fbeeee", ec="none", zorder=0))
ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=1)
ax.text(-0.2, 1.0, "behavior exceeds\nthe linear probe", fontsize=6.4, color=LOST, va="top", linespacing=1.1)
ax.text(0.98, -0.2, "behavior lags the state", fontsize=6.4, color=DGREY, ha="right", va="bottom")
pos = {}
for name, (s0, s1), (b0, b1), (dx, dy), ha in PTS:
    s, b = (s0 + s1) / 2, (b0 + b1) / 2
    pos[name] = (s, b)
    if s1 > s0:
        ax.plot([s0, s1], [b, b], color=INK, lw=0.8, zorder=2)
    if b1 > b0:
        ax.plot([s, s], [b0, b1], color=INK, lw=0.8, zorder=2)
    ax.plot(s, b, "o", ms=4.5, color=INK, zorder=3)
    ax.text(s + dx, b + dy, name, fontsize=6.3, color=INK, ha=ha, va="center")
for name, to, lab in REPAIRS:
    s, b = pos[name]
    ax.add_patch(FancyArrowPatch((s, b + 0.04), (s, to - 0.01), arrowstyle="-|>", mutation_scale=8, color=FIX, lw=1.2, zorder=2))
    ax.text(s + 0.025, (b + to) / 2 + 0.02, lab, fontsize=6.2, color=FIX, ha="left", va="center", rotation=90)
s, b = pos["OLMo-2 7B"]
ax.add_patch(FancyArrowPatch((s + 0.01, b - 0.035), (0.50, b - 0.035), arrowstyle="-|>", mutation_scale=7, color=REP, lw=0.9, ls="-", zorder=2))
ax.text(0.25, b - 0.11, "best nonlinear\nprobe", fontsize=6.0, color=REP, ha="center", va="center")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("state: linear probe on parts new to the probe", color=REP)
ax.set_ylabel("behavior: own output on new parts", color=BEH)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_aspect("equal")
fig.savefig(f"{FIGDIR}/fig_ladder.pdf"); fig.savefig(f"{NOTES}/fig_ladder.png", dpi=170)
print("saved")
