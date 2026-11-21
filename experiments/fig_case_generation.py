"""Appendix gallery: SD 1.5 (rows a-c, three noun pairs each for left of / above / on) and SDXL (row d, one pair per relation),
canonical prompt next to its converse under the SAME seed, with the detector boxes (figure noun blue, ground noun grey) and a mark
whether the figure landed in its place.
Selection rule (no hand picking): seed 0; for each relation the first pairs in index order for which both phrasings have both
objects detected as separate boxes (score >= 0.35, IoU < 0.6, the paper's detector setting). SDXL: first such pair per relation."""
from fig_case_common import *

W = 6.75
S, G1, G2 = 1.04, 0.04, 0.16
ROWH = 0.18 + S + 0.40 + 0.10
rows = [("a", "SD 1.5", "sd15", "left", 3), ("b", "SD 1.5", "sd15", "above", 3), ("c", "SD 1.5", "sd15", "on", 3)]
RELN = {"left": "left of", "above": "above", "on": "on"}
H = 0.30 + 4 * ROWH
P = Page(W, H)
# legend
P.ax.add_patch(Rectangle((0.02, 0.07), 0.2, 0.1, fill=False, ec=FIGC, lw=1.1)); P.text(0.27, 0.05, "figure noun (first noun of the canonical prompt)", fs=6.2)
P.ax.add_patch(Rectangle((2.55, 0.07), 0.2, 0.1, fill=False, ec=GRDC, lw=1.1)); P.text(2.80, 0.05, "ground noun", fs=6.2)
P.text(3.65, 0.05, "mark: figure box centre left of / above the ground box", fs=6.2, c=DGREY)

def xpos(j):  # j = 0..5, pairs of two
    pair, k = divmod(j, 2)
    return 0.01 + pair * (2 * S + G1 + G2) + k * (S + G1)

def draw(model, items, y0, tag):
    for pi, (c, ic, v, iv) in enumerate(items):
        for k, (r, i) in enumerate(((c, ic), (v, iv))):
            x = xpos(2 * pi + k)
            P.img(x, y0, load_png(model, r), S, edge=LGREY)
            P.box(x, y0, S, i["by"], GRDC); P.box(x, y0, S, i["bx"], FIGC)
            t, col = verdict(r, i)
            mark(P, x + 0.01, y0 + S + 0.05, i["xfirst"]); P.text(x + 0.13, y0 + S + 0.03, t, fs=6.2, c=col, fontweight="bold")
            P.text(x, y0 + S + 0.16, wrap(r["caption"]), fs=6.0, c=DGREY, linespacing=1.15)

used = []
y = 0.30
for letter, name, model, rel, k in rows:
    items = pick_pairs(model, rel, k)
    P.head(0.01, y, letter, f"{name}, “left of” / “right of”" if rel == "left" else (f"{name}, “above” / “below”" if rel == "above" else f"{name}, “on” / “under”"))
    draw(model, items, y + 0.18, rel)
    used += [(model, rel, c["pair"], c["seed"], c["x"], c["y"]) for c, _, _, _ in items]
    y += ROWH
# SDXL: one pair per relation
items = []
for rel in ("left", "above", "on"):
    items += pick_pairs("sdxl", rel, 1)
P.head(0.01, y, "d", "SDXL, one pair for each of “left of”, “above”, “on”")
draw("sdxl", items, y + 0.18, "sdxl")
used += [("sdxl", c["rel"], c["pair"], c["seed"], c["x"], c["y"]) for c, _, _, _ in items]
P.save("fig_case_generation")
print(used)
