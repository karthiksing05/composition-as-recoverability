"""Appendix: composable diffusion (a) and place words (b), SD 1.5, guidance weight 7.5, from results/e32.
a  AND composition "a x + a y".  Swapping the two concepts gives the identical latent (results/e32/swapcheck_fp16.json:
   max_abs_latent_diff 0.0, identical pixels).  Rule: the first four 'and' entries of swapcheck_fp16.json (pairs 0-3 of the held-out
   150, one seed each as listed there); the image shown is that entry's own pair and seed.  Boxes: every noun detected with
   score >= 0.35 (e31_analyze threshold).
b  place words "a x on the left" + "a y on the right".  Rule: the pair "tiger"/"glass" named in the paper (seed 0) followed by
   the first three other pairs in index order (seed 0) that have both objects detected as separate boxes (score >= 0.35, IoU < 0.6).
   Mark: the figure x lands on the left (box centre left of the ground's)."""
from fig_case_common import *

sw = [r for r in jload(f"{R32}/swapcheck_fp16.json") if r["mode"] == "and"][:4]
and_idx = jload(f"{R32}/index/and_w7.5.json"); and_det = jl(f"{R32}/det_and_w7.5.jsonl")
pos_idx = jload(f"{R32}/index/pos_w7.5.json"); pos_det = jl(f"{R32}/det_pos_w7.5.jsonl")
m_pos = [r for r in jload(f"{R32}/metrics_pos_thr0.35.json")["rows"]][0]
m_and = [r for r in jload(f"{R32}/metrics_and_thr0.35.json")["rows"]][0]

def png(rec):
    return np.asarray(Image.open(f"{R32}/images/{rec['file']}.png").convert("RGB"))

W, S, G = 6.75, 1.5, 0.25
H = 0.25 + 0.18 + S + 0.62 + 0.15 + 0.18 + S + 0.5 + 0.22
P = Page(W, H)
P.ax.add_patch(Rectangle((0.02, 0.07), 0.2, 0.1, fill=False, ec=FIGC, lw=1.1)); P.text(0.27, 0.05, "first-named noun", fs=6.2)
P.ax.add_patch(Rectangle((1.6, 0.07), 0.2, 0.1, fill=False, ec=GRDC, lw=1.1)); P.text(1.85, 0.05, "second-named noun", fs=6.2)

# ---- a
y = 0.28
P.head(0.01, y, "a", "Composable diffusion: the two concepts swapped give the same image, to the last bit")
shown = []
for j, s in enumerate(sw):
    rec = next(r for r in and_idx if r["x"] == s["x"] and r["y"] == s["y"] and r["seed"] == s["seed"])
    x = 0.01 + j * (S + G)
    P.img(x, y + 0.2, png(rec), S, edge=LGREY)
    d = and_det[rec["file"]]
    for n, col in ((rec["y"], GRDC), (rec["x"], FIGC)):
        if d[n]["score"] >= THR:
            P.box(x, y + 0.2, S, d[n]["box"], col)
    P.text(x, y + 0.2 + S + 0.04, f"a {rec['x']} + a {rec['y']}", fs=6.2)
    P.text(x, y + 0.2 + S + 0.17, f"a {rec['y']} + a {rec['x']}", fs=6.2)
    P.text(x, y + 0.2 + S + 0.30, f"identical: latent diff {s['max_abs_latent_diff']:g}", fs=6.0, c=OKC)
    shown.append(("and", rec["file"], s["max_abs_latent_diff"]))

# ---- b
y2 = y + 0.2 + S + 0.62 + 0.15
P.head(0.01, y2, "b", "Place words: “a x on the left” + “a y on the right”")
bypair = {}
for r in pos_idx:
    if r["seed"] == 0:
        bypair[r["pair"]] = r
tiger = [r for p, r in sorted(bypair.items()) if (r["x"], r["y"]) == ("tiger", "glass")][0]
items = [(tiger, info(tiger, pos_det))]
for p, r in sorted(bypair.items()):
    i = info(r, pos_det)
    if i["both"] and r is not tiger:
        items.append((r, i))
    if len(items) == 4:
        break
for j, (r, i) in enumerate(items):
    x = 0.01 + j * (S + G)
    P.img(x, y2 + 0.2, png(r), S, edge=LGREY)
    d = pos_det[r["file"]]
    for n, col in ((r["y"], GRDC), (r["x"], FIGC)):
        if d[n]["score"] >= THR:
            P.box(x, y2 + 0.2, S, d[n]["box"], col)
    P.text(x, y2 + 0.2 + S + 0.04, f"a {r['x']} on the left", fs=6.2)
    P.text(x, y2 + 0.2 + S + 0.17, f"a {r['y']} on the right", fs=6.2)
    if i["both"]:
        mark(P, x + 0.01, y2 + 0.2 + S + 0.32, i["xfirst"])
        P.text(x + 0.13, y2 + 0.2 + S + 0.30, f"{r['x']} {'left' if i['xfirst'] else 'right'}", fs=6.2, c=OKC if i["xfirst"] else BADC, fontweight="bold")
    else:
        P.text(x, y2 + 0.2 + S + 0.30, "not both detected", fs=6.0, c=DGREY)
    shown.append(("pos", r["file"], i["both"], i["xfirst"]))
P.text(0.01, y2 + 0.2 + S + 0.55,
       f"Over all 150 pairs x 4 seeds, {m_pos['n_both']} of {m_pos['n']} place-word images hold both objects, and the figure is on the left in {m_pos['rel_acc']:.2f} of them "
       f"(metrics_pos_thr0.35.json).", fs=6.2, c=DGREY)
P.save("fig_case_compose")
print(shown)
