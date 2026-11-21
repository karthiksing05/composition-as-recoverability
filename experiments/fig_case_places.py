"""Appendix: image-encoder cases.
a  Fixed quadrants (E8 BESIDE canvas, 224 px, photograph A left, B right).  Rule: the first four photograph pairs (named classes) of
   results/case/images.json in file order whose two classes have a name (no selection on correctness); the thumbnails are the photographs that the quadrant-state
   probe of SigLIP so400m (trained) and of its random-weight copy returns for each role (nearest of 300 new photographs, saved in
   images.json).  Green frame = the true photograph, red = another one.  Captions are built from the Tiny-ImageNet class names with
   E17's template ("a photo of a A on the left and a B on the right"); the swapped caption names the places the other way round.
   The caption scores printed are E17's aggregate over pairs (results/e17), since the per-canvas scores were not saved.
b  Moved photographs (E34 construction, seed 0): two 64x64 photographs on one random row of a 448x448 grey canvas, left one = figure.
   Rule: the first three pairs with named classes of the fresh split in E34 draw order, placements from E34's generator (default_rng(7000+seed)).
   kappa values: results/e34/agg_kappa.csv, quadrant-average readout 'G2'.
No model is run; the canvases are rebuilt from the saved pair ids and the Tiny-ImageNet parquet."""
import csv
from fig_case_common import *

V, ims, labels, photo = photo_loader()
try:
    names = class_names()
except Exception as e:  # timm missing
    print("no class names:", e); names = [None] * 200
I = jload(f"{RES}/case/images.json")
nm = lambda i: names[int(labels[i])] or f"class {int(labels[i])}"

# ---- E17 aggregate (SigLIP, beside and above, as in the paper): correct caption beats role-swapped caption
agg = []
for s in range(3):
    for r in jload(f"{RES}/e17/siglip-so400m_s{s}.json"):
        if r["rel"] in ("beside", "above"):
            agg.append(r)
roles = float(np.mean([r["roles"] for r in agg])); ident = float(np.mean([r["identity"] for r in agg]))

# ---- E34 kappa (quadrant averages, G2)
kap = {}
with open(f"{RES}/e34/agg_kappa.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["readout"] == "G2":
            kap[(r["model"], r["cond"])] = float(r["kappa"])

W = 6.75
RH, TH = 0.70, 0.56
H = 5.9
P = Page(W, H)
y = 0.0
P.head(0.01, y, "a", "Fixed quadrants: the matching cannot tell the places, the quadrant probe returns both photographs")
cx = {"canvas": 0.02, "cap": 1.50, "sig": 4.12, "rnd": 5.50}
yh = y + 0.24
P.text(cx["canvas"], yh, "canvas", fs=6.2, c=DGREY)
P.text(cx["cap"], yh, "captions", fs=6.2, c=DGREY)
P.text(cx["sig"], yh, "SigLIP probe: left | right", fs=6.2, c=REP, fontweight="bold")
P.text(cx["rnd"], yh, "random weights", fs=6.2, c=DGREY, fontweight="bold")
shown = []
named = lambda i: names[int(labels[i])] is not None
sel = [(k, p) for k, p in enumerate(I["pairs"]) if named(p[0]) and named(p[1])][:4]  # rule: first four pairs whose two classes have a name (E17 skips the others)
for r, (k_, (a, b)) in enumerate(sel):
    y0 = y + 0.42 + r * RH
    comp = np.asarray(V.compose("beside", photo(a), photo(b)), np.uint8)[56:168]
    P.img(cx["canvas"], y0, comp, 1.30, h=0.65, edge=LGREY)
    na, nb = nm(a), nm(b)
    P.text(cx["cap"], y0 + 0.03, "correct", fs=6.0, c=FIX, fontweight="bold")
    P.text(cx["cap"] + 0.50, y0 + 0.03, "\n".join(textwrap.wrap(f"a photo of a {na} on the left and a {nb} on the right", 42)), fs=6.0, linespacing=1.2)
    P.text(cx["cap"], y0 + 0.36, "swapped", fs=6.0, c=LOST, fontweight="bold")
    P.text(cx["cap"] + 0.50, y0 + 0.36, "\n".join(textwrap.wrap(f"a photo of a {nb} on the left and a {na} on the right", 42)), fs=6.0, linespacing=1.2)
    for key, m in (("sig", "siglip-so400m"), ("rnd", "rand-siglip-so400m")):
        for role, truth in ((0, a), (1, b)):
            p = I["models"][m][f"pred_role{role + 1}"][k_]
            P.img(cx[key] + role * (TH + 0.06), y0 + 0.045, np.asarray(photo(p).resize((112, 112))), TH, edge=GREEN if p == truth else RED, lw=1.3)
    shown.append((a, b, na, nb, [I["models"][m][f"pred_role{k}"][k_] for m in ("siglip-so400m", "rand-siglip-so400m") for k in (1, 2)]))
yf = y + 0.42 + 4 * RH + 0.02
m1, m2, r1, r2 = (I["models"]["siglip-so400m"]["id_role1"], I["models"]["siglip-so400m"]["id_role2"], I["models"]["rand-siglip-so400m"]["id_role1"], I["models"]["rand-siglip-so400m"]["id_role2"])
P.text(cx["canvas"], yf, f"Green frame = the true photograph among 300 new ones, red = another.\nOver 300 canvases the probe returns the right photograph for {np.mean([m1, m2]):.0%} (SigLIP) and {np.mean([r1, r2]):.0%} (random weights).\nCaptions (E17, SigLIP, 3 seeds): the correct caption beats the place-swapped one for {roles:.2f} of pairs, and one with a wrong object for {ident:.2f}.\nCanvases are shown without their grey top and bottom bands.", fs=6.0, c=DGREY, linespacing=1.3)

# ---- b
y2 = yf + 0.68
P.head(0.01, y2, "b", "Moved photographs: two 64 px photographs on one random row of a 448 px canvas, the left one is the figure")
# rebuild E34 draws (seed 0)
seed = 0
rng = np.random.default_rng(seed)
cls = rng.permutation(200); splits, s0 = {}, 0
for k, n, per in zip(("fit", "val", "fresh"), V.N_CLS, V.PER_CLS):
    ids = []
    for c in cls[s0:s0 + n]:
        ids += list(rng.choice(np.where(labels == c)[0], per, replace=False))
    splits[k] = [int(i) for i in ids]; s0 += n
def sample(ns, k):
    prs = [(x, y_) for x in ns for y_ in ns if x != y_]
    return [prs[i] for i in rng.choice(len(prs), min(int(k), len(prs)), replace=False)]
pairs = {"fit": sample(splits["fit"], 3000), "val": sample(splits["val"], 600), "fresh": sample(splits["fresh"], 1000)}
allp = pairs["fit"] + pairs["val"] + pairs["fresh"]
CANVAS, PH = 448, 64
r2g = np.random.default_rng(7000 + seed)
def place():
    while True:
        u, v = (int(r2g.integers(0, CANVAS - PH + 1)) for _ in range(2))
        if abs(u - v) >= PH:
            break
    yy = int(r2g.integers(0, CANVAS - PH + 1))
    return min(u, v), max(u, v), yy
pl = [place() for _ in allp]
off = len(pairs["fit"]) + len(pairs["val"])
S = 1.5
fsel = [j for j, p in enumerate(pairs["fresh"]) if named(p[0]) and named(p[1])][:3]  # rule: first three named fresh pairs
for j, jj in enumerate(fsel):
    (a, b), (xa, xb, yy) = pairs["fresh"][jj], pl[off + jj]
    c = Image.new("RGB", (CANVAS, CANVAS), V.BG)
    c.paste(photo(a).resize((PH, PH), Image.BICUBIC), (xa, yy)); c.paste(photo(b).resize((PH, PH), Image.BICUBIC), (xb, yy))
    x = 0.02 + j * (S + 0.12)
    P.img(x, y2 + 0.22, np.asarray(c), S, edge=LGREY)
    f = S / CANVAS
    P.box(x, y2 + 0.22, S, ((xa + PH / 2) / CANVAS, (yy + PH / 2) / CANVAS, PH / CANVAS, PH / CANVAS), FIGC, lw=1.0)
    P.box(x, y2 + 0.22, S, ((xb + PH / 2) / CANVAS, (yy + PH / 2) / CANVAS, PH / CANVAS, PH / CANVAS), GRDC, lw=1.0)
    P.text(x, y2 + 0.22 + S + 0.04, f"figure: {nm(a)}  |  ground: {nm(b)}", fs=6.0)
    shown.append(("moved", a, b, nm(a), nm(b), xa, xb, yy))
xt = 0.02 + 3 * (S + 0.12) + 0.05
P.ax.add_patch(Rectangle((xt, y2 + 0.26), 0.2, 0.1, fill=False, ec=FIGC, lw=1.0)); P.text(xt + 0.26, y2 + 0.24, "figure (left)", fs=6.0)
P.ax.add_patch(Rectangle((xt, y2 + 0.44), 0.2, 0.1, fill=False, ec=GRDC, lw=1.0)); P.text(xt + 0.26, y2 + 0.42, "ground (right)", fs=6.0)
P.text(xt, y2 + 0.68, "quadrant-state probe, κ (3 seeds):", fs=6.0, c=DGREY)
rows = [("CLIP-B/32", "clip-b32-img"), ("SigLIP", "siglip-so400m"), ("random CLIP-B/32", "rand-clip-b32-img")]
for k, (lab, m) in enumerate(rows):
    P.text(xt, y2 + 0.84 + k * 0.17, lab, fs=6.0)
    P.text(xt + 1.05, y2 + 0.84 + k * 0.17, f"fixed {kap[(m, 'fixed')]:+.2f}", fs=6.0, c=DGREY)
    P.text(xt + 1.65, y2 + 0.84 + k * 0.17, f"moved {kap[(m, 'rand')]:+.2f}", fs=6.0, c=REP if k < 2 else LOST, fontweight="bold")
P.save("fig_case_places")
print(shown)
print("kappa", {k: round(v, 3) for k, v in kap.items() if k[0] in dict(rows).values()})
