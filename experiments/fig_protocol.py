"""Experimental design, part 1: the protocol as a flow chart, drawn for the glyph systems we train."""
import numpy as np
from diagram import *
from e2_glyph import World

F = Flow(6.5, 2.9)
G = World(0).glyph(np.random.default_rng(4).normal(size=(40, 8)))
pair = lambda a, b: np.concatenate([a, b], 1)

# ---------------- a: the pipeline ----------------
F.head(0.03, 2.8, "a", "Measure recovery on fresh parts")
y0, h = 1.66, 0.98
ws = [1.27, 0.86, 0.86, 1.2, 1.79]
gap = 0.13
xs = np.concatenate([[0.03], 0.03 + np.cumsum(np.array(ws[:-1]) + gap)])
titles = ["Split the parts", "Compose", "Encode", "Read back", "Calibrate"]
for k, (x, w, t) in enumerate(zip(xs, ws, titles)):
    F.card(x, y0, w, h, GREY, fill=True, lw=0.6, ec=LGREY)
    F.step(x + 0.1, y0 + h - 0.1, k + 1)
    F.text(x + 0.19, y0 + h - 0.1, t, fs=6.6, ha="left", fontweight="bold")
    if k < 4:
        F.arrow((x + w + 0.005, y0 + h / 2), (x + w + gap - 0.005, y0 + h / 2), lw=0.9)
top = y0 + h - 0.24  # first content line

# 1 split
x = xs[0]
for r, (c, name, role, n) in enumerate(((BLUE, "probe pool", "fits readouts", "400"),
                                        (DGREY, "validation", "selects settings", ""),
                                        (ORANGE, "fresh parts", "give every number", "256"))):
    yy = top - 0.04 - r * 0.24
    for j in range(3):
        F.img(x + 0.07 + j * 0.135, yy - 0.06, G[6 * r + j], 0.12, edge=c, lw=0.7)
    F.text(x + 0.5, yy + 0.03, name + (f"  {n}" if n else ""), fs=5.9, ha="left", fontweight="bold", c=c)
    F.text(x + 0.5, yy - 0.05, role, fs=5.4, ha="left", c=DGREY)

# 2 compose
x = xs[1]
F.img(x + 0.08, top - 0.15, G[30], 0.16, edge=ORANGE); F.text(x + 0.05, top - 0.07, "$x$", fs=6.4)
F.img(x + 0.08, top - 0.52, G[31], 0.16, edge=ORANGE); F.text(x + 0.05, top - 0.44, "$y$", fs=6.4)
over = np.pad(G[30], ((0, 0), (4, 4))) + np.pad(G[31][:, ::-1], ((0, 0), (4, 4))); over = over / over.max()
F.arrow((x + 0.25, top - 0.07), (x + 0.42, top - 0.25)); F.arrow((x + 0.25, top - 0.44), (x + 0.42, top - 0.3))
F.img(x + 0.44, top - 0.36, over, 0.16, edge=DGREY, lw=0.6)
F.text(x + 0.6, top - 0.5, "composite", fs=5.6, c=DGREY)

# 3 encode
x = xs[2]
F.card(x + 0.08, top - 0.42, 0.4, 0.34, DGREY, fill=False, lw=0.7)
F.text(x + 0.28, top - 0.25, "system", fs=6.2, fontweight="bold")
F.arrow((x + 0.48, top - 0.25), (x + 0.6, top - 0.25))
F.state(x + 0.62, top - 0.42, n=7, w=0.065, h=0.05)
F.text(x + 0.655, top - 0.52, "state $h$", fs=5.6, c=DGREY)

# 4 read back
x = xs[3]
F.card(x + 0.08, top - 0.42, 0.5, 0.34, BLUE, fill=False, ls="--", lw=0.7)
F.text(x + 0.33, top - 0.2, "affine", fs=6.0, fontweight="bold"); F.text(x + 0.33, top - 0.31, "readouts", fs=6.0, fontweight="bold")
F.arrow((x + 0.58, top - 0.18), (x + 0.75, top - 0.06)); F.arrow((x + 0.58, top - 0.33), (x + 0.75, top - 0.45))
F.img(x + 0.76, top - 0.13, G[30] * 0.85 + 0.05, 0.15, edge=BLUE); F.text(x + 0.95, top - 0.055, "role 1", fs=5.4, ha="left", c=DGREY)
F.img(x + 0.76, top - 0.53, G[31] * 0.85 + 0.05, 0.15, edge=BLUE); F.text(x + 0.95, top - 0.455, "role 2", fs=5.4, ha="left", c=DGREY)
F.text(x + 0.6, y0 + 0.08, r"error $\varepsilon$ on fresh parts", fs=5.7, c=ORANGE)

# 5 calibrate
x = xs[4]
F.text(x + 0.1, top - 0.02, "rerun 2–4 on two references:", fs=5.6, ha="left", c=DGREY)
F.text(x + 0.1, top - 0.15, r"concatenation $[e(x), e(y)]$", fs=5.9, ha="left", c=BLUE)
F.text(x + w - 0.08, top - 0.15, r"$\varepsilon_{\mathrm{ref}}$", fs=6.2, ha="right", c=BLUE)
F.text(x + 0.1, top - 0.27, r"sum $e(x) + e(y)$", fs=5.9, ha="left", c=DGREY)
F.text(x + w - 0.08, top - 0.27, r"$\varepsilon_{\mathrm{blend}}$", fs=6.2, ha="right", c=DGREY)
F.card(x + 0.08, y0 + 0.07, w - 0.16, 0.3, ORANGE, lw=0.7)
F.text(x + w / 2, y0 + 0.265, r"$\kappa = (\varepsilon_{\mathrm{blend}} - \varepsilon)\,/\,(\varepsilon_{\mathrm{blend}} - \varepsilon_{\mathrm{ref}})$", fs=6.6)
F.text(x + w / 2, y0 + 0.14, "1 for role–filler, 0 for a blend", fs=5.5, c=DGREY)

# ---------------- b: role structure ----------------
yb, hb = 0.04, 1.32
F.head(0.03, 1.47, "b", "Test role structure through the system's decoder")
F.card(0.03, yb, 3.5, hb, PURPLE, lw=0.7)
F.text(1.78, yb + hb - 0.13, r"role maps fitted on the pool:  $h \approx c + B_1 e(x) + B_2 e(y)$,  residual $\delta$", fs=6.3)
tests = [("reversal", "predict $h(y, x)$ from $h(x, y)$", (G[32], G[33]), (G[33], G[32])),
         ("new role", "a known part, an unseen role", (G[34], G[35]), (G[39], G[34])),
         ("interchange", "swap role 1, keep role 2", (G[36], G[37]), (G[38], G[37]))]
cw = 1.06
for k, (t, s, a, b) in enumerate(tests):
    x = 0.12 + k * (cw + 0.08)
    F.card(x, yb + 0.27, cw, 0.8, PURPLE, fill=False, lw=0.6)
    F.text(x + cw / 2, yb + 0.95, t, fs=6.4, fontweight="bold")
    F.text(x + cw / 2, yb + 0.83, s, fs=5.5, c=DGREY)
    F.img(x + 0.09, yb + 0.4, pair(*a), 0.19)
    F.arrow((x + 0.48, yb + 0.495), (x + 0.58, yb + 0.495))
    F.img(x + 0.59, yb + 0.4, pair(*b), 0.19)
    F.arrow((x + cw / 2, yb + 0.27), (x + cw / 2, yb + 0.18), color=PURPLE)
F.text(1.78, yb + 0.1, "the system's own decoder renders every predicted or edited state", fs=6.0, c=PURPLE)

# ---------------- c: representation and behavior ----------------
xc = 3.65
F.head(xc, 1.47, "c", "Measure representation and behavior apart")
F.card(xc, yb, 2.82, hb, GREEN, lw=0.7)
F.state(xc + 0.16, yb + 0.5, n=7, w=0.065, h=0.05)
F.text(xc + 0.19, yb + 0.38, "state $h$", fs=5.6, c=DGREY)
paths = [(yb + 1.08, BLUE, "population readout", "fitted on the pool", "representation"),
         (yb + 0.71, GREEN, "vocabulary readout", "fitted on training parts", "a typical probe"),
         (yb + 0.34, DGREY, "the system's decoder", "trained with the encoder", "behavior")]
for y, c, a, b, tag in paths:
    F.arrow((xc + 0.24, yb + 0.67), (xc + 0.5, y - 0.02), color=c, lw=0.8)
    F.text(xc + 0.53, y + 0.03, a, fs=6.3, ha="left", fontweight="bold", c=c)
    F.text(xc + 0.53, y - 0.09, b, fs=5.6, ha="left", c=DGREY)
    F.text(xc + 2.72, y - 0.02, tag, fs=6.0, ha="right", c=c, style="italic")
F.text(xc + 1.41, yb + 0.1, "each also scored with untrained or random weights", fs=5.7, c=DGREY)
F.save("fig_protocol")
print("ok")
