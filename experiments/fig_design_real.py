"""Experimental design, part 2: the same protocol on pretrained language and image models, as a flow chart."""
import io
import numpy as np
from PIL import Image
from diagram import *
import e8_vision as V

F = Flow(6.5, 2.4)
ims, labels = V.load_images()
pick = [int(np.where(labels == c)[0][0]) for c in (52, 77, 101, 140, 166, 188)]
P = [Image.open(io.BytesIO(ims[i])).convert("RGB") for i in pick]
arr = lambda im: np.asarray(im, np.float32) / 255

# ---------------- a: two input lanes through one pipeline ----------------
F.head(0.03, 2.3, "a", "Sentences and photographs pass through one pipeline")
F.card(0.03, 0.04, 3.93, 2.16, GREY, lw=0.6, ec=LGREY)
mx, mw = 1.42, 0.62  # the shared model
F.card(mx, 0.2, mw, 1.66, DGREY, fill=False, lw=0.8)
F.text(mx + mw / 2, 1.15, "pretrained\nmodel", fs=6.8, fontweight="bold", linespacing=1.1)
F.text(mx + mw / 2, 0.84, "trained or\nrandom\nweights", fs=5.6, c=DGREY, linespacing=1.15)
ex, rx, ox = 2.14, 2.86, 3.5  # part code / state, readout, outputs
for lane, (yc, lab) in enumerate(((1.62, "sentences: five relations and “a $x$ and a $y$”"), (0.66, "photographs: three layouts"))):
    F.text(0.12, yc + 0.43, lab, fs=5.8, ha="left", c=DGREY, style="italic")
    yp, ys = yc + 0.17, yc - 0.17  # part row, composite row
    if lane == 0:
        F.chip(0.14, yp, "a tulip", ORANGE); F.chip(0.14, ys, "a tulip on a barrel", ORANGE)
        F.arrow((0.56, yp), (mx, yp)); F.arrow((1.12, ys), (mx, ys))
    else:
        F.img(0.14, yp - 0.12, arr(V.canvas([(P[0], 112, (56, 56))]))[40:184, 40:184], 0.24, edge=ORANGE, lw=0.7, gray=False)
        comp = arr(V.compose("beside", P[0], P[1]))[56:168]
        F.img(0.38, ys - 0.12, comp, 0.24, edge=ORANGE, lw=0.7, gray=False)
        F.arrow((0.4, yp), (mx, yp)); F.arrow((0.88, ys), (mx, ys))
    F.text(1.16, yp + 0.08, "part", fs=5.4, c=DGREY); F.text(1.16, ys + 0.08, "composite", fs=5.4, c=DGREY)
    # model outputs: the part code and the composite's state
    F.arrow((mx + mw, yp), (ex, yp)); F.arrow((mx + mw, ys), (ex + 0.12, ys))
    F.card(ex, yp - 0.1, 0.58, 0.2, BLUE, lw=0.7)
    F.text(ex + 0.29, yp, "part code $e(x)$", fs=5.8)
    F.state(ex + 0.13, ys - 0.15, n=6, w=0.06, h=0.05)
    F.text(ex + 0.33, ys - 0.03, "state $h$", fs=5.6, c=DGREY, ha="left")
    F.arrow((ex + 0.33, ys - 0.12), (rx, ys - 0.12))
    F.card(rx, ys - 0.27, 0.5, 0.3, BLUE, fill=False, ls="--", lw=0.7)
    F.text(rx + 0.25, ys - 0.12, "readouts", fs=6.0, fontweight="bold")
    F.arrow((ex + 0.58, yp), (rx + 0.25, ys + 0.03), color=BLUE, lw=0.6, elbow=True)
    F.text(rx + 0.3, yp + 0.03, "target", fs=5.2, c=BLUE, ha="left")
    F.arrow((rx + 0.5, ys - 0.06), (ox, ys + 0.05)); F.arrow((rx + 0.5, ys - 0.18), (ox, ys - 0.29))
    if lane == 0:
        F.chip(ox + 0.02, ys + 0.05, "tulip", BLUE, fs=5.8); F.chip(ox + 0.02, ys - 0.29, "barrel", BLUE, fs=5.8)
    else:
        F.img(ox + 0.02, ys - 0.04, arr(P[0].resize((96, 96))), 0.19, edge=BLUE, lw=0.7, gray=False)
        F.img(ox + 0.02, ys - 0.39, arr(P[1].resize((96, 96))), 0.19, edge=BLUE, lw=0.7, gray=False)
F.text(0.12, 0.11, "fresh nouns and photograph classes never fit a readout", fs=5.6, c=ORANGE, ha="left")

# ---------------- b: models ----------------
xb, wb = 4.08, 2.39
F.head(xb, 2.3, "b", "Models across scale and training")
F.card(xb, 1.05, wb, 1.15, BLUE, lw=0.6)
rows = [("scale", "Pythia, 7 sizes from 160M to 12B"), ("training", "Pythia-1.4b, 6 checkpoints from step 0"),
        ("other LLMs", "Qwen3-8B, OLMo-2 7B and 13B, phi-4"), ("text encoders", "CLIP, T5, BERT, GPT-2 (11 models)"),
        ("image encoders", "CLIP, DINOv2, SigLIP (6 models)")]
for k, (a, b) in enumerate(rows):
    y = 2.05 - k * 0.165
    F.text(xb + 0.08, y, a, fs=5.9, ha="left", fontweight="bold", c=BLUE)
    F.text(xb + 0.8, y, b, fs=5.8, ha="left")
F.text(xb + wb / 2, 1.14, "every encoder also runs with random weights", fs=5.6, c=DGREY)

# ---------------- c: roles ----------------
F.head(xb, 0.88, "c", "Do roles follow meaning or word order?")
F.card(xb, 0.04, wb, 0.76, RED, lw=0.6)
F.text(xb + 0.08, 0.66, "fit", fs=5.6, ha="left", c=DGREY); F.chip(xb + 0.3, 0.66, "a tulip on a barrel", GREY, fs=5.9)
F.text(xb + 0.08, 0.38, "test", fs=5.6, ha="left", c=DGREY); F.chip(xb + 0.3, 0.38, "a barrel under a tulip", GREY, fs=5.9)
ox2 = xb + 1.62
F.text(ox2 + 0.35, 0.68, "figure readout returns", fs=5.4, c=DGREY)
F.arrow((xb + 1.45, 0.38), (ox2, 0.47)); F.arrow((xb + 1.45, 0.38), (ox2, 0.2))
F.card(ox2, 0.36, 0.7, 0.23, GREEN, lw=0.7); F.text(ox2 + 0.35, 0.475, "tulip: meaning", fs=5.8, fontweight="bold")
F.card(ox2, 0.09, 0.7, 0.23, ORANGE, lw=0.7); F.text(ox2 + 0.35, 0.205, "barrel: word order", fs=5.8, fontweight="bold")
F.save("fig_design_real")
print("ok")
