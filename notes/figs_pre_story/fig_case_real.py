"""Case studies of the large pretrained models: Qwen3-8B paraphrases (E7) and SigLIP retrievals (E8)."""
import io, json
import numpy as np
from PIL import Image
from diagram import *
import e8_vision as V

T = json.load(open("results/case/text.json", encoding="utf-8"))
I = json.load(open("results/case/images.json", encoding="utf-8"))
ims, _ = V.load_images()
photo = lambda i: Image.open(io.BytesIO(ims[i])).convert("RGB")
arr = lambda im: np.asarray(im, np.float32) / 255

C = Canvas(6.5, 2.5)

# ---------------- a: Qwen3-8B, readout of the figure fitted on canonical sentences ----------------
C.title(0.008, 0.955, "a", "Qwen3-8B: the figure readout follows word order")
C.text(0.012, 0.86, "sentence (fresh nouns)", fs=6.0, ha="left", c=DGREY, style="italic")
C.text(0.335, 0.86, "readout of the figure returns", fs=6.0, ha="left", c=DGREY, style="italic")
want = ["yacht", "hat", "shovel"]
ex = [e for e in T["examples"] if e["figure"] in want]
y = 0.76
for e in ex:
    for key, sent, tag in (("pred_canon", e["canon"], "fit phrasing"), ("pred_conv", e["conv"], "converse")):
        C.chip(0.015, y, sent, GREY, fs=6.2)
        C.text(0.235, y, tag, fs=5.6, ha="left", c=DGREY)
        C.arrow((0.297, y), (0.325, y), lw=0.6)
        p = e[key]
        role, c = ("figure", GREEN) if p == e["figure"] else (("ground", ORANGE) if p == e["ground"] else ("other", GREY))
        C.text(0.337, y, p, fs=6.6, ha="left", fontweight="bold", c=c)
        C.text(0.425, y, role, fs=5.8, ha="left", c=c, style="italic")
        y -= 0.095
    y -= 0.03
acc = T["acc"]
C.text(0.012, 0.075, f"over 1,000 fresh sentences, the readout lands nearer the figure in {acc['canon_fit_on_canon']:.0%} of fit phrasings\n"
                     f"and {acc['canon_fit_on_conv']:.1%} of converse ones", fs=5.8, ha="left", c=DGREY, linespacing=1.25)

# ---------------- b: SigLIP, trained against random weights ----------------
x0 = 0.52
C.title(x0, 0.955, "b", "SigLIP: trained weights return fresh photographs")
H = 0.42  # thumbnail height, inches
tw = H / C.W
cols = [("composite", None), ("trained", "siglip-so400m"), ("random weights", "rand-siglip-so400m")]
cx = [x0 + 0.005, x0 + 0.005 + 2 * tw + 0.035, x0 + 0.005 + 4 * tw + 0.06]
for (lab, _), xx in zip(cols, cx):
    C.text(xx + tw * (1 if lab != "composite" else 1), 0.86, lab, fs=6.2, c=BLUE if lab == "trained" else DGREY,
           fontweight="bold" if lab != "composite" else "normal")
for r, (a, b) in enumerate(I["pairs"][:3]):
    yb = 0.62 - r * 0.245
    comp = arr(V.compose("beside", photo(a), photo(b))); comp = comp[56:168]  # drop the empty bands
    C.img(cx[0] + 0.01, yb + H * 0.2 / C.H, comp, H * 0.6, edge=LGREY, lw=0.6, gray=False)
    for k, (_, m) in enumerate(cols[1:]):
        for role, truth in ((0, a), (1, b)):
            p = I["models"][m][f"pred_role{role + 1}"][r]
            C.img(cx[k + 1] + role * (tw + 0.006), yb, arr(photo(p).resize((112, 112))), H,
                  edge=GREEN if p == truth else RED, lw=1.2, gray=False)
for k, (_, m) in enumerate(cols[1:]):
    C.text(cx[k + 1] + tw / 2, 0.82, "left", fs=5.6, c=DGREY); C.text(cx[k + 1] + 1.5 * tw + 0.006, 0.82, "right", fs=5.6, c=DGREY)
    r1, r2 = I["models"][m]["id_role1"], I["models"][m]["id_role2"]
    C.text(cx[k + 1] + tw + 0.003, 0.075, f"{np.mean([r1, r2]):.0%} correct among 300", fs=5.8, c=BLUE if k == 0 else DGREY)
C.text(cx[0] + tw, 0.075, "green: correct photograph", fs=5.6, c=DGREY)
C.save("fig_case_real")
print({m: (v["id_role1"], v["id_role2"]) for m, v in I["models"].items()})
