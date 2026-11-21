"""Introduction figure "Same meaning, different order": three compact real cases.
a  SD 1.5, one canonical/converse pair under a shared seed.  Rule: walk the gallery order of fig_case_generation.py (left of, above,
   on; pairs in index order, seed 0, both objects detected in both phrasings) and take the first pair for which the two phrasings give
   the same layout while the figure is in the wrong place in both, i.e. the relation word did not set the layout.  (This is chosen to
   show the failure; fig_case_generation.pdf shows the unselected pairs.)
b  Language models, sentence pair "a cake on a plate" / "a plate under a cake", question and answers from
   experiments/results/case_lm/{microsoft_phi-4,EleutherAI_pythia-1.4b}.json ('cases', field answer.preferred).
c  SigLIP so400m on a fixed-quadrant canvas (E8 BESIDE).  Rule: first pair of results/case/images.json whose two classes are named and
   whose two returned photographs (trained probe) are both right.  The caption numbers are E17's aggregate (results/e17, SigLIP,
   beside, 3 seeds); the probe's returned photographs are saved in results/case/images.json."""
from fig_case_common import *

W, H = 6.75, 1.95
P = Page(W, H)
# ---------------- a
items = None
for rel in ("left", "above", "on"):
    for c, ic, v, iv in pick_pairs("sd15", rel, 100):
        if ic["xfirst"] == iv["xfirst"] and not ic["xfirst"]:
            items = (c, ic, v, iv); break
    if items: break
c, ic, v, iv = items
S = 1.05
P.head(0.01, 0.0, "a", "Image generation (SD 1.5, same seed)")
for k, (r, i) in enumerate(((c, ic), (v, iv))):
    x = 0.01 + k * (S + 0.1)
    P.img(x, 0.2, load_png("sd15", r), S, edge=LGREY)
    P.box(x, 0.2, S, i["by"], GRDC); P.box(x, 0.2, S, i["bx"], FIGC)
    P.text(x, 0.2 + S + 0.04, wrap(r["caption"], 24), fs=6.2, linespacing=1.15)
    t, col = verdict(r, i)
    mark(P, x + 0.01, 0.2 + S + 0.30, i["xfirst"]); P.text(x + 0.13, 0.2 + S + 0.28, t, fs=6.2, c=col, fontweight="bold")

# ---------------- b
xb = 2.55
P.head(xb, 0.0, "b", "Language models")
phi = jload(f"{RES}/case_lm/microsoft_phi-4.json"); pyt = jload(f"{RES}/case_lm/EleutherAI_pythia-1.4b.json")
def case(d, ph):
    return next(x for x in d["cases"] if x["sentence"] in ("a cake on a plate", "a plate under a cake") and x["phrasing"] == ph)
rowsb = [("canonical", "a cake on a plate"), ("converse", "a plate under a cake")]
P.text(xb, 0.24, "Which object is on top?", fs=6.2, c=DGREY, style="italic")
P.text(xb + 1.02, 0.46, "phi-4", fs=6.2, c=DGREY, fontweight="bold"); P.text(xb + 1.50, 0.46, "Pythia", fs=6.2, c=DGREY, fontweight="bold")
ans = []
for k, (ph, sent) in enumerate(rowsb):
    y = 0.68 + k * 0.30
    cp, cy = case(phi, ph), case(pyt, ph)
    assert cp["sentence"] == sent and cy["sentence"] == sent
    P.text(xb, y, sent, fs=6.2)
    for dx, cs in ((1.02, cp), (1.50, cy)):
        a = cs["answer"]["preferred"]; ok = a == cs["figure"]
        mark(P, xb + dx, y + 0.015, ok, s=0.07)
        P.text(xb + dx + 0.11, y, a, fs=6.2, c=OKC if ok else BADC, fontweight="bold")
        ans.append((ph, a))
P.text(xb, 1.34, f"Answers right over all sentences:\nphi-4 {phi['headline']['answers']:.2f}, Pythia-1.4b {pyt['headline']['answers']:.2f}", fs=6.0, c=DGREY, linespacing=1.25)

# ---------------- c
xc = 4.62
P.head(xc, 0.0, "c", "Image–text encoder (SigLIP)")
V, ims, labels, photo = photo_loader()
names = class_names()
I = jload(f"{RES}/case/images.json")
named = lambda i: names[int(labels[i])] is not None
k_ = next(k for k, (a, b) in enumerate(I["pairs"]) if named(a) and named(b)
          and I["models"]["siglip-so400m"]["pred_role1"][k] == a and I["models"]["siglip-so400m"]["pred_role2"][k] == b)
a, b = I["pairs"][k_]; na, nb = names[int(labels[a])], names[int(labels[b])]
comp = np.asarray(V.compose("beside", photo(a), photo(b)), np.uint8)[56:168]
P.img(xc, 0.2, comp, 0.90, h=0.45, edge=LGREY)
P.text(xc, 0.70, "canvas", fs=6.0, c=DGREY)
P.text(xc + 1.05, 0.2, "probe returns", fs=6.0, c=REP, fontweight="bold")
for role, truth in ((0, a), (1, b)):
    p = I["models"]["siglip-so400m"][f"pred_role{role + 1}"][k_]
    P.img(xc + 1.05 + role * 0.5, 0.34, np.asarray(photo(p).resize((112, 112))), 0.45, edge=GREEN if p == truth else RED, lw=1.2)
P.text(xc, 0.86, "\n".join(textwrap.wrap(f"a photo of a {na} on the left and a {nb} on the right", 46)), fs=6.0, c=FIX, linespacing=1.2)
P.text(xc, 1.14, "\n".join(textwrap.wrap(f"a photo of a {nb} on the left and a {na} on the right", 46)), fs=6.0, c=LOST, linespacing=1.2)
agg = [r for s in range(3) for r in jload(f"{RES}/e17/siglip-so400m_s{s}.json") if r["rel"] in ("beside", "above")]
roles = float(np.mean([r["roles"] for r in agg]))
m = I["models"]["siglip-so400m"]
P.text(xc, 1.44, f"Matching: the correct caption beats the swapped one\nin {roles:.2f} of pairs (chance).  Probe: right photograph\nin {np.mean([m['id_role1'], m['id_role2']]):.0%} of canvases.", fs=6.0, c=DGREY, linespacing=1.25)
P.save("fig_case_motivate")
print("a:", c["file"], c["caption"], "|", v["caption"], ic["xfirst"], iv["xfirst"])
print("c:", a, b, na, nb, "roles", roles, m["id_role1"], m["id_role2"])
print("lm headline", phi["headline"], pyt["headline"])
