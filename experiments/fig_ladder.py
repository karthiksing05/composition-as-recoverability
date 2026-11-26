"""Results overview: behavior (own procedure) against the state (declared readout), both on held-out parts.

Every score is scaled so that 0 is a blend, chance or word order and 1 is exact recovery:
  R^2 for glyph and tree states (broad probe) and outputs; kappa (probe R^2 rescaled by the sum and concatenation references) for the probed states of
  image and diffusion systems; transfer of the figure readout for language-model question states (0 = word order, like the 2a - 1
  behavior, both measure converse success); answer and caption-matching accuracy a rescaled to 2a - 1.
Every plotted number is computed below from files in results/ (loaders with the file and statistic in their docstrings); ranges give the
spread over models, architectures or relations, and the marker sits at the midpoint. Green arrows: procedures that read the state.
Running the script also prints a table of value used before, value from files and value in the paper text, with mismatches flagged."""
from figstyle import *
from matplotlib.patches import FancyArrowPatch, Polygon
import csv, glob, json
import numpy as np, pandas as pd
from agg_r2 import load as r2load, e2cols

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results"
acc = lambda a: 2 * a - 1


# ---------------------------------------------------------------- loaders: every plotted number is computed from files in results/
def jrows(pattern):
    """All records of the json files matching pattern (each file holds a dict or a list of dicts)."""
    out = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return out


def lm_state():
    """results/e11 (question state, 3 seeds): xfer_role_conv = figure probe fitted on canonical sentences, scored on converse ones, at the
    (layer, code) with the best validation score (sem_rec_val + xfer_rec_val) pooled over seeds and relations (POOLED rule of
    audit_lm_selection.py / e35/audit_lm_selection.txt); then mean over relations and seeds."""
    d = pd.DataFrame(jrows(f"{R}/e11/*.json"))
    d["model"] = d.model.astype(str).str.split("/").str[-1]
    d = d[(d.system == "model") & (d.state == "query")].copy()
    d["val"] = d.sem_rec_val + d.xfer_rec_val
    out = {}
    for m, g in d.groupby("model"):
        layer, code = g.groupby(["layer", "code"]).val.mean().idxmax()
        gp = g[(g.layer == layer) & (g.code == code)]
        out[m] = float(gp.groupby("seed").xfer_role_conv.mean().mean())
    return out


def lm_answers():
    """results/agg_r8_e9.csv (agg_r8.py over results/e9): answer accuracy 'choice' per model, main prompt, mean over seeds and relations."""
    return {r["model"]: float(r["choice"]) for r in csv.DictReader(open(f"{R}/agg_r8_e9.csv", encoding="utf-8"))}


def olmo7_other_layer():
    """results/e26L32, OLMo-2 7B question state at layer 32: the LINEAR ridge probe's transfer (ridge_conv), mean over seeds; the 0.51 of the text."""
    d = pd.DataFrame(jrows(f"{R}/e26L32/*.json"))
    d = d[(d.model.astype(str).str.endswith("OLMo-2-1124-7B")) & (d.state == "query")]
    return float(d.groupby("seed")["ridge_conv"].mean().mean())


def glyphs():
    """results/e2v2 (OVER relation, 3 architectures slot/mlp/cnn, m = 64, as fig_e2.py): per V the (min, max) over architectures of the
    mean over seeds and relation indices of pool_fresh_1 (broad probe, state; both roles) and beh_fresh_1 (own output on fresh glyphs)."""
    df = e2cols(r2load("e2v2"))
    d = df[(df.m == 64) & df.arch.isin(["slot", "mlp", "cnn"])]
    # kappa = (R2 - sum reference) / (concatenation reference - sum reference), references of results/e2refs/refs.json at m = 64 averaged
    # over all rows, as agg_r4.py (e2h block) does
    rf = pd.DataFrame(json.load(open(f"{R}/e2refs/refs.json", encoding="utf-8")))
    cc, ss = rf.concat_m64.mean(), rf.sum_m64.mean()
    out = {}
    for V in (8, 64):
        t = d[d.V == V].groupby("arch")[["pool_fresh_1", "beh_fresh_1"]].mean()
        t["kappa"] = (t.pool_fresh_1 - ss) / (cc - ss)
        out[V] = {k: (float(t[k].min()), float(t[k].max())) for k in t}
    return out


def new_decoder():
    """results/e2redec: encoder trained on 8 glyphs, frozen, new decoder (redec_trained) on 64 other glyphs; mean of r10, r11 per file,
    then over architectures slot/mlp/cnn (fig_e2.py)."""
    rows = [r for r in jrows(f"{R}/e2redec/*.json") if r["arch"] in ("slot", "mlp", "cnn")]
    t = pd.DataFrame([dict(arch=r["arch"], v=np.mean([r["redec_trained"]["r10"], r["redec_trained"]["r11"]])) for r in rows])
    return float(t.groupby("arch").v.mean().mean())


def tree():
    """results/e3v2, recursive composer (raam), V = 16, width 256, depth 3, 5 seeds: pop_r2 (broad probe of the root state) and
    fresh_r2 (own decomposer); fig_e3e4.py panel a."""
    e3 = pd.DataFrame(jrows(f"{R}/e3v2/*.json"))
    s = e3[(e3.model == "raam") & (e3.V == 16) & (e3.w == 256) & (e3.depth == 3)]
    return float(s.pop_r2.mean()), float(s.fresh_r2.mean())


def enc_state():
    """results/agg_r8_e8grid.csv (agg_r8.py over results/e8grid): kappa of quadrant-pooled states, fixed places, CLIP-B/32, CLIP-H/14, SigLIP;
    (min, max)."""
    k = {r["model"]: float(r["kappa"]) for r in csv.DictReader(open(f"{R}/agg_r8_e8grid.csv", encoding="utf-8")) if r["pool"] == "grid"}
    v = [k[m] for m in ("clip-b32-img", "clip-h14-img", "siglip-so400m")]
    return min(v), max(v)


def enc_swapped():
    """results/e17 image-text matching: share of canvases where the correct caption beats the one with swapped places ('roles'),
    mean over seeds and relations, per model; (min, max) over CLIP-B/32, CLIP-H/14, SigLIP."""
    v = [np.mean([r["roles"] for r in jrows(f"{R}/e17/{m}_s*.json")]) for m in ("clip-b32", "clip-h14", "siglip-so400m")]
    return float(min(v)), float(max(v))


def unet_state_one(name):
    """results/e30/agg_kappa.csv (agg_e30.py): kappa of quadrant-pooled denoiser features (pool G) against the model's own part vectors,
    fixed places, trained weights, all relations (name sd15 and sdxl are the 512 px runs)."""
    for r in csv.DictReader(open(f"{R}/e30/agg_kappa.csv", encoding="utf-8")):
        if r["name"] == name and r["source"] == "trained" and r["codes"] == "own" and r["pool"] == "G" and r["rel"] == "ALL":
            return float(r["kappa"])


def unet_state():
    """(min, max) of unet_state_one over SD 1.5 and SDXL, both at 512 px."""
    k = [unet_state_one("sd15"), unet_state_one("sdxl")]
    return min(k), max(k)


def unet_swapped():
    """results/e30/{sd15,sdxl-1024}_classifier_s*_*.json: share of canvases where the denoising error prefers the correct caption over the
    one with swapped places ('roles'), mean over seeds and relations; (min, max) over SD 1.5 (512 px) and SDXL (1,024 px)."""
    v = [np.mean([r["roles"] for r in jrows(f"{R}/e30/{m}_classifier_s*_*.json")]) for m in ("sd15", "sdxl-1024")]
    return float(min(v)), float(max(v))


def dit_point(name, clf):
    """Transformer denoiser: state = results/e30/agg_kappa.csv kappa of quadrant-pooled features (pool G), trained weights, own part vectors,
    fixed places, rel ALL; behavior = swapped-place 'roles' share of results/e30/{clf}_classifier_s*_*.json (1,024 px), mean over seeds and relations."""
    k = None
    for r in csv.DictReader(open(f"{R}/e30/agg_kappa.csv", encoding="utf-8")):
        if r["name"] == name and r["source"] == "trained" and r["codes"] == "own" and r["pool"] == "G" and r["rel"] == "ALL":
            k = float(r["kappa"])
    return k, float(np.mean([r["roles"] for r in jrows(f"{R}/e30/{clf}_classifier_s*_*.json")]))


_lm, _ans, _g = lm_state(), lm_answers(), glyphs()
_wo = [m for m in _ans if m.startswith(("pythia", "gpt2")) and "@" not in m]  # every Pythia and GPT-2 model (sec:behavior)
_wo_state = [_lm[m] for m in ("gpt2-large", "pythia-1.4b", "pythia-12b")]      # word-order models with question-state rows in e11
_enc_s, _enc_b = enc_state(), enc_swapped()
_un_s, _un_b = unet_state(), unet_swapped()
_tree_probe, _tree_own = tree()
_other = olmo7_other_layer()
_newdec = new_decoder()
_pix_s, _pix_b = dit_point("pixart", "pixart-1024")
_lum_s, _lum_b = dit_point("lumina", "lumina-1024")
_pair = lambda v: (v, v)
_accr = lambda t: (acc(t[0]), acc(t[1]))

# name, (state lo, hi), (behavior lo, hi), label offset (dx, dy), ha
PTS = [
    ("phi-4", _pair(_lm["phi-4"]), _pair(acc(_ans["phi-4"])), (0.035, 0.0), "left"),             # e11 transfer; e9 answers
    ("Qwen3-8B", _pair(_lm["Qwen3-8B"]), _pair(acc(_ans["Qwen3-8B"])), (-0.035, 0.0), "right"),
    ("OLMo-2 13B", _pair(_lm["OLMo-2-1124-13B"]), _pair(acc(_ans["OLMo-2-1124-13B"])), (-0.035, 0.0), "right"),
    ("OLMo-2 7B", _pair(_lm["OLMo-2-1124-7B"]), _pair(acc(_ans["OLMo-2-1124-7B"])), (0.03, 0.04), "left"),
    ("Pythia, GPT-2", (min(_wo_state), max(_wo_state)), _accr((min(_ans[m] for m in _wo), max(_ans[m] for m in _wo))), (0.04, -0.07), "left"),
    ("glyphs, V = 64", _g[64]["pool_fresh_1"], _g[64]["beh_fresh_1"], (0.025, -0.08), "left"),   # e2v2 broad-probe R^2 (state) and output R^2, m = 64
    ("glyphs, V = 8", _g[8]["pool_fresh_1"], _g[8]["beh_fresh_1"], (-0.035, 0.0), "right"),
    # not in saved results: no sum/concatenation reference for trees in e3v2, so the tree state stays the probe's R^2
    ("tree, depth 3", _pair(_tree_probe), _pair(_tree_own), (0.0, -0.1), "right"),             # e3v2, raam, V = 16, w = 256
    ("image encoders, places", _enc_s, _accr(_enc_b), (0.0, -0.075), "center"),   # e8grid kappa; e17 swapped-caption share
    ("U-Nets, places", _un_s, _accr(_un_b), (0.04, 0.065), "left"),  # e30 kappa (SD 1.5 and SDXL at 512 px); classifier SD 1.5 at 512 px, SDXL at 1,024 px
    ("PixArt-Σ", _pair(_pix_s), _pair(acc(_pix_b)), (-0.035, -0.02), "right"),   # e30 kappa trained own G ALL; 1,024 px classifier
    ("Lumina 2.0", _pair(_lum_s), _pair(acc(_lum_b)), (0.03, 0.0), "left"),
]
REPAIRS = [  # (from point name, to behavior value, label)  sec:success
    ("glyphs, V = 8", _newdec, "new decoder"),   # e2redec
]

# ---------------------------------------------------------------- table: before (hand-typed constants), from files, paper text
fmt3 = lambda t: f"{t[0]:.3f}" if abs(t[1] - t[0]) < 5e-4 else f"{t[0]:.3f} to {t[1]:.3f}"
BEFORE = {  # (state, behavior) of the hand-typed / R^2 version of this script
    "phi-4": ((.98, .98), (acc(.99),) * 2), "Qwen3-8B": ((.87, .87), (acc(.99),) * 2), "OLMo-2 13B": ((.73, .73), (acc(.92),) * 2),
    "OLMo-2 7B": ((.20, .20), (acc(.77),) * 2), "Pythia, GPT-2": ((.07, .10), (acc(.41), acc(.56))),
    "glyphs, V = 64": ((.89, .89), (.85, .91)), "glyphs, V = 8": ((.83, .86), (.20, .34)), "tree, depth 3": ((.99, .99), (.26, .26)),
    "image encoders, places": ((.81, .94), (acc(.50), acc(.51))), "U-Nets, places": ((.43, .70), (acc(.49), acc(.50))),
    "PixArt-Σ": None, "Lumina 2.0": None,
    "new decoder (behavior)": (None, (.75, .75)), "OLMo-2 7B arrow end (state)": ((.50, .50), None),
}
PAPER = {  # numbers quoted in main.tex / parts/results.tex, on the plotted scale (accuracies through 2a - 1)
    "phi-4": ((.98, .98), (acc(.99),) * 2), "Qwen3-8B": ((.87, .87), (acc(.99),) * 2), "OLMo-2 13B": ((.73, .73), (acc(.92),) * 2),
    "OLMo-2 7B": ((.07, .35), (acc(.77),) * 2),  # text: 0.07-0.35 across selection rules; tab:lm and its caption: 0.20 (ridge, layer 16)
    "Pythia, GPT-2": ((.07, .10), (acc(.41), acc(.56))),
    "glyphs, V = 64": ((.89, .89), (.85, .91)), "glyphs, V = 8": ((.83, .86), (.20, .34)),  # broad-probe R^2, as in the text
    "tree, depth 3": ((.99, .99), (.26, .26)),  # R^2; the paper gives no tree kappa
    "image encoders, places": ((.81, .94), (acc(.50), acc(.51))), "U-Nets, places": ((.43, .70), (acc(.49), acc(.51))),  # SDXL 1,024 px swapped .51 (tab:places)
    "PixArt-Σ": ((.75, .75), (acc(.60), acc(.61))), "Lumina 2.0": ((.89, .89), (acc(.66), acc(.68))),
    "new decoder (behavior)": (None, (.75, .75)), "OLMo-2 7B arrow end (state)": ((.51, .51), None),
}
NOW = {p[0]: (p[1], p[2]) for p in PTS}
NOW["new decoder (behavior)"] = (None, _pair(_newdec))
NOW["OLMo-2 7B arrow end (state)"] = (_pair(_other), None)
ACC_SCALED = {"phi-4", "Qwen3-8B", "OLMo-2 13B", "OLMo-2 7B", "Pythia, GPT-2", "image encoders, places", "U-Nets, places", "PixArt-Σ", "Lumina 2.0"}  # behavior = 2a - 1
# equal when the numbers agree to the paper's two decimals (tolerance 0.0075: rounding of a third decimal); accuracies are compared as accuracies
same = lambda a, b, scale=1.0: all(abs(x - y) / scale < 0.0075 for x, y in zip(a, b))
print(f"{'point':34s}{'axis':9s}{'before':>16s}{'from files':>16s}{'paper text':>16s}  flag")
for name in PAPER:
    for ax_i, axn in enumerate(("state", "behavior")):
        n, p = NOW[name][ax_i], PAPER[name][ax_i]
        b = BEFORE[name][ax_i] if BEFORE[name] else None
        if n is None:
            continue
        sc = 2.0 if (ax_i == 1 and name in ACC_SCALED) else 1.0
        flags = []
        if b is not None and not same(b, n, sc):
            flags.append("changed vs before")
        if not same(n, p, sc):
            flags.append("MISMATCH paper")
        print(f"{name:34s}{axn:9s}{(fmt3(b) if b else 'new'):>16s}{fmt3(n):>16s}{fmt3(p):>16s}  {'; '.join(flags)}")
print("resolutions: U-Nets state = SD 1.5 at 512 px (kappa %.3f) and SDXL at 512 px (e30 'sdxl', kappa %.3f); U-Nets behavior = SD 1.5 at 512 px (%.3f) "
      "and SDXL at 1,024 px (sdxl-1024, %.3f); PixArt-Sigma and Lumina behavior at 1,024 px, state from e30 agg_kappa.csv (trained own-code, native feature maps)"
      % (unet_state_one("sd15"), unet_state_one("sdxl"), np.mean([r["roles"] for r in jrows(f"{R}/e30/sd15_classifier_s*_*.json")]),
         np.mean([r["roles"] for r in jrows(f"{R}/e30/sdxl-1024_classifier_s*_*.json")])))

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
print("final coordinates (state, behavior) of the plotted markers:")
for name, (s, b) in pos.items():
    print(f"  {name:26s} ({s:6.3f}, {b:6.3f})  {'ABOVE diagonal (behavior > state)' if b > s else 'below diagonal'}")
for name, to, lab in REPAIRS:
    s, b = pos[name]
    ax.add_patch(FancyArrowPatch((s, b + 0.04), (s, to - 0.01), arrowstyle="-|>", mutation_scale=8, color=FIX, lw=1.2, zorder=2))
    ax.text(s + 0.025, (b + to) / 2 + 0.02, lab, fontsize=6.2, color=FIX, ha="left", va="center", rotation=90)
s, b = pos["OLMo-2 7B"]
ax.add_patch(FancyArrowPatch((s + 0.01, b - 0.035), (_other, b - 0.035), arrowstyle="-|>", mutation_scale=7, color=REP, lw=0.9, ls="-", zorder=2))
ax.text(0.25, b - 0.11, "best probe,\nother layers", fontsize=6.0, color=REP, ha="center", va="center")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("state: linear probe on parts new to the probe", color=REP)
ax.set_ylabel("behavior: own output on new parts", color=BEH)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_aspect("equal")
fig.savefig(f"{FIGDIR}/fig_ladder.pdf"); fig.savefig(f"{NOTES}/fig_ladder.png", dpi=170)
print("saved")
