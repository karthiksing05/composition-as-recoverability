"""Case-study panels from results/case_lm (per-sentence answers and readouts) for fig_story.py.

A returned noun is drawn in the color of the measurement that produced it (orange for the model's
answer, blue for a readout of the question state, grey for a readout of the sentence average) when it
is the figure, and in red when it is the ground, i.e. when it follows word order."""
import json
from figstyle import *

FILES = {"phi-4": "microsoft_phi-4", "Qwen3-8B": "Qwen_Qwen3-8B", "OLMo-2 7B": "allenai_OLMo-2-1124-7B",
         "Pythia-1.4b": "EleutherAI_pythia-1.4b", "GPT-2 large": "gpt2-large"}


def load(model):
    return json.load(open(f"results/case_lm/{FILES[model]}.json", encoding="utf-8"))


def cases(model, figure):
    d = load(model)
    return [c for c in d["cases"] if c["figure"] == figure], d["aggregates"]


def word(ax, x, y, c, key, col, **kw):
    ret = c[key]["preferred"] if key == "answer" else c[key]["returned"]
    ok = ret == c["figure"]
    ax.text(x, y, ret, color=col if ok else RED, fontweight="bold" if ok else "normal", transform=ax.transAxes, va="center", **kw)


def block(ax, x0, y0, w, model, figure, title=None, cols=("answer", "readout_query", "readout_mean")):
    cs, agg = cases(model, figure)
    cs = sorted(cs, key=lambda c: c["phrasing"] != "canonical")
    HEAD = {"answer": ("answer", BEH), "readout_query": ("question\nstate", REP), "readout_mean": ("sentence\naverage", DGREY)}
    xs = [x0 + w * f for f in ((0.0, 0.52, 0.72, 0.9) if len(cols) == 3 else (0.0, 0.6, 0.82))]
    ax.text(x0, y0, title or model, fontsize=7.2, fontweight="bold", color=INK, transform=ax.transAxes, va="bottom")
    for j, k in enumerate(cols):
        ax.text(xs[j + 1], y0 - 0.08, HEAD[k][0], color=HEAD[k][1], fontsize=6.2, transform=ax.transAxes, va="center", linespacing=1.0)
    for i, c in enumerate(cs):
        y = y0 - 0.3 - i * 0.2
        ax.text(xs[0], y, f"{c['sentence']}", fontsize=6.8, color=INK, transform=ax.transAxes, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LGREY, lw=0.6))
        for j, k in enumerate(cols):
            word(ax, xs[j + 1], y, c, k, HEAD[k][1], fontsize=6.8)
    return agg


def draw(ax, which):
    if which == "both":
        panel(ax, "a", "Two models that answer by meaning: the question state returns the figure in both phrasings")
        ax.text(0.0, 0.9, "Question: Which object is on top?  Bold: the figure.  Red: the noun that word order picks.", fontsize=6.4, color=DGREY, transform=ax.transAxes, va="center")
        block(ax, 0.0, 0.68, 0.48, "phi-4", "cake")
        block(ax, 0.52, 0.68, 0.48, "Qwen3-8B", "leaf")
    elif which == "neither":
        panel(ax, "b", "Models that answer by word order: answers and readouts follow position")
        ax.text(0.0, 0.9, "Question: Which object is on top?  Bold: the figure.  Red: the noun that word order picks.", fontsize=6.4, color=DGREY, transform=ax.transAxes, va="center")
        block(ax, 0.0, 0.68, 0.47, "Pythia-1.4b", "cake", cols=("answer", "readout_query"))
        block(ax, 0.53, 0.68, 0.47, "GPT-2 large", "leaf", cols=("answer", "readout_query"))
