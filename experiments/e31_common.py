"""Shared definitions for e31/e32 (diffusion tests D1, D2, D4): noun split, relations, captions."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_pretrained as e4

ROOT = "/coc/flash10/ksingara3/compositionality-theory"
R31 = f"{ROOT}/results/e31"
R32 = f"{ROOT}/results/e32"

# (canonical template, converse template, axis, figure is first slot?)  figure = {x}; first slot = top / left
RELS = {
    "on":    ("a {x} on a {y}", "a {y} under a {x}", "v"),
    "above": ("a {x} above a {y}", "a {y} below a {x}", "v"),
    "left":  ("a {x} to the left of a {y}", "a {y} to the right of a {x}", "h"),
}


def split(seed=0):
    """Same noun split as e4_pretrained.run (shuffle with default_rng(seed) of the deduplicated list)."""
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(e4.NOUNS))
    rng.shuffle(nouns)
    return dict(fit=nouns[:165], val=nouns[165:205], fresh=nouns[205:305], all=nouns)


def imageable():
    with open(f"{R31}/imageable.json", encoding="utf-8") as f:
        return json.load(f)


def sample_pairs(nouns, k, seed):
    rng = np.random.default_rng(seed)
    prs = [(x, y) for x in nouns for y in nouns if x != y]
    return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
