"""E37 planners: a language model assigns the roles in each E31 caption before GLIGEN draws it.
E9's few-shot format: four demonstrations on imageable fitting nouns (one per phrasing x question), then
"Sentence: <caption>.\nQuestion: <figure question>\nAnswer: the"; the planner's figure is the noun whose continuation is
more probable.  One choice per (relation, pair, phrasing); the noise seed does not enter.
Writes results/e37/plan_<model>.json: {"<rel>|<pair>|<phr>": figure noun}.  usage: e37_plan.py --models a+b"""
import argparse, json, os
import numpy as np, torch
from e31_common import *
from e9_behavior import fmt, seq_logprob

Q = {"on": ("Which object is on top?", "Which object is underneath?"), "above": ("Which object is higher up?", "Which object is lower down?"),
     "left": ("Which object is on the left?", "Which object is on the right?")}
OUT = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e37"

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--npairs", type=int, default=150)
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
im = imageable()
for name in a.models.split("+"):
    fn = f"{OUT}/plan_{name.replace('/', '_')}.json"
    if os.path.exists(fn):
        continue
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda").eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    out, acc = {}, {}
    for rel, (canon, conv, axis) in RELS.items():
        rng = np.random.default_rng(500 + list(RELS).index(rel))
        dn = rng.choice(im["fit"], 8, replace=False)
        demo = []
        for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)]):
            x, y = dn[2 * k], dn[2 * k + 1]
            s = (canon, conv)[p].format(x=x, y=y)
            demo.append(fmt(s, Q[rel][q]) + f" {(x, y)[q]}\n\n")
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(4))
        prs = sample_pairs(im["new"], a.npairs, 100 + list(RELS).index(rel))
        for p, tmpl in enumerate((canon, conv)):
            prompts = [head + fmt(tmpl.format(x=x, y=y), Q[rel][0]) for x, y in prs]
            lx = seq_logprob(tok, mod, prompts, [" " + x for x, y in prs], "cuda")
            ly = seq_logprob(tok, mod, prompts, [" " + y for x, y in prs], "cuda")
            for pi, ((x, y), u, v) in enumerate(zip(prs, lx, ly)):
                out[f"{rel}|{pi}|{('canon', 'conv')[p]}"] = x if u > v else y
            acc[f"{rel}|{('canon', 'conv')[p]}"] = float(np.mean(lx > ly))
    print(name, {k: round(v, 3) for k, v in acc.items()}, flush=True)
    json.dump(dict(model=name, choices=out, figure_acc=acc), open(fn, "w", encoding="utf-8"), indent=1)
    del mod; torch.cuda.empty_cache()
