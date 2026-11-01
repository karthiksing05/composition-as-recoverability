"""E16: do the role-question results of E9 and E11 survive other prompts and a second task?

E9 asks for the figure with one few-shot prompt, and E11 reads the figure from the state at the
end of that prompt.  Here three variants change the prompt or the task:
  reword    other instruction, labels and question wording ("Text: ... / Q: Which thing sits
            higher? / A: the"), four demonstrations as in E9
  zeroshot  E9's instruction and question with no demonstrations
  ground    E9's prompt with the question answered by the ground ("Which object is
            underneath?"), and the readout recovers the ground
For each variant we score the model's own answer on fresh nouns in both phrasings (the answer
noun's continuation beats the other noun's; a word-order shortcut scores 0.5), and the E11 transfer
readout of the state at the prompt's final token: fitted on canonical sentences of fitting nouns,
penalty and layer/code selected on canonical validation sentences, scored by role assignment on
converse sentences of fresh nouns (0 = word order, 1 = meaning).
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS, readout
from e9_behavior import QUESTIONS, item, fmt, N_DEMO, seq_logprob
from e11_tokens import embed_query

VARIANTS = ("reword", "zeroshot", "ground")
REWORD = {
    "on": "Which thing sits higher?",
    "left": "Which thing is further to the left?",
    "inside": "Which thing is enclosed by the other?",
    "behind": "Which thing is farther from the viewer?",
    "hit": "Which thing struck the other?",
}
REWORD_G = {
    "on": "Which thing sits lower?",
    "left": "Which thing is further to the right?",
    "inside": "Which thing encloses the other?",
    "behind": "Which thing is nearer to the viewer?",
    "hit": "Which thing was struck?",
}
N_FIT, N_VAL, N_FR = 1500, 300, 600


def build(variant, rel, demo_nouns, rng):
    """Returns (prompt(sentence) -> text, question index answered: 0 figure, 1 ground)."""
    if variant == "reword":
        f = lambda s, q: f"Text: {s}.\nQ: {q}\nA: the"
        q, qi, intro = REWORD[rel], 0, "Answer each question about the text with one word.\n\n"
    else:
        f, qi, intro = fmt, (1 if variant == "ground" else 0), "Read each sentence and answer the question with one object.\n\n"
        q = QUESTIONS[rel][qi]
    demo = ""
    if variant != "zeroshot":
        ds = []
        for k, (p, qq) in enumerate([(p, qq) for p in (0, 1) for qq in (0, 1)]):
            sent, _, ans, _, _ = item(rel, demo_nouns[2 * k], demo_nouns[2 * k + 1], p, qq)
            qtext = (REWORD if qq == 0 else REWORD_G)[rel] if variant == "reword" else QUESTIONS[rel][qq]
            ds.append(f(sent, qtext) + f" {ans}\n\n")
        demo = "".join(ds[i] for i in rng.permutation(len(ds)))
    return (lambda s: intro + demo + f(s, q)), qi


def run(name, seed, variant, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    from transformers import AutoTokenizer, AutoModelForCausalLM  # as in E9: the model must answer
    repo, _, rev = name.partition("@")
    kw = dict(revision=rev) if rev else {}
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    mod = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, **kw).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    codes = {"phrase": E.embed(tok, mod, [f"a {x}" for x in nouns], dev, name),
             "carrier": E.embed(tok, mod, [f"there is a {x} here" for x in nouns], dev, name)}
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    res = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp, frp = sample(fit_n, N_FIT), sample(val_n, N_VAL), sample(fr_n, N_FR)
        prompt, qi = build(variant, rel, rng.choice(fit_n, 2 * N_DEMO, replace=False), rng)
        # readout rows: fitting and validation pairs canonical, fresh pairs in both phrasings
        texts, meta = [], []
        for split, prs, phs in (("fit", fitp, (0,)), ("val", valp, (0,)), ("fresh", frp, (0, 1))):
            for x, y in prs:
                for ph in phs:
                    texts.append((canon if ph == 0 else conv).format(x=x, y=y)); meta.append((split, ph, x, y))
        H = embed_query(tok, mod, [prompt(t) for t in texts], dev)
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        tgt = [m[2] if qi == 0 else m[3] for m in meta]; oth = [m[3] if qi == 0 else m[2] for m in meta]
        R = np.where(sp == "fresh")[0]
        # the model's own answer on the fresh rows
        lp_t = seq_logprob(tok, mod, [prompt(texts[i]) for i in R], [" " + tgt[i] for i in R], dev)
        lp_o = seq_logprob(tok, mod, [prompt(texts[i]) for i in R], [" " + oth[i] for i in R], dev)
        ok = lp_t > lp_o
        beh = dict(choice=float(ok.mean()), choice_can=float(ok[ph[R] == 0].mean()), choice_conv=float(ok[ph[R] == 1].mean()))
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        F, V = sel(sp == "fit"), sel(sp == "val")
        Rc, Rv = sel(sp == "fresh", ph == 0), sel(sp == "fresh", ph == 1)
        g = lambda rows, L: [L[i] for i in rows]
        for (_, li), Hs in H.items():
            for cname in ("phrase", "carrier"):
                P = codes[cname][f"L{li}"]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                r = readout(Hs, T, idx, F, g(F, tgt), V, g(V, tgt),
                            {"val": (V, g(V, tgt), g(V, oth)), "can": (Rc, g(Rc, tgt), g(Rc, oth)), "conv": (Rv, g(Rv, tgt), g(Rv, oth))})
                res.append(dict(model=name, seed=seed, variant=variant, rel=rel, layer=f"L{li}", code=cname, **beh,
                                **{f"xfer_{k}": v for k, v in r.items()}))
        print(name, seed, variant, rel, beh, max((x["xfer_rec_val"], x["layer"], x["code"], round(x["xfer_role_conv"], 3))
                                                  for x in res if x["rel"] == rel), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--outdir", default="results/e16")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            for v in a.variants.split(","):
                fn = os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}_{v}.json")
                if os.path.exists(fn):
                    continue
                r = run(name, seed, v, dev)
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(r, f, indent=1)
