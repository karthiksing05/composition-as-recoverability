"""E14 (remedy S3): pool by query.  Do role-question states give sentence embeddings that keep roles?

E11 finds that a language model's sentence states bind nouns to word order, while the state at
the end of a question about a role carries the semantic role.  Here no readout is fitted: each
sentence is embedded and compared by cosine similarity.  For a fresh pair (x, y) and relation r:
  S   canonical(x, y)   "a x on a y"         the query
  P   converse(x, y)    "a y under a x"      same meaning, reversed order of mention
  R1  canonical(y, x)   "a y on a x"         reversed roles, same relation word
  R2  converse(y, x)    "a x under a y"      reversed roles, same order of mention
Embeddings:
  mean   the token-averaged state (E7's state)
  last   the state at the sentence's final token
  roleq  the states at the end of the figure question and of the ground question (E9's prompt,
         four demonstrations with fitting nouns), each centred and normalized, concatenated
  roleqf the figure-question state alone
Scores, on the fresh nouns: pair (P is more similar to S than R1 and R2 are; chance 1/3), and
top1 (P is the nearest of all other sentences of the relation).  Layer per embedding is chosen
on validation nouns.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO
from e11_tokens import embed_tokens, embed_query, spans

N_FRESH_PAIRS, N_VAL_PAIRS = 200, 100


def norm(X, mu):
    X = X - mu
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def scores(Z, n):
    """Z holds 4 blocks of n rows: S, P, R1, R2.  Cosine on rows already normalized."""
    S, P, R1, R2 = Z[:n], Z[n:2 * n], Z[2 * n:3 * n], Z[3 * n:]
    sp, s1, s2 = (S * P).sum(1), (S * R1).sum(1), (S * R2).sum(1)
    pair = float(((sp > s1) & (sp > s2)).mean())
    sim = S @ Z.T
    sim[np.arange(n), np.arange(n)] = -np.inf
    top1 = float((sim.argmax(1) == n + np.arange(n)).mean())
    return dict(pair=pair, top1=top1, beats_r1=float((sp > s1).mean()), beats_r2=float((sp > s2).mean()))


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
    tok.padding_side = "right"
    res = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        for split, prs in (("val", sample(val_n, N_VAL_PAIRS)), ("fresh", sample(fr_n, N_FRESH_PAIRS))):
            n = len(prs)
            items = [(canon, x, y) for x, y in prs] + [(conv, x, y) for x, y in prs] + \
                    [(canon, y, x) for x, y in prs] + [(conv, y, x) for x, y in prs]
            texts = [t.format(x=a, y=b) for t, a, b in items]
            sps = [spans(t, a, b) for t, a, b in items]
            H = embed_tokens(tok, mod, texts, sps, dev)
            Qf = embed_query(tok, mod, [head + fmt(t, QUESTIONS[rel][0]) for t in texts], dev)
            Qg = embed_query(tok, mod, [head + fmt(t, QUESTIONS[rel][1]) for t in texts], dev)
            for (state, li), X in H.items():
                if state == "nouns":
                    continue
                res.append(dict(model=name, seed=seed, rel=rel, split=split, emb=state, layer=f"L{li}",
                                **scores(norm(X, X.mean(0)), n)))
            for (_, li), F in Qf.items():
                G = Qg[("query", li)]
                f, g = norm(F, F.mean(0)), norm(G, G.mean(0))
                res.append(dict(model=name, seed=seed, rel=rel, split=split, emb="roleqf", layer=f"L{li}", **scores(f, n)))
                res.append(dict(model=name, seed=seed, rel=rel, split=split, emb="roleq", layer=f"L{li}",
                                **scores(np.c_[f, g] / np.sqrt(2), n)))
            print(name, rel, split, {r["emb"] + r["layer"]: round(r["pair"], 2) for r in res if r["rel"] == rel and r["split"] == split}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e14")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
