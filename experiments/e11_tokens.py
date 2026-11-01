"""E11: do token-level states carry the semantic roles that a language model's answers use?

E7 reads the token-averaged state and finds roles bound to word order, while the same models
answer questions about the figure by meaning (E9).  Here the E7 readouts (ridge, whitened 16-PC
part codes, penalty on validation nouns, fresh nouns for every number) read four states:
  mean   the average over the sentence's tokens (E7's state)
  last   the state at the sentence's final token
  nouns  the states at the last token of the first- and second-mentioned noun, concatenated
  query  the state at the final token of E9's prompt (four demonstrations, the sentence, and the
         question about the figure, ending "Answer: the"), the state the model answers from
Readouts: pos (first-mentioned noun), sec (second-mentioned noun), sem (figure; both fitted on
both phrasings), xfer (figure, fitted on canonical sentences, scored on converse ones).  The
references are the concatenation in order of mention (ref_pos), in semantic order (ref_sem), and
the sum (ref_sum) of the same part codes, so kappa follows for each readout.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS, readout
from e9_behavior import QUESTIONS, item, fmt, N_DEMO

STATES = ("mean", "last", "nouns", "query")


def spans(tmpl, x, y):
    """Character spans of the first- and second-mentioned noun in tmpl.format(x=x, y=y)."""
    i, j = tmpl.index("{"), tmpl.index("{", tmpl.index("}"))
    n1, n2 = (x, y) if tmpl[i + 1] == "x" else (y, x)
    s1 = i
    s2 = s1 + len(n1) + (j - tmpl.index("}") - 1)
    return (s1, s1 + len(n1)), (s2, s2 + len(n2))


def tok_at(offsets, end):
    """Index of the token that holds character end - 1."""
    for k, (a, b) in enumerate(offsets):
        if a <= end - 1 < b:
            return k
    raise ValueError("noun not found in offsets")


@torch.no_grad()
def embed_tokens(tok, mod, texts, noun_spans, dev, bs=64):
    """Per layer: mean over tokens (BOS excluded), last token, and the two noun tokens concatenated."""
    out = {}
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], padding=True, return_tensors="pt", return_offsets_mapping=True)
        offs = b.pop("offset_mapping").tolist()
        b = b.to(dev)
        hs = mod(**b, output_hidden_states=True).hidden_states
        mask = b["attention_mask"].unsqueeze(-1).float()
        if tok.bos_token_id is not None:
            mask[:, 0] *= (b["input_ids"][:, 0] != tok.bos_token_id).float().unsqueeze(-1)
        last = b["attention_mask"].sum(1) - 1
        n1 = torch.tensor([tok_at(o, sp[0][1]) for o, sp in zip(offs, noun_spans[s:s + bs])])
        n2 = torch.tensor([tok_at(o, sp[1][1]) for o, sp in zip(offs, noun_spans[s:s + bs])])
        ar = torch.arange(len(last))
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            h = hs[li]
            out.setdefault(("mean", li), []).append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
            out.setdefault(("last", li), []).append(h[ar, last].float().cpu().numpy())
            out.setdefault(("nouns", li), []).append(torch.cat([h[ar, n1], h[ar, n2]], 1).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def embed_query(tok, mod, texts, dev, bs=32):
    out = {}
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        hs = mod(**b, output_hidden_states=True).hidden_states
        last = b["attention_mask"].sum(1) - 1
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            out.setdefault(("query", li), []).append(hs[li][torch.arange(len(last)), last].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run(name, seed, dev, ckpt=None, only=None):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
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
        fitp, valp, frp = sample(fit_n, 3000), sample(val_n, 600), sample(fr_n, 1000)
        texts, meta, sps = [], [], []
        for split, prs in (("fit", fitp), ("val", valp), ("fresh", frp)):
            for i, (x, y) in enumerate(prs):
                ph = i % 2
                tm = canon if ph == 0 else conv
                texts.append(tm.format(x=x, y=y)); meta.append((split, ph, x, y)); sps.append(spans(tm, x, y))
        # the E9 prompt, with demonstrations of fitting nouns outside the probe pairs' roles
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        # every random draw for this relation is made above, so a finished relation can be skipped
        part = f"{ckpt}.{rel}.part" if ckpt else None
        if part and os.path.exists(part):
            with open(part, encoding="utf-8") as f:
                res += json.load(f)
            continue
        if only and rel not in only:
            continue
        n0 = len(res)
        H = embed_tokens(tok, mod, texts, sps, dev)
        H.update(embed_query(tok, mod, [head + fmt(t, QUESTIONS[rel][0]) for t in texts], dev))
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
        first = [f if p == 0 else g for f, g, p in zip(fig, gnd, ph)]
        second = [g if p == 0 else f for f, g, p in zip(fig, gnd, ph)]
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        F, V, R = sel(sp == "fit"), sel(sp == "val"), sel(sp == "fresh")
        Fc, Vc = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0)
        Rc, Rv = sel(sp == "fresh", ph == 0), sel(sp == "fresh", ph == 1)
        g = lambda Lst, rows: [Lst[i] for i in rows]
        for (state, li), Hall in H.items():
            layer = f"L{li}"
            for cname in ("phrase", "carrier"):
                P = codes[cname][layer]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A = P[[idx[x] for x in first]]; B = P[[idx[x] for x in second]]
                Fg = P[[idx[x] for x in fig]]; Gd = P[[idx[x] for x in gnd]]
                systems = {"model": Hall}
                if state == "mean":  # references depend only on the part codes
                    systems.update(ref_pos=np.c_[A, B], ref_sem=np.c_[Fg, Gd], ref_sum=A + B)
                for sname, Hs in systems.items():
                    row = dict(model=name, seed=seed, rel=rel, state=state, layer=layer, code=cname, system=sname, dim=int(Hs.shape[1]))
                    for key, tgt, oth in (("pos", first, second), ("sec", second, first), ("sem", fig, gnd)):
                        row.update({f"{key}_{k}": v for k, v in readout(Hs, T, idx, F, g(tgt, F), V, g(tgt, V),
                                    {"val": (V, g(tgt, V), g(oth, V)), "fresh": (R, g(tgt, R), g(oth, R)),
                                     "conv": (Rv, g(tgt, Rv), g(oth, Rv))}).items()})
                    row.update({f"xfer_{k}": v for k, v in readout(Hs, T, idx, Fc, g(fig, Fc), Vc, g(fig, Vc),
                                {"val": (Vc, g(fig, Vc), g(gnd, Vc)), "can": (Rc, g(fig, Rc), g(gnd, Rc)),
                                 "conv": (Rv, g(fig, Rv), g(gnd, Rv))}).items()})
                    res.append(row)
                r = [x for x in res if x["system"] == "model"][-1]
                print(name, rel, state, layer, cname, {k: round(r[k], 3) for k in ("pos_rec_fresh", "sec_rec_fresh", "sem_role_fresh", "xfer_role_conv")}, flush=True)
        if part:
            with open(part, "w", encoding="utf-8") as f:
                json.dump(res[n0:], f)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rels", default=None, help="comma list: compute only these relations into checkpoints")
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)
    E.WHITEN = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev, ckpt=fn, only=a.rels.split(",") if a.rels else None)
            if a.rels:
                continue
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
