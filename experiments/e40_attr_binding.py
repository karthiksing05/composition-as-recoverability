"""E40: attribute binding in language models, built on the E33 pipeline.

Wholes: two nouns x, y with two colours.  Canonical "There is a red x and a blue y", converse "There is a blue y
and a red x" (same meaning, reversed order of mention).  Five colour pairs play the role of E33's five relations.
Same nouns, splits, prompt (4 demonstrations), part codes (whitened 16-PC of "a x" / "there is a x here"), ridge
(penalty by validation nouns) and metric as E33: a probe is fitted on CANONICAL sentences of fitting nouns to
return the asked noun's code; role transfer = share of CONVERSE sentences of new nouns whose prediction is nearer
the asked noun's code than the other noun's (chance 0.5; 0 = word order).  Roles are defined by meaning:
c1 = the noun bound to the first colour of the pair (red), c2 = the noun bound to the second colour (blue).
States (final token unless noted):
  qa      "Which object is red?"  at "Answer: the"  (asked = c1)
  qb      "Which object is blue?" at "Answer: the"  (asked = c2)
  period  the "." that ends the sentence inside the prompt, BEFORE any question
  mean    bare sentence, token average (BOS excluded)
Targets: c1, c2.  Behavior: the model's own answer (answer noun more probable than the other noun, E9 'choice')
on canonical and converse sentences of new nouns.  Output JSON per model/seed: {"rows": [...], "behavior": [...]}.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e9_behavior import fmt, N_DEMO, seq_logprob
from e11_tokens import tok_at

OUTDIR = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e40"
COLORS = {"redblue": ("red", "blue"), "greenyellow": ("green", "yellow"), "blackwhite": ("black", "white"),
          "orangepurple": ("orange", "purple"), "pinkbrown": ("pink", "brown")}
ALPHAS = np.logspace(-3, 9, 25)


def art(c):
    return "an" if c[0] in "aeiou" else "a"


def make_pairs(c1, c2):
    """(canonical, converse) templates; x carries colour c1, y carries colour c2."""
    return (f"There is {art(c1)} {c1} {{x}} and {art(c2)} {c2} {{y}}", f"There is {art(c2)} {c2} {{y}} and {art(c1)} {c1} {{x}}")


def item(c1, c2, x, y, phr, q):
    """x is bound to c1, y to c2.  q=0 asks for c1 (answer x), q=1 asks for c2 (answer y).  -> (sentence, question, answer, other)"""
    sent = make_pairs(c1, c2)[phr].format(x=x, y=y)
    ans, oth = (x, y) if q == 0 else (y, x)
    return sent, f"Which object is {(c1, c2)[q]}?", ans, oth


def layer_list(L):
    return sorted({L // 4, L // 2, (3 * L) // 4, L} | {max(1, round(L * k / 12)) for k in range(1, 13)})


@torch.no_grad()
def embed_bare(tok, mod, texts, layers, dev, bs=64):
    out = {}
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        hs = mod(**b, output_hidden_states=True).hidden_states
        mask = b["attention_mask"].unsqueeze(-1).float()
        if tok.bos_token_id is not None:
            mask[:, 0] *= (b["input_ids"][:, 0] != tok.bos_token_id).float().unsqueeze(-1)
        for li in layers:
            h = hs[li]
            out.setdefault(("mean", li), []).append(((h * mask).sum(1) / mask.sum(1)).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def embed_prompts(tok, mod, prompts, pchar, layers, dev, bs=32, want_period=True):
    """Final-token state and state at the character pchar[i] (the period) of each prompt."""
    out = {}
    for s in range(0, len(prompts), bs):
        b = tok(prompts[s:s + bs], padding=True, return_tensors="pt", return_offsets_mapping=True)
        offs = b.pop("offset_mapping").tolist()
        b = b.to(dev)
        hs = mod(**b, output_hidden_states=True).hidden_states
        last = b["attention_mask"].sum(1) - 1
        ar = torch.arange(len(last))
        pt = torch.tensor([tok_at(o, pc + 1) for o, pc in zip(offs, pchar[s:s + bs])]) if want_period else None
        for li in layers:
            out.setdefault(("final", li), []).append(hs[li][ar, last].float().cpu())
            if want_period:
                out.setdefault(("period", li), []).append(hs[li][ar, pt].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def fit_state(H, F, V, Rc, Rv, targets, dev):
    """H: (N, D) states.  targets: name -> dict(code -> (T_fit, T_val, T_can_tgt, T_can_oth, T_conv_tgt, T_conv_oth)).
    One SVD of the centred fitting states serves every target and code."""
    X = torch.tensor(H[F], device=dev)
    mu = X.mean(0)
    U, s, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    Z = {k: ((torch.tensor(H[r], device=dev) - mu) @ Vt.T) for k, r in (("val", V), ("can", Rc), ("conv", Rv))}
    res = {}
    for tname, codes in targets.items():
        for cname, (Tf, Tv, Tct, Tco, Tvt, Tvo) in codes.items():
            Tf, Tv, Tct, Tco, Tvt, Tvo = [torch.tensor(a, device=dev, dtype=torch.float32) for a in (Tf, Tv, Tct, Tco, Tvt, Tvo)]
            ym = Tf.mean(0)
            UtY = U.T @ (Tf - ym)
            best = None
            for a in ALPHAS:
                W = (s / (s ** 2 + a))[:, None] * UtY
                sc = r2(Z["val"] @ W + ym, Tv)
                if best is None or sc > best[0]:
                    best = (sc, a, W)
            sc, a, W = best
            row = dict(val_rec=sc, alpha=float(a))
            for k, (tg, ot) in (("can", (Tct, Tco)), ("conv", (Tvt, Tvo))):
                p = Z[k] @ W + ym
                row[f"rec_{k}"] = r2(p, tg)
                row[f"role_{k}"] = float((((p - tg) ** 2).sum(1) < ((p - ot) ** 2).sum(1)).float().mean())
            res[(tname, cname)] = row
    return res


def run(name, seed, dev, small=False, outfn=None):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
    tok.padding_side = "right"
    L = mod.config.num_hidden_layers
    layers = layer_list(L)
    codes_raw = {"phrase": embed_bare(tok, mod, [f"a {x}" for x in nouns], layers, dev), "carrier": embed_bare(tok, mod, [f"there is a {x} here" for x in nouns], layers, dev)}
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    Tcodes = {}
    for cname, d in codes_raw.items():
        for li in layers:
            P = d[("mean", li)]
            pmu = P[pca].mean(0)
            _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
            Tcodes[(cname, li)] = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
    nfit, nval, nfr = (3000, 600, 1000) if not small else (300, 100, 100)
    res, beh, bjobs = [], [], []
    cols = list(COLORS.items())[:2] if small else list(COLORS.items())
    for rel, (c1, c2) in cols:
        canon, conv = make_pairs(c1, c2)
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp, frp = sample(fit_n, 3000), sample(val_n, 600), sample(fr_n, 1000)
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = []
        for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)]):
            s_, qq_, a_, _ = item(c1, c2, dn[2 * k], dn[2 * k + 1], p, q)
            demo.append(fmt(s_, qq_) + f" {a_}\n\n")
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        if small:
            fitp, valp, frp = fitp[:nfit], valp[:nval], frp[:nfr]
        texts, meta = [], []
        for split, prs in (("fit", fitp), ("val", valp), ("fresh", frp)):
            for i, (x, y) in enumerate(prs):
                ph = i % 2
                texts.append((canon if ph == 0 else conv).format(x=x, y=y)); meta.append((split, ph, x, y))
        qs = [f"Which object is {c1}?", f"Which object is {c2}?"]
        pchar = [len(head) + len("Sentence: ") + len(t) for t in texts]
        pf = [head + fmt(t, qs[0]) for t in texts]
        assert all(p[c] == "." for p, c in zip(pf, pchar))
        H = embed_prompts(tok, mod, pf, pchar, layers, dev)
        Hg = embed_prompts(tok, mod, [head + fmt(t, qs[1]) for t in texts], pchar, layers, dev, want_period=False)
        Hb = embed_bare(tok, mod, texts, layers, dev)
        # behavior (E9 'choice') on new-noun pairs: both phrasings x both questions, same head
        for p_ in (0, 1):
            for q_ in (0, 1):
                its = [item(c1, c2, x, y, p_, q_) for x, y in frp[:200 if not small else 40]]
                prm = [head + fmt(s_, qq_) for s_, qq_, _, _ in its]
                bjobs.append((rel, p_, q_, prm, [" " + a_ for _, _, a_, _ in its], [" " + o_ for _, _, _, o_ in its]))
        states = {}
        for li in layers:
            states[("qa", li)] = H[("final", li)]; states[("period", li)] = H[("period", li)]
            states[("qb", li)] = Hg[("final", li)]; states[("mean", li)] = Hb[("mean", li)]
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        t1 = [m[2] for m in meta]; t2 = [m[3] for m in meta]
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        Fc, Vc, Rc, Rv = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0), sel(sp == "fresh", ph == 0), sel(sp == "fresh", ph == 1)
        g = lambda Lst, rows: [Lst[i] for i in rows]
        Fa, Va = sel(sp == "fit"), sel(sp == "val")  # both phrasings: the "sem" fit of E11/E33, for its layer rule
        for (st, li), Hs in states.items():
            targets = {}; targets_s = {}
            for tname, tg, ot in (("c1", t1, t2), ("c2", t2, t1)):
                targets[tname] = {}; targets_s[tname] = {}
                for cname in ("phrase", "carrier"):
                    T = Tcodes[(cname, li)]
                    ix = lambda rows, lst: T[[idx[x] for x in g(lst, rows)]]
                    targets[tname][cname] = (ix(Fc, tg), ix(Vc, tg), ix(Rc, tg), ix(Rc, ot), ix(Rv, tg), ix(Rv, ot))
                    targets_s[tname][cname] = (ix(Fa, tg), ix(Va, tg), ix(Rc, tg), ix(Rc, ot), ix(Rv, tg), ix(Rv, ot))
            r = fit_state(Hs, Fc, Vc, Rc, Rv, targets, dev)
            rs = fit_state(Hs, Fa, Va, Rc, Rv, targets_s, dev)
            for (tname, cname), row in r.items():
                row["sem_val_rec"] = rs[(tname, cname)]["val_rec"]
                res.append(dict(model=name, seed=seed, rel=rel, state=st, layer=li, nlayers=L, code=cname, target=tname, **row))
        show = {f"{r['state']}-{r['target']}": round(r["role_conv"], 2) for r in res if r["rel"] == rel and r["layer"] == layers[-3] and r["code"] == "phrase"}
        print(name, seed, rel, "layer", layers[-3], show, flush=True)
        if outfn:
            with open(outfn + ".part", "w", encoding="utf-8") as f:
                json.dump(dict(rows=res, behavior=beh), f)
    # behavior needs the LM head: E.load gives the base model, so reload as a causal LM after freeing it
    import gc; mod.to("cpu"); del mod; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    repo = name.partition("@")[0]
    lm = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16).to(dev).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for rel, p_, q_, prm, ans, oth in bjobs:
        la = seq_logprob(tok, lm, prm, ans, dev)
        lo = seq_logprob(tok, lm, prm, oth, dev)
        beh.append(dict(model=name, seed=seed, rel=rel, phrasing=["canonical", "converse"][p_], question=["qa", "qb"][q_],
                        choice=float((la > lo).mean()), margin=float((la - lo).mean())))
    print(name, seed, "behavior", {f"{b['rel'][:5]}-{b['phrasing'][:3]}-{b['question']}": round(b["choice"], 2) for b in beh}, flush=True)
    return res, beh


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r, b = run(name, seed, dev, a.smoke, fn)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(dict(rows=r, behavior=b), f)
