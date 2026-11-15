"""E21b: which subspace of the question state carries the answer?

E21 found: whole-state patches from a source with the opposite figure flip the answer at late layers,
while patching or ablating the 16-dimensional readout subspace leaves it alone.  Three explanations:
a bug in the patch, redundancy (the noun's identity is written in many more than 16 dimensions), or a
readable copy the model does not use.  This script separates them.

Check:      after the patch, the readout must return the source's figure (patch_rec, patch_rec_src).
Sufficiency: patch only a subspace from the source, for
  readout_K  row space of the figure readout with K part-code components (16, 64, 128)
  diff_k     top k directions of h_source - h_target on fitting nouns (the directions the swap moves)
  pc_k       top k principal components of the fitting states
  random_k   random k dimensions within the top principal components
Necessity:  patch everything except a subspace (allbut_readout_16, allbut_diff_64): if the answer still
            flips, the subspace was not needed.
Control:    diff_64 with the same-figure source, whose answer is the target's own.
Sources are "a x under a y" for a target "a x on a y": same order of mention, opposite figure.  All test
nouns are held out from the readout and from the difference directions.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO
from e11_tokens import embed_query
from e21_intervene import PICK, forward, N_FIT, N_VAL, N_TEST

NDIFF = 600


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pick_layer, cname = PICK[name]
    codes = E.embed(tok, mod, [(f"a {x}" if cname == "phrase" else f"there is a {x} here") for x in nouns], dev, name)
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    first = lambda x: tok(" " + x, add_special_tokens=False)["input_ids"][0]
    res = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp, tp = sample(fit_n, N_FIT), sample(val_n, N_VAL), sample(fr_n, N_TEST * 2)
        tp = [p for p in tp if first(p[0]) != first(p[1])][:N_TEST]
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        q = QUESTIONS[rel][0]
        P = lambda t: head + fmt(t, q)
        Hfit = embed_query(tok, mod, [P(canon.format(x=x, y=y)) for x, y in fitp], dev)
        Hval = embed_query(tok, mod, [P(canon.format(x=x, y=y)) for x, y in valp], dev)
        dp = fitp[:NDIFF]
        Hdt = embed_query(tok, mod, [P(canon.format(x=x, y=y)) for x, y in dp], dev)
        Hds = embed_query(tok, mod, [P(conv.format(x=y, y=x)) for x, y in dp], dev)
        ptarget = [P(canon.format(x=x, y=y)) for x, y in tp]
        psrc = [P(conv.format(x=y, y=x)) for x, y in tp]        # "a x under a y": figure y
        psame = [P(conv.format(x=x, y=y)) for x, y in tp]       # "a y under a x": figure x
        Ht, Hs, Hsf = embed_query(tok, mod, ptarget, dev), embed_query(tok, mod, psrc, dev), embed_query(tok, mod, psame, dev)
        ax = np.array([first(x) for x, _ in tp]); ay = np.array([first(y) for _, y in tp])
        m0 = forward(tok, mod, ptarget, dev, ax, ay)
        ms = forward(tok, mod, psrc, dev, ax, ay)
        for (_, li) in Hfit:
            Pc = codes[f"L{li}"]
            pmu = Pc[pca].mean(0)
            _, sv, Vt = np.linalg.svd(Pc[pca] - pmu, full_matrices=False)
            F_, V_ = Hfit[("query", li)], Hval[("query", li)]
            mu = F_.mean(0)
            _, _, Vf = np.linalg.svd(F_ - mu, full_matrices=False)
            D = np.asarray(Hds[("query", li)] - Hdt[("query", li)], dtype=np.float64)
            _, _, Vd = np.linalg.svd(D, full_matrices=False)
            h0, hs, hsf = Ht[("query", li)], Hs[("query", li)], Hsf[("query", li)]
            g = np.random.default_rng(1000 + seed)
            nb = Vf.shape[0]
            subs, regs = {}, {}
            for K in (16, 64, 128):
                Tc = (Pc - pmu) @ Vt[:K].T / (sv[:K] / np.sqrt(len(pca)))
                reg = E.RidgePath(F_ - mu, Tc[[idx[x] for x, _ in fitp]]).select(V_ - mu, Tc[[idx[x] for x, _ in valp]])
                subs[f"readout_{K}"] = np.linalg.svd(reg.W, full_matrices=False)[0]
                regs[K] = (reg, Tc)
            for k in (16, 64, 256):
                subs[f"diff_{k}"] = Vd[:k].T
                subs[f"pc_{k}"] = Vf[:k].T
                subs[f"random_{k}"] = np.linalg.qr(Vf[:nb].T @ g.normal(size=(nb, k)))[0]
            base = dict(model=name, seed=seed, rel=rel, layer=li, picked=li == pick_layer, code=cname, n=len(tp), orig_acc=float((m0 > 0).mean()),
                        src_flip=float((ms < 0).mean()), diff_sv=(np.linalg.svd(D, compute_uv=False)[:64] ** 2).round(2).tolist())
            reg16, T16 = regs[16]
            read = lambda h: (h - mu) @ reg16.W + reg16.ym
            Tx, Ty = T16[[idx[x] for x, _ in tp]], T16[[idx[y] for _, y in tp]]
            r2 = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())
            conds = [("whole", hs, hs)]
            for kn, B in subs.items():
                conds.append((kn, h0 + (hs - h0) @ B @ B.T, hs))
            for kn in ("readout_16", "diff_64"):
                B = subs[kn]
                conds.append((f"allbut_{kn}", hs + (h0 - hs) @ B @ B.T, hs))
            for sname, hn, _ in conds:
                mp = forward(tok, mod, ptarget, dev, ax, ay, patch=np.asarray(hn, dtype=np.float32), li=li)
                pr = read(np.asarray(hn))
                res.append(dict(base, kind=sname, flip=float((mp < 0).mean()), margin=float(mp.mean()),
                                patch_rec=r2(pr, Tx), patch_rec_src=r2(pr, Ty), shift=float(np.linalg.norm(hn - h0, axis=1).mean()), norm=float(np.linalg.norm(h0, axis=1).mean())))
            B = subs["diff_64"]
            mp = forward(tok, mod, ptarget, dev, ax, ay, patch=np.asarray(h0 + (hsf - h0) @ B @ B.T, dtype=np.float32), li=li)
            res.append(dict(base, kind="samefig_diff_64", flip=float((mp < 0).mean()), margin=float(mp.mean())))
            print(name, rel, li, "orig", round(base["orig_acc"], 3), {r["kind"]: round(r["flip"], 2) for r in res[-(len(conds) + 1):]}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e21b")
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)
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
