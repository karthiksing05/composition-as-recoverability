"""E23: parts held out from the model itself.

Pretrained models saw every noun in pretraining, so "held-out nouns" are held out from the readout only.
Here the test nouns are invented words, written with the same part-code pipeline, and we compare with real
held-out nouns under the identical code:
  answers    the model's own choice on role questions (the E9 prompt), with 0.5 for word order
  transfer   the E11 figure readout at the question state, fitted on canonical sentences of real nouns,
             scored on converse sentences of test nouns
  kappa      token-averaged state recovery of test nouns' codes with both references
Both nouns are in the sentence, so copying tokens cannot exceed 0.5: the role decides the answer.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS, readout
from e9_behavior import QUESTIONS, item, fmt, N_DEMO, seq_logprob
from e11_tokens import embed_query
from e21_intervene import PICK

NONCE = """blicket daxen wuggle fepton gazzer tulver morbit zandle pruven kolber nifter vosket
drambel sprock glimber throndle skivet bromple quindle jarvet plondik trazzle vintle crubbet
mindrel fosker yaldin hepsol wimbet tarkle rizzet lompet gorvel snaffin dretch pobbin kesple
nurdle zomber frindle hoskin varbit clumber shondle teffin brazzik wendrel gurple moskin
pindle ravvet solbin tronkle yembel zivvet dorple felkin grombit hiskel jandrel kumpet
lorvin nesket ospel pelvin quarbit rundle stagget tolpin ulvet vospel wintle yarkel zembit
blorven crostik dalver flintek grovvet hemble ilvet jospin korble lumpet marvin nolket
pirvel rostik selvin trumbel unkle velpin wozzel yintle zarpet bofkin cregget dulvin
fombler glossik harbel irkel jempet kolvin lazzik""".split()


def run(name, seed, dev, nonce_n=100):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    REAL = {"marvin", "unkle", "pelvin", "nurdle", "clumber", "hoskin", "rundle", "frindle", "wintle", "dretch", "moskin", "sprock"}
    non = [w for w in dict.fromkeys(NONCE) if w not in set(nouns) and w not in REAL]
    rng.shuffle(non); non = non[:nonce_n]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    every = nouns + non
    idx = {x: i for i, x in enumerate(every)}
    codes = {"phrase": E.embed(tok, mod, [f"a {x}" for x in every], dev, name),
             "carrier": E.embed(tok, mod, [f"there is a {x} here" for x in every], dev, name)}
    pca = list(range(E.N_FIT))
    ntok = {x: len(tok(" " + x, add_special_tokens=False)["input_ids"]) for x in every}
    meta0 = dict(tokens_real=float(np.mean([ntok[x] for x in fr_n])), tokens_nonce=float(np.mean([ntok[x] for x in non])))
    res = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp = sample(fit_n, 1500), sample(val_n, 300)
        tests = {"real": sample(fr_n, 400), "nonce": sample(non, 400)}
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        q = QUESTIONS[rel][0]
        # rows: fit and validation canonical, each test pair in both phrasings
        texts, meta = [], []
        for split, prs, phs in [("fit", fitp, (0,)), ("val", valp, (0,))] + [(k, v, (0, 1)) for k, v in tests.items()]:
            for x, y in prs:
                for ph in phs:
                    texts.append((canon if ph == 0 else conv).format(x=x, y=y)); meta.append((split, ph, x, y))
        Q = embed_query(tok, mod, [head + fmt(t, q) for t in texts], dev)
        A = E.embed(tok, mod, texts, dev, name)
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        F, V = sel(sp == "fit"), sel(sp == "val")
        G = lambda L, r: [L[i] for i in r]
        row0 = dict(model=name, seed=seed, rel=rel, **meta0)
        for tname in tests:
            R = sel(sp == tname)
            pr = [head + fmt(texts[i], q) for i in R]
            lf = seq_logprob(tok, mod, pr, [" " + fig[i] for i in R], dev)
            lg = seq_logprob(tok, mod, pr, [" " + gnd[i] for i in R], dev)
            ok = lf > lg
            beh = dict(answers=float(ok.mean()), answers_can=float(ok[ph[R] == 0].mean()), answers_conv=float(ok[ph[R] == 1].mean()))
            Rc, Rv = sel(sp == tname, ph == 0), sel(sp == tname, ph == 1)
            for (_, li), Hq in Q.items():
                layer = f"L{li}"
                for cname, Pc in codes.items():
                    P = Pc[layer]
                    pmu = P[pca].mean(0)
                    _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                    T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                    xf = readout(Hq, T, idx, F, G(fig, F), V, G(fig, V), {"val": (V, G(fig, V), G(gnd, V)), "conv": (Rv, G(fig, Rv), G(gnd, Rv)), "can": (Rc, G(fig, Rc), G(gnd, Rc))})
                    # token-averaged state, canonical sentences: both roles and both references
                    first, second = lambda rr: G(fig, rr), lambda rr: G(gnd, rr)
                    a_, b_ = P[[idx[x] for x in fig]], P[[idx[x] for x in gnd]]
                    kap = {}
                    for sname, Hs in (("model", A[layer]), ("ref_concat", np.c_[a_, b_]), ("ref_sum", a_ + b_)):
                        r1 = readout(Hs, T, idx, F, first(F), V, first(V), {"val": (V, first(V), second(V)), "test": (Rc, first(Rc), second(Rc))})
                        r2 = readout(Hs, T, idx, F, second(F), V, second(V), {"val": (V, second(V), first(V)), "test": (Rc, second(Rc), first(Rc))})
                        kap[f"{sname}_rec"] = (r1["rec_test"] + r2["rec_test"]) / 2
                        kap[f"{sname}_val"] = (r1["rec_val"] + r2["rec_val"]) / 2
                    res.append(dict(row0, test=tname, layer=layer, code=cname, picked=(li == PICK[name][0] and cname == PICK[name][1]), **beh,
                                    xfer_conv=xf["role_conv"], xfer_can=xf["role_can"], xfer_val=xf["rec_val"], **kap))
            print(name, rel, tname, beh, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e23")
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
