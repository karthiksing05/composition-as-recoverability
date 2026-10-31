"""Case studies for the large pretrained models (seed 0 splits, the E7 and E8 protocols).

text    Qwen3-8B, layer and code selected in E7 (L9, phrase), relation on/under.  Readouts of
        the figure fitted on canonical sentences, and on both phrasings, predict a code for
        fresh sentences in each phrasing; we report the nearest of the 100 fresh nouns.
images  SigLIP so400m and its random-weight copy at their selected layer and code (L6, center),
        relation beside.  Readouts of each role predict a code for fresh composites; we report
        the nearest of the 300 fresh photographs.
Writes results/case/text.json and results/case/images.json."""
import io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V

E.ALPHAS = np.logspace(-3, 9, 25)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results/case", exist_ok=True)


def whiten(P, rows):
    mu = P[rows].mean(0)
    _, sv, Vt = np.linalg.svd(P[rows] - mu, full_matrices=False)
    return (P - mu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(rows)))


def readout(Hf, Yf, Hv, Yv):
    mu = Hf.mean(0)
    reg = E.RidgePath(Hf - mu, Yf).select(Hv - mu, Yv)
    return lambda H: reg.predict(H - mu)


def nearest(pred, T, cand):
    d = ((pred[:, None, :] - T[cand][None]) ** 2).sum(-1)
    return [cand[i] for i in d.argmin(1)]


def sample(rng, ns, k):
    prs = [(x, y) for x in ns for y in ns if x != y]
    return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]


# ---------------- text ----------------
if not os.path.exists("results/case/text.json"):  # the text half is cached
    name, LAYER = "Qwen/Qwen3-8B", "L9"
    rng = np.random.default_rng(0)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL]
    fr_n = nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    idx = {x: i for i, x in enumerate(nouns)}
    tok, mod = E.load(name, dev)
    T = whiten(E.embed(tok, mod, [f"a {x}" for x in nouns], dev, name)[LAYER], [idx[x] for x in fit_n])
    canon, conv = "a {x} on a {y}", "a {y} under a {x}"
    fitp, valp, frp = sample(rng, fit_n, 3000), sample(rng, val_n, 600), sample(rng, fr_n, 1000)
    emb = lambda prs, t: E.embed(tok, mod, [t.format(x=x, y=y) for x, y in prs], dev, name)[LAYER]
    Hfc, Hfv, Hvc, Hvv, Hrc, Hrv = emb(fitp, canon), emb(fitp, conv), emb(valp, canon), emb(valp, conv), emb(frp, canon), emb(frp, conv)
    fig = lambda prs: T[[idx[x] for x, _ in prs]]
    gnd = lambda prs: T[[idx[y] for _, y in prs]]
    f_can = readout(Hfc, fig(fitp), Hvc, fig(valp))
    f_both = readout(np.r_[Hfc, Hfv], np.r_[fig(fitp), fig(fitp)], np.r_[Hvc, Hvv], np.r_[fig(valp), fig(valp)])
    cand = [idx[x] for x in fr_n]


    def role_acc(pred, prs):
        return float((((pred - fig(prs)) ** 2).sum(1) < ((pred - gnd(prs)) ** 2).sum(1)).mean())


    res = {"model": name, "layer": LAYER,
           "acc": {"canon_fit_on_canon": role_acc(f_can(Hrc), frp), "canon_fit_on_conv": role_acc(f_can(Hrv), frp),
                   "both_fit_on_conv": role_acc(f_both(Hrv), frp)}, "examples": []}
    n_can, n_conv, n_both = nearest(f_can(Hrc[:8]), T, cand), nearest(f_can(Hrv[:8]), T, cand), nearest(f_both(Hrv[:8]), T, cand)
    for k, (x, y) in enumerate(frp[:8]):
        res["examples"].append(dict(figure=x, ground=y, canon=canon.format(x=x, y=y), conv=conv.format(x=x, y=y),
                                    pred_canon=nouns[n_can[k]], pred_conv=nouns[n_conv[k]], pred_both_conv=nouns[n_both[k]]))
    json.dump(res, open("results/case/text.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    del mod; torch.cuda.empty_cache()

# ---------------- images ----------------
ims, labels = V.load_images()
rng = np.random.default_rng(0)
cls = rng.permutation(200)
splits, s0 = {}, 0
for k, n, per in zip(("fit", "val", "fresh"), V.N_CLS, V.PER_CLS):
    ids = []
    for c in cls[s0:s0 + n]:
        ids += list(rng.choice(np.where(labels == c)[0], per, replace=False))
    splits[k] = [int(i) for i in ids]; s0 += n
used = splits["fit"] + splits["val"] + splits["fresh"]
pil = {i: Image.open(io.BytesIO(ims[i])).convert("RGB") for i in used}
fit_pairs, val_pairs, fresh_pairs = sample(rng, splits["fit"], 3000), sample(rng, splits["val"], 600), sample(rng, splits["fresh"], 1000)
out = {"pairs": [list(p) for p in fresh_pairs[:6]], "models": {}}
for mname in ("siglip-so400m", "rand-siglip-so400m"):
    m = V.load(mname, dev)
    codes = V.embed(m, [V.canvas([(pil[i], 112, (56, 56))]) for i in used], dev)["L6"]
    jdx = {x: j for j, x in enumerate(used)}
    Tm = whiten(codes, list(range(len(splits["fit"]))))
    allp = fit_pairs + val_pairs + fresh_pairs
    H = V.embed(m, [V.compose("beside", pil[x], pil[y]) for x, y in allp], dev)["L6"]
    nf, nv = len(fit_pairs), len(val_pairs)
    Hf, Hv, Hr = H[:nf], H[nf:nf + nv], H[nf + nv:]
    fc = [jdx[i] for i in splits["fresh"]]
    rec = {}
    for role in (0, 1):
        tgt = lambda prs: Tm[[jdx[p[role]] for p in prs]]
        f = readout(Hf, tgt(fit_pairs), Hv, tgt(val_pairs))
        pred = f(Hr)
        nn = nearest(pred, Tm, fc)
        rec[f"id_role{role + 1}"] = float(np.mean([used[j] == p[role] for j, p in zip(nn, fresh_pairs)]))
        rec[f"pred_role{role + 1}"] = [used[j] for j in nn[:6]]
    out["models"][mname] = rec
    print(mname, {k: v for k, v in rec.items() if k.startswith("id")}, flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open("results/case/images.json", "w"), indent=1)
