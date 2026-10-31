"""E7: are the recovered roles word positions or semantic roles?

Each relation comes with a converse paraphrase that swaps the order of mention and keeps the
meaning: "a {x} on a {y}" and "a {y} under a {x}" both put x on top.  The semantic role
("figure": the noun on top, on the left, inside, behind, or doing the hitting) and the position
(first-mentioned noun) then dissociate.  Readouts from the declared class (ridge on the composite
state, targets the whitened 16-PC part code, penalty chosen on validation nouns) are fitted on
fitting nouns and scored on fresh nouns:
  pos     target = first-mentioned noun, fitted on both phrasings
  sem     target = figure, fitted on both phrasings
  xfer    target = figure, fitted on the canonical phrasing only, scored on the converse
For each: R^2 on fresh nouns (rec_*) and role assignment (role_*: the prediction is closer to the
target noun's code than to the other noun's).  A readout of position alone scores xfer role ~ 0
(it returns the ground); a readout of semantic roles scores ~ 1.  References built from the
model's own part codes: concatenation in order of mention ("ref_pos") and in semantic order
("ref_sem"), so the pipeline's ceiling for each target is known.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E

PAIRS = {  # relation: (canonical, converse); x is the figure in both
    "on": ("a {x} on a {y}", "a {y} under a {x}"),
    "left": ("a {x} to the left of a {y}", "a {y} to the right of a {x}"),
    "inside": ("a {x} inside a {y}", "a {y} containing a {x}"),
    "behind": ("a {x} behind a {y}", "a {y} in front of a {x}"),
    "hit": ("a {x} hit a {y}", "a {y} was hit by a {x}"),
}


def readout(Hs, T, idx, fit, fit_t, val, val_t, tests):
    """Fit on rows `fit` with target nouns fit_t, select the penalty on `val`; score each test
    (rows, target nouns, other nouns) by R^2 and role assignment."""
    mu = Hs[fit].mean(0)
    reg = E.RidgePath(Hs[fit] - mu, T[[idx[x] for x in fit_t]]).select(Hs[val] - mu, T[[idx[x] for x in val_t]])
    out = {}
    for k, (rows, tgt, oth) in tests.items():
        p = reg.predict(Hs[rows] - mu)
        y, yo = T[[idx[x] for x in tgt]], T[[idx[x] for x in oth]]
        out[f"rec_{k}"] = E.r2(p, y)
        out[f"role_{k}"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
    return out


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
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
        # every pair appears in one phrasing; half canonical, half converse
        texts, meta = [], []  # meta: (split, phrasing, figure, ground)
        for split, prs in (("fit", fitp), ("val", valp), ("fresh", frp)):
            for i, (x, y) in enumerate(prs):
                ph = i % 2
                texts.append((canon if ph == 0 else conv).format(x=x, y=y))
                meta.append((split, ph, x, y))
        # the same fresh pairs in the other phrasing, for a paired comparison
        for i, (x, y) in enumerate(frp):
            ph = 1 - i % 2
            texts.append((canon if ph == 0 else conv).format(x=x, y=y)); meta.append(("fresh2", ph, x, y))
        H = E.embed(tok, mod, texts, dev, name)
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
        first = [f if p == 0 else g for f, g, p in zip(fig, gnd, ph)]
        second = [g if p == 0 else f for f, g, p in zip(fig, gnd, ph)]
        sel = lambda *conds: np.where(np.logical_and.reduce(conds))[0]
        F, V = sel(sp == "fit"), sel(sp == "val")
        Fc, Vc = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0)
        R_all = np.r_[sel(sp == "fresh"), sel(sp == "fresh2")]
        R_conv = np.r_[sel(sp == "fresh", ph == 1), sel(sp == "fresh2", ph == 1)]
        R_can = np.r_[sel(sp == "fresh", ph == 0), sel(sp == "fresh2", ph == 0)]
        g = lambda L, rows: [L[i] for i in rows]
        for layer, Hall in H.items():
            for cname in ("phrase", "carrier"):
                P = codes[cname][layer]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A = P[[idx[x] for x in first]]; B = P[[idx[x] for x in second]]
                Fg = P[[idx[x] for x in fig]]; Gd = P[[idx[x] for x in gnd]]
                systems = {"model": Hall, "ref_pos": np.c_[A, B], "ref_sem": np.c_[Fg, Gd]}
                for sname, Hs in systems.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname)
                    row.update({f"pos_{k}": v for k, v in readout(Hs, T, idx, F, g(first, F), V, g(first, V),
                                {"val": (V, g(first, V), g(second, V)), "fresh": (R_all, g(first, R_all), g(second, R_all))}).items()})
                    row.update({f"sem_{k}": v for k, v in readout(Hs, T, idx, F, g(fig, F), V, g(fig, V),
                                {"val": (V, g(fig, V), g(gnd, V)), "fresh": (R_all, g(fig, R_all), g(gnd, R_all)),
                                 "conv": (R_conv, g(fig, R_conv), g(gnd, R_conv))}).items()})
                    row.update({f"xfer_{k}": v for k, v in readout(Hs, T, idx, Fc, g(fig, Fc), Vc, g(fig, Vc),
                                {"can": (R_can, g(fig, R_can), g(gnd, R_can)),
                                 "conv": (R_conv, g(fig, R_conv), g(gnd, R_conv))}).items()})
                    res.append(row)
                r = res[-3]
                print(name, rel, layer, cname, {k: round(r[k], 3) for k in ("pos_rec_fresh", "sem_rec_fresh", "xfer_role_conv")}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", required=True)
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
