"""E19: do nonlinear readouts find the semantic figure in the states of the sentence alone?

E7 and E11 read the figure with ridge readouts: fitted on canonical sentences, they transfer to
converse sentences at 0.00 from the states of the sentence (the token average and the two noun
states), and at 0.73-0.98 from the state at the end of a question about the figure.  A richer
readout class might find a figure the linear class misses.  Here the same rows, splits and part
codes as E11 (identical random draws) are read by three classes, each fitted on canonical
sentences of fitting nouns with its hyperparameters chosen on canonical validation sentences:
  ridge   the declared linear class (E11's readout)
  kernel  kernel ridge regression with a Gaussian kernel on the top 256 principal components
          of the state (bandwidth a multiple of the median distance, penalty, both on validation)
  mlp     a two-layer perceptron (512 tanh units) on the same components, weight decay on validation
Scored by role assignment on fresh nouns: canonical (can) and converse (conv) sentences.  The
*_oracle_* fields give the best score over every hyperparameter in the grids (selected on the test
rows, an upper bound for the class at these hyperparameters).  A
readout of word order scores conv ~ 0, a readout of meaning ~ 1.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO
from e11_tokens import spans, embed_tokens, embed_query

E11_PICK = {"microsoft/phi-4": (30, "phrase"), "Qwen/Qwen3-8B": (27, "carrier"), "allenai/OLMo-2-1124-13B": (30, "phrase")}
N_PC = 256


def role_acc(p, y, yo):
    return float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())


def r2(p, y):
    return E.r2(p, y)


def readouts(X, T_of, Fc, Vc, tests):
    """X: all rows; T_of(rows, which) -> targets; returns {class: {test: role acc}}."""
    mu = X[Fc].mean(0)
    Xc = X - mu
    out = {}
    reg = E.RidgePath(Xc[Fc], T_of(Fc, 0)).select(Xc[Vc], T_of(Vc, 0))
    out["ridge"] = {k: role_acc(reg.predict(Xc[r]), T_of(r, 0), T_of(r, 1)) for k, r in tests.items()}
    # top principal components of the fitting rows, scaled to unit variance
    _, sv, Vt = np.linalg.svd(Xc[Fc], full_matrices=False)
    Z = Xc @ Vt[:N_PC].T / (sv[:N_PC] / np.sqrt(len(Fc)))
    Zf, Yf = torch.tensor(Z[Fc], dtype=torch.float64), torch.tensor(T_of(Fc, 0), dtype=torch.float64)
    d2 = torch.cdist(Zf, Zf) ** 2
    med = float(d2[d2 > 0].median())
    best, orc = None, {k: 0.0 for k in tests}
    for bw in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
        K = torch.exp(-d2 / (bw * med))
        ev, U = torch.linalg.eigh(K)
        UtY = U.T @ Yf
        Zv = torch.tensor(Z[Vc], dtype=torch.float64)
        Kv = torch.exp(-torch.cdist(Zv, Zf) ** 2 / (bw * med))
        Kt = {k: torch.exp(-torch.cdist(torch.tensor(Z[r], dtype=torch.float64), Zf) ** 2 / (bw * med)) for k, r in tests.items()}
        for a in np.logspace(-7, 3, 21):
            A = U @ (UtY / (ev + a)[:, None])
            sc = r2((Kv @ A).numpy(), T_of(Vc, 0))
            for k, r in tests.items():
                orc[k] = max(orc[k], role_acc((Kt[k] @ A).numpy(), T_of(r, 0), T_of(r, 1)))
            if best is None or sc > best[0]:
                best = (sc, bw, a, A)
    _, bw, a, A = best
    kp = lambda r: (torch.exp(-torch.cdist(torch.tensor(Z[r], dtype=torch.float64), Zf) ** 2 / (bw * med)) @ A).numpy()
    out["kernel"] = {k: role_acc(kp(r), T_of(r, 0), T_of(r, 1)) for k, r in tests.items()}
    out["kernel_hp"] = {"bw": bw, "alpha": float(a), "val_r2": float(best[0])}
    out["kernel_oracle"] = orc
    # two-layer perceptron, full-batch Adam, weight decay chosen on validation
    best, orc = None, {k: 0.0 for k in tests}
    for wd in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(N_PC, 512), torch.nn.Tanh(), torch.nn.Linear(512, Yf.shape[1])).double()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        Zv = torch.tensor(Z[Vc], dtype=torch.float64)
        bsc, bstate = -np.inf, None
        for ep in range(2000):
            opt.zero_grad()
            loss = ((net(Zf) - Yf) ** 2).mean()
            loss.backward(); opt.step()
            if ep % 50 == 49:
                with torch.no_grad():
                    sc = r2(net(Zv).numpy(), T_of(Vc, 0))
                if sc > bsc:
                    bsc, bstate = sc, {k: v.clone() for k, v in net.state_dict().items()}
        if best is None or bsc > best[0]:
            best = (bsc, wd, bstate)
        net.load_state_dict(bstate)
        with torch.no_grad():
            for k, r in tests.items():
                orc[k] = max(orc[k], role_acc(net(torch.tensor(Z[r], dtype=torch.float64)).numpy(), T_of(r, 0), T_of(r, 1)))
    net.load_state_dict(best[2])
    with torch.no_grad():
        mp = lambda r: net(torch.tensor(Z[r], dtype=torch.float64)).numpy()
        out["mlp"] = {k: role_acc(mp(r), T_of(r, 0), T_of(r, 1)) for k, r in tests.items()}
    out["mlp_hp"] = {"wd": best[1], "val_r2": float(best[0])}
    out["mlp_oracle"] = orc
    return out


def run(name, seed, dev):
    rng = np.random.default_rng(seed)  # the same draws as e11_tokens.run
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
    tok.padding_side = "right"
    li, cname = E11_PICK[name]
    P = E.embed(tok, mod, [(f"a {x}" if cname == "phrase" else f"there is a {x} here") for x in nouns], dev, name)[f"L{li}"]
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    pmu = P[pca].mean(0)
    _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
    T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
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
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        H = embed_tokens(tok, mod, texts, sps, dev)
        H.update(embed_query(tok, mod, [head + fmt(t, QUESTIONS[rel][0]) for t in texts], dev))
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        Fc, Vc = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0)
        tests = {"can": sel(sp == "fresh", ph == 0), "conv": sel(sp == "fresh", ph == 1)}
        T_of = lambda rows, which: T[[idx[(fig if which == 0 else gnd)[i]] for i in rows]]
        for state in ("mean", "nouns", "query"):
            r = readouts(H[(state, li)], T_of, Fc, Vc, tests)
            res.append(dict(model=name, seed=seed, rel=rel, state=state, layer=f"L{li}", code=cname, **{
                f"{c}_{k}": v for c in ("ridge", "kernel", "mlp") for k, v in r[c].items()},
                kernel_hp=r["kernel_hp"], mlp_hp=r["mlp_hp"],
                **{f"{c}_oracle_{k}": v for c in ("kernel", "mlp") for k, v in r[f"{c}_oracle"].items()}))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e19")
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
            r = run(name, seed, dev)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
