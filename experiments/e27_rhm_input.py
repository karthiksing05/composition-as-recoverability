"""E27: input floor for recoverability on the Random Hierarchy Model (Cagnetta et al. 2024).

For each grammar we sample strings, take the latent symbols at level l (l=1 sits just above the
tokens) as the parts and their positions as the roles, and fit an affine (ridge) readout from the
one-hot string to each role's part code. Readouts fit on strings whose role-i latent is in a fitting
pool of symbols and are scored on strings whose role-i latent is a held-out symbol, as in the paper.
The same pipeline on the concatenation and on the sum of the part codes gives kappa.

Part codes (system-independent descriptions of a symbol):
  phi    : the symbol's place in its parents' rules, P(parent b, slot j | a), centred, top-8 PCs, whitened
  random : an 8-dim Gaussian vector per symbol
Pre-declared bar (from E20): an input readout of at most 0.5 R^2 leaves room for a trained state to rise above it.
"""
import json, os, sys, itertools
import numpy as np

OUT = "results/e27"
os.makedirs(OUT, exist_ok=True)
CODE_DIM = 8


def make_rules(v, s, m, L, rng):
    """rules[l][a] = (m, s) array of level-(l-1) symbols; level L is the root. Unambiguous: tuples distinct across parents."""
    rules = {}
    for l in range(L, 0, -1):
        idx = rng.choice(v ** s, size=m * v, replace=False)
        tup = np.stack([(idx // v ** (s - 1 - j)) % v for j in range(s)], 1)
        rules[l] = tup.reshape(v, m, s)
    return rules


def sample(rules, v, s, m, L, n, rng):
    """Returns latents[l] = (n, s**(L-l)) symbols at level l, for l = 0..L (l=0 are the tokens)."""
    lat = {L: rng.integers(0, v, size=(n, 1))}
    for l in range(L, 0, -1):
        par = lat[l]
        ch = rng.integers(0, m, size=par.shape)
        kids = rules[l][par, ch]  # (n, P, s)
        lat[l - 1] = kids.reshape(n, -1)
    return lat


def phi_codes(rules, v, s, m, L, rng):
    """Code for each symbol at levels 1..L-1: P(parent, slot | symbol) under the grammar's marginals."""
    marg = {L: np.full(v, 1.0 / v)}
    for l in range(L, 0, -1):
        p = np.zeros(v)
        for b in range(v):
            for t in rules[l][b]:
                for c in t:
                    p[c] += marg[l][b] / (m * s)
        marg[l - 1] = p / p.sum()
    codes = {}
    for l in range(1, L):
        F = np.zeros((v, v * s))
        for b in range(v):
            for t in rules[l + 1][b]:
                for j, a in enumerate(t):
                    F[a, b * s + j] += marg[l + 1][b] / m
        F = F / np.maximum(F.sum(1, keepdims=True), 1e-12)
        F = F - F.mean(0)
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        C = U[:, :CODE_DIM] * S[:CODE_DIM]
        codes[l] = (C - C.mean(0)) / (C.std(0) + 1e-9)
    return codes


def onehot(tok, v):
    n, d = tok.shape
    X = np.zeros((n, d * v), np.float32)
    X[np.arange(n)[:, None], np.arange(d)[None] * v + tok] = 1
    return X


def ridge_fit(X, Y, a):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    if X.shape[1] <= X.shape[0]:
        W = np.linalg.solve(Xc.T @ Xc + a * np.eye(X.shape[1]), Xc.T @ Yc)
    else:
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + a * np.eye(X.shape[0]), Yc)
    return lambda Z: (Z - mx) @ W + my


def eps(P, Y):
    return float(((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


ALPHAS = (1e-3, 1e-1, 1, 10, 100, 1000)


def score(feat, code_of, lat_l, pools, role):
    """Ridge from feat to role's part code; alpha by validation symbols; error on held-out symbols and on held-out strings of fitting symbols."""
    sym = lat_l[:, role]
    fit, val, test = (np.isin(sym, p) for p in pools)
    nf = fit.sum()
    # held-out strings of fitting symbols: last 20% of fit rows
    fidx = np.where(fit)[0]
    cut = int(0.8 * len(fidx))
    tr, tr_test = fidx[:cut], fidx[cut:]
    Y = code_of[sym]
    best = min(ALPHAS, key=lambda a: eps(ridge_fit(feat[tr], Y[tr], a)(feat[val]), Y[val]))
    f = ridge_fit(feat[tr], Y[tr], best)
    return eps(f(feat[test]), Y[test]), eps(f(feat[tr_test]), Y[tr_test]), best


def run(v, s, m, L, seed, n):
    rng = np.random.default_rng(seed)
    rules = make_rules(v, s, m, L, rng)
    lat = sample(rules, v, s, m, L, n, rng)
    X = onehot(lat[0], v)
    phi = phi_codes(rules, v, s, m, L, rng)
    out = []
    for l in range(1, L):
        k = s ** (L - l)
        for cname in ("phi", "random"):
            C = phi[l] if cname == "phi" else rng.standard_normal((v, CODE_DIM))
            perm = rng.permutation(v)
            nf, nv = int(0.55 * v), int(0.15 * v)
            pools = (perm[:nf], perm[nf:nf + nv], perm[nf + nv:])
            codes = C[lat[l]]  # (n, k, D)
            feats = {"input": X, "ref_concat": codes.reshape(n, -1), "ref_sum": codes.sum(1)}
            roles = range(k) if k <= 4 else rng.choice(k, 4, replace=False)
            for name, F in feats.items():
                r = [score(F, C, lat[l], pools, int(i)) for i in roles]
                out.append(dict(v=v, s=s, m=m, L=L, level=l, k=k, code=cname, seed=seed, system=name,
                                eps_heldout=float(np.mean([x[0] for x in r])), eps_train_sym=float(np.mean([x[1] for x in r])),
                                alpha=[x[2] for x in r], f=m / v ** (s - 1)))
    return out


if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    grid = [(v, s, m, L) for (s, L) in ((2, 3), (2, 4), (3, 2), (3, 3)) for v in (32, 64, 128) for m in (2, 4, 8)
            if m * v <= v ** s and s ** L * v <= 4096]
    jobs = [(g, seed) for g in grid for seed in range(3)]
    if task < 0:
        print(len(jobs), "jobs"); sys.exit()
    for (v, s, m, L), seed in jobs[task::8]:
        n = 30000
        fn = f"{OUT}/v{v}_s{s}_m{m}_L{L}_seed{seed}.json"
        if os.path.exists(fn):
            continue
        res = run(v, s, m, L, seed, n)
        with open(fn, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        for r in res:
            if r["system"] == "input":
                print(v, s, m, L, seed, "lvl", r["level"], r["code"], "R2 held-out %.3f train-sym %.3f" % (1 - r["eps_heldout"], 1 - r["eps_train_sym"]), flush=True)
