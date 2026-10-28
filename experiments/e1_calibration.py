"""E1: calibration of analytic modes of combination.

For each operator g(x, y) over Gaussian parts in R^d we fit readouts that recover
the role-1 part from the composite, on composites built from a finite vocabulary,
and evaluate them on (a) unseen pairs of vocabulary parts ("familiar") and
(b) pairs of freshly drawn parts ("fresh", from a pool of 2,000 parts independent of V).
Recovery is reported as
  - R2: variance of the part explained by a regression readout (linear / MLP), against
        the population variance of the parts,
  - ID: identification of the part among V candidates by nearest neighbour on
        the regression output (chance 1/V),
  - ROLE: fraction of composites whose role-1 readout is closer to the role-1
        part than to the role-2 part.
Prop. 2 predicts fresh R2 = 1 iff the operator has full column rank; the remark
after Prop. 2 predicts familiar ID can be high for any injective-on-vocabulary
operator and degrades with V.
"""
import argparse, json, os, time
import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge


def sample_parts(rng, n, d, dist):
    """Part distributions: Gaussian (the ceiling of Prop. 3 is exact), uniform, sparse
    (10% nonzero), and correlated (Gaussian with a decaying spectrum), all unit variance."""
    if dist == "gauss":
        return rng.normal(size=(n, d))
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, d))
    if dist == "sparse":
        return rng.normal(size=(n, d)) * (rng.random((n, d)) < 0.1) / np.sqrt(0.1)
    if dist == "corr":
        s = 1.0 / np.arange(1, d + 1)
        return rng.normal(size=(n, d)) * np.sqrt(s * d / s.sum())
    raise ValueError(dist)


def circ_conv(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a, axis=-1) * np.fft.fft(b, axis=-1), axis=-1))


def make_ops(d, rng):
    ops = {}
    ops["sum"] = (lambda x, y: x + y)
    ops["weighted"] = (lambda x, y: x + 0.5 * y)
    ops["hadamard"] = (lambda x, y: x * y)
    ops["concat"] = (lambda x, y: np.concatenate([x, y], -1))
    for ratio in [0.5, 0.75, 1.0, 1.5]:
        m = int(round(ratio * 2 * d))
        A = rng.normal(size=(d, m)) / np.sqrt(d)
        B = rng.normal(size=(d, m)) / np.sqrt(d)
        ops[f"proj_m{ratio:g}"] = (lambda x, y, A=A, B=B: x @ A + y @ B)
    r1, r2 = rng.normal(size=d) / np.sqrt(d), rng.normal(size=d) / np.sqrt(d)
    ops["hrr"] = (lambda x, y: circ_conv(x, r1) + circ_conv(y, r2))
    s1, s2 = rng.choice([-1.0, 1.0], size=d), rng.choice([-1.0, 1.0], size=d)
    ops["map_vsa"] = (lambda x, y: x * s1 + y * s2)
    R = rng.normal(size=(2, 2))  # two random (linearly independent) role vectors in R^2
    ops["tpr"] = (lambda x, y: (np.einsum("...i,j->...ij", x, R[0]) + np.einsum("...i,j->...ij", y, R[1])).reshape(*x.shape[:-1], -1))
    for ratio in [0.5, 1.0, 2.0]:
        m = int(round(ratio * 2 * d))
        W = rng.normal(size=(2 * d, m)) / np.sqrt(2 * d)
        ops[f"tanh_m{ratio:g}"] = (lambda x, y, W=W: np.tanh(np.concatenate([x, y], -1) @ W))
    return ops


def r2(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def ident(pred, true_idx, cands):
    # cosine nearest neighbour among candidate parts
    p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    c = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-9)
    return float(((p @ c.T).argmax(1) == true_idx).mean())


def role_ok(pred, x, y):
    return float((((pred - x) ** 2).sum(1) < ((pred - y) ** 2).sum(1)).mean())


def r2_pop(pred, true):
    """R^2 against the population variance of the parts (d: every distribution has unit
    variance per coordinate), as in Eq. (1); the sample spread of a few test parts is not used."""
    return float(1 - ((pred - true) ** 2).sum() / (len(true) * true.shape[1]))


N_FRESH = 2000  # fresh parts are drawn from a fixed pool independent of V


def run(d, V, seed, n_train=6000, n_test=2000, dist="gauss", kernel=False):
    rng = np.random.default_rng(seed)
    ops = make_ops(d, rng)
    vocab = sample_parts(rng, V, d, dist)
    fresh = sample_parts(rng, V, d, dist)          # V fresh candidates, for identification only
    pool = sample_parts(rng, N_FRESH, d, dist)     # fresh parts for R^2
    # hold out 20% of ordered pairs (i != j) for the familiar test
    pairs = np.array([(i, j) for i in range(V) for j in range(V) if i != j])
    rng.shuffle(pairs)
    n_ho = max(1, len(pairs) // 5)
    test_pairs, train_pairs = pairs[:n_ho], pairs[n_ho:]
    tr = train_pairs[rng.integers(0, len(train_pairs), n_train)]
    te = test_pairs[rng.integers(0, len(test_pairs), n_test)]
    fr = np.stack([rng.integers(0, V, n_test), rng.integers(0, V, n_test)], 1)
    fr = fr[fr[:, 0] != fr[:, 1]]
    fp = rng.integers(0, N_FRESH, (n_test, 2))
    fp = fp[fp[:, 0] != fp[:, 1]]
    out = []
    for name, g in ops.items():
        Xtr0 = g(vocab[tr[:, 0]], vocab[tr[:, 1]])
        Xte0 = g(vocab[te[:, 0]], vocab[te[:, 1]])
        Xfr0 = g(fresh[fr[:, 0]], fresh[fr[:, 1]])
        Xfp0 = g(pool[fp[:, 0]], pool[fp[:, 1]])
        mu, sd = Xtr0.mean(0), Xtr0.std(0) + 1e-8
        Xtr, Xte, Xfr, Xfp = (Xtr0 - mu) / sd, (Xte0 - mu) / sd, (Xfr0 - mu) / sd, (Xfp0 - mu) / sd
        Ytr = vocab[tr[:, 0]]
        readouts = {
            "linear": RidgeCV(alphas=np.logspace(-4, 3, 15)),
            "ols": LinearRegression(),  # minimum-norm least squares
            "ols_raw": LinearRegression(),  # minimum-norm least squares on centred, unstandardized composites
            "mlp": MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=400, early_stopping=True, random_state=seed),
            **({"kernel": KernelRidge(alpha=1e-2, kernel="rbf", gamma=1.0 / Xtr.shape[1])} if kernel else {}),
        }
        for rname, model in readouts.items():
            t0 = time.time()
            A = (lambda X0, X: X0 - mu) if rname == "ols_raw" else (lambda X0, X: X)
            model.fit(A(Xtr0, Xtr), Ytr)
            pte, pfr, pfp = model.predict(A(Xte0, Xte)), model.predict(A(Xfr0, Xfr)), model.predict(A(Xfp0, Xfp))
            out.append(dict(op=name, readout=rname, d=d, V=V, seed=seed, m=int(Xtr.shape[1]),
                            fam_r2=r2(pte, vocab[te[:, 0]]), fresh_r2=r2_pop(pfp, pool[fp[:, 0]]),
                            fresh_r2_small=r2(pfr, fresh[fr[:, 0]]),
                            fam_id=ident(pte, te[:, 0], vocab), fresh_id=ident(pfr, fr[:, 0], fresh),
                            fam_role=role_ok(pte, vocab[te[:, 0]], vocab[te[:, 1]]),
                            fresh_role=role_ok(pfp, pool[fp[:, 0]], pool[fp[:, 1]]),
                            secs=time.time() - t0))
        # nearest-neighbour decoder: match composite to the stored composite table of all
        # vocabulary pairs (pure vocabulary lookup; for fresh parts the table is the
        # fresh vocabulary's, i.e. the decoder is given the new parts but no training)
        for split, P, idx in [("fam", vocab, te), ("fresh", fresh, fr)]:
            table_pairs = np.array([(i, j) for i in range(V) for j in range(V) if i != j])
            T = (g(P[table_pairs[:, 0]], P[table_pairs[:, 1]]) - mu) / sd
            Q = g(P[idx[:, 0]], P[idx[:, 1]])
            Q = (Q - mu) / sd
            # chunked nearest neighbour
            hits = []
            for s in range(0, len(Q), 500):
                dist = ((Q[s:s + 500, None, :] - T[None]) ** 2).sum(-1) if len(T) * Q.shape[1] < 4e6 else \
                    (Q[s:s + 500] ** 2).sum(1)[:, None] - 2 * Q[s:s + 500] @ T.T + (T ** 2).sum(1)[None]
                hits.append(table_pairs[dist.argmin(1), 0] == idx[s:s + 500, 0])
            out.append(dict(op=name, readout="nn_table", d=d, V=V, seed=seed, m=int(T.shape[1]),
                            **{f"{split}_id": float(np.concatenate(hits).mean())}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist", default="gauss")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.d, a.V, a.seed, dist=a.dist, kernel=a.kernel)
    for row in res:
        row["dist"] = a.dist
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out, len(res))
