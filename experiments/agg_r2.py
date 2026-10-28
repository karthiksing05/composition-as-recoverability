"""Aggregate round-2 results (E2 v2 evaluation, E6 controls, E5 structure) into summary tables.

Relation index r1 is the second relation (OVER or OCCLUDE); r0 is BESIDE.  Behavior is
averaged over output relations; probes are averaged over the two roles."""
import json, glob, os, sys
import numpy as np, pandas as pd

R = "results"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)


def load(d):
    rows = []
    for f in glob.glob(f"{R}/{d}/*.json"):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        rows += r if isinstance(r, list) else [r]
    return pd.DataFrame(rows)


def e2cols(df):
    for r in "01":
        for s in ("train", "heldout", "strong", "fresh"):
            df[f"beh_{s}_{r}"] = df[[f"beh_{s}_r{r}{o}" for o in "01"]].mean(1)
        for p in ("pool", "vocab", "init"):
            for s in ("heldout", "fresh"):
                df[f"{p}_{s}_{r}"] = df[[f"probe_{p}_{s}_r{r}_role{k}" for k in "12"]].mean(1)
            df[f"{p}_strong_{r}"] = df[f"probe_{p}_strong_r{r}_role2"]
    return df


COLS = ["beh_heldout_1", "beh_strong_1", "beh_fresh_1", "pool_fresh_1", "vocab_fresh_1", "init_fresh_1",
        "ey_ceiling", "inter_fresh_r1", "inter_input_fresh_r1", "beh_fresh_0", "pool_fresh_0"]


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def table(df, keys, cols, name):
    g = df.groupby(keys)[cols]
    print(f"\n=== {name}: n per cell {g.size().min()}-{g.size().max()}, total {len(df)}")
    print(g.mean().round(3).to_string())
    return g


def within_corr(df, x, y, keys):
    """Pearson correlation of x and y after removing each condition's mean (within-condition)."""
    d = df[[x, y] + keys].dropna()
    dx = d[x] - d.groupby(keys)[x].transform("mean")
    dy = d[y] - d.groupby(keys)[y].transform("mean")
    return float(np.corrcoef(dx, dy)[0, 1]), float(np.corrcoef(d[x], d[y])[0, 1]), len(d)


def boot_ci(df, x, y, keys, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = df[[x, y] + keys].dropna().reset_index(drop=True)
    vals = []
    for _ in range(n):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        dx = s[x] - s.groupby(keys)[x].transform("mean"); dy = s[y] - s.groupby(keys)[y].transform("mean")
        vals.append(np.corrcoef(dx, dy)[0, 1])
    return np.nanpercentile(vals, [2.5, 97.5]).round(3).tolist()


if __name__ == "__main__":
    out = {}
    e2 = e2cols(load("e2v2"))
    table(e2, ["arch", "V", "m"], COLS, "E2 v2 (OVER)")
    occ = e2cols(load("e2v2occ"))
    if len(occ):
        table(occ, ["arch", "V", "m"], COLS, "E2 v2 (OCCLUDE)")
    # does representation (population probe) predict behavior on fresh parts beyond condition?
    for nm, d in (("over", e2), ("occ", occ)):
        if not len(d):
            continue
        for y in ("beh_fresh_1", "beh_heldout_1", "beh_strong_1"):
            w, pooled, n = within_corr(d, "pool_fresh_1", y, ["arch", "V", "m"])
            print(f"{nm}: corr(pool_fresh_1, {y}) within={w:.3f} pooled={pooled:.3f} n={n} CI={boot_ci(d, 'pool_fresh_1', y, ['arch','V','m'], 500)}")
            out[f"corr_{nm}_{y}"] = dict(within=w, pooled=pooled, n=n)
    # seed spread of new-role behavior
    print("\nstd across seeds of beh_strong_1 / beh_fresh_1 (m>=32)")
    print(e2[e2.m >= 32].groupby(["arch", "V"])[["beh_strong_1", "beh_fresh_1"]].std().round(3).unstack(0).to_string())
    # E6: fixed number of training pairs (vary V at constant pairs), aux reconstruction loss, latent dim q
    for d, keys in (("e6pairs", ["arch", "V"]), ("e6aux", ["arch", "V"]), ("e2q", ["arch", "m"])):
        df = load(d)
        if len(df):
            table(e2cols(df), keys, COLS + ["n_train"], d)
    # matched E2 baselines for E6 comparisons (m = 64, OVER)
    table(e2[e2.m == 64], ["arch", "V"], COLS + ["n_train"], "E2 v2 baseline m=64")
    # E5
    e5 = load("e5")
    if len(e5):
        c5 = [c for c in e5.columns if c.endswith("_r1") or c.startswith("newpart")]
        for sec in sorted(e5.second.unique()):
            table(e5[e5.second == sec], ["arch", "V", "m"], sorted(c5), f"E5 second={sec}")
    # HRR per-height roles
    f = f"{R}/e3hrr2/hrr_unitary.json"
    if os.path.exists(f):
        h = pd.DataFrame(json.load(open(f, encoding="utf-8")))
        print("\n=== HRR per-height roles")
        print(h.groupby(["w", "V", "split", "depth"])[["r2", "id"]].mean().round(3).unstack("depth").to_string())
    with open(f"{R}/agg_r2.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
