"""E15 (remedy S2): add parts by writing codes, not by gradient steps, one part after another.

E5 adds one new glyph either by fine-tuning on five composites or by reading its code from one
composite and writing it through the role-2 map.  Here 32 new glyphs arrive in sequence, each
shown in five composites with trained glyphs, always in role 1 (over BESIDE).  Methods:
  none      the trained network, unchanged (encode the composite, decode)
  finetune  200 Adam steps on each new glyph's five composites, in sequence
  replay    the same, with 32 composites of training pairs in every step
  write     the network stays frozen; role maps c, B1, B2 and a code readout are fitted once on
            a pool of parts (as in E5), each new glyph's role-1 code is read from its five
            composites and stored (16 numbers), and a composite is built as c + B1 e(a) + B2 e(b)
            and decoded by the network's own decoder
After 1, 4, 16 and 32 additions we score the decoded composite (BESIDE) for: added glyphs in role
1, added glyphs in role 2 (never shown there), two added glyphs together (never shown together),
and held-out pairs of training glyphs (forgetting).  The fine-tuned and unchanged networks
receive the composite image; the write method receives only the stored codes.
"""
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, RidgeCV
import e2_glyph as E

K = 16
CHECK = (1, 4, 16, 32)


def r2(p, t):
    p, t = p.reshape(len(p), -1), t.reshape(len(t), -1)
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def run(arch, m, V, seed, steps, n_new=32, ft_steps=200, base="e2"):
    E.SECOND["rel"] = "over"
    if base == "e2":
        ctx = E.train_model(arch, m, V, seed, steps)
        model, world, vocab, train, heldout = (ctx[k] for k in ("model", "world", "vocab", "train", "heldout"))
    else:  # remedy S1 first: the same composites plus singletons of N other glyphs (E12), base = "singleN"
        from e12_singletons import train as train_single
        model, world, vocab, heldout = train_single(arch, m, V, seed, steps, "new", int(base[6:]), 0.5)
        r0 = np.random.default_rng(seed); r0.normal(size=(V, E.Q)); r0.normal(size=(64, E.Q))
        train, _, _ = E.splits(V, r0)  # E12 draws its split from the same stream
    rng = np.random.default_rng(30_000 + seed)
    G = world.glyph(rng.normal(size=(4000, E.Q))).reshape(4000, -1)
    gmu = G.mean(0)
    _, _, Vt = np.linalg.svd(G - gmu, full_matrices=False)
    code = lambda z: (world.glyph(z).reshape(len(z), -1) - gmu) @ Vt[:K].T

    def enc(mm, za, zb):
        with torch.no_grad():
            return mm.encode(torch.tensor(world.render(za, zb, 0), dtype=torch.float32)).numpy()

    def dec(mm, h):
        with torch.no_grad():
            return mm.decode(torch.tensor(h, dtype=torch.float32), torch.zeros(len(h), dtype=torch.long)).numpy()

    # role maps and code readout of the frozen network, fitted on a pool of parts it never saw
    pool = rng.normal(size=(400, E.Q)); pp = rng.integers(0, 400, (2000, 2))
    H = enc(model, pool[pp[:, 0]], pool[pp[:, 1]])
    X = np.concatenate([code(pool[pp[:, 0]]), code(pool[pp[:, 1]])], 1)
    fit = Ridge(alpha=1e-3).fit(X, H)
    c, B1, B2 = fit.intercept_, fit.coef_[:, :K], fit.coef_[:, K:]
    U = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(H, X)

    new = rng.normal(size=(n_new, E.Q))
    partners = vocab[rng.integers(0, V, (n_new, 5))]
    old_codes = {}  # code of each trained glyph, read from composites with other trained glyphs
    for i, z in enumerate(vocab):
        oth = vocab[rng.integers(0, V, 8)]
        old_codes[i] = U.predict(enc(model, np.repeat(z[None], 8, 0), oth))[:, :K].mean(0)
    ev_old_idx = rng.integers(0, V, (n_new, 8))
    ev_old = vocab[ev_old_idx]
    ho = heldout[:256]

    def score(kind, mm, stored, k):
        """Scores after k additions.  kind 'image' encodes the composite; 'write' builds it from codes."""
        out = {}
        a_new = np.repeat(new[:k], 8, 0); o = ev_old[:k].reshape(-1, E.Q); oi = ev_old_idx[:k].reshape(-1)
        pair = rng.permutation(k) if k > 1 else None
        for tag, (za, zb) in {"role1": (a_new, o), "role2": (o, a_new)}.items():
            tgt = world.render(za, zb, 0)
            if kind == "image":
                out[tag] = r2(dec(mm, enc(mm, za, zb)), tgt)
            else:
                en = np.repeat(np.array(stored[:k]), 8, 0); eo = np.array([old_codes[j] for j in oi])
                ea, eb = (en, eo) if tag == "role1" else (eo, en)
                out[tag] = r2(dec(model, c + ea @ B1.T + eb @ B2.T), tgt)
        if k > 1:
            za, zb = new[:k], new[pair]
            keep = np.arange(k) != pair
            za, zb = za[keep], zb[keep]
            tgt = world.render(za, zb, 0)
            if kind == "image":
                out["newnew"] = r2(dec(mm, enc(mm, za, zb)), tgt)
            else:
                st = np.array(stored[:k])
                out["newnew"] = r2(dec(model, c + st[np.arange(k)[keep]] @ B1.T + st[pair[keep]] @ B2.T), tgt)
        za, zb = vocab[ho[:, 0]], vocab[ho[:, 1]]
        out["heldout"] = r2(dec(mm, enc(mm, za, zb)), world.render(za, zb, 0)) if kind == "image" else \
            r2(dec(model, enc(model, za, zb)), world.render(za, zb, 0))
        return out

    res = []
    mm = {"finetune": copy.deepcopy(model), "replay": copy.deepcopy(model)}
    opts = {k: torch.optim.Adam(v.parameters(), lr=3e-4) for k, v in mm.items()}
    stored = []
    for i in range(n_new):
        za, zb = np.repeat(new[i:i + 1], 5, 0), partners[i]
        stored.append(U.predict(enc(model, za, zb))[:, :K].mean(0))
        for meth, net in mm.items():
            net.train()
            for step in range(ft_steps):
                rin = step % 2
                x = torch.tensor(world.render(za, zb, rin), dtype=torch.float32)
                xs, rs = [x], [torch.full((5,), rin, dtype=torch.long)]
                if meth == "replay":
                    idx = rng.integers(0, len(train), 32)
                    xs.append(torch.tensor(world.render(vocab[train[idx, 0]], vocab[train[idx, 1]], rin), dtype=torch.float32))
                    rs.append(torch.full((32,), rin, dtype=torch.long))
                xb, rb = torch.cat(xs), torch.cat(rs)
                loss = F.mse_loss(net.decode(net.encode(xb), rb), xb)
                opts[meth].zero_grad(); loss.backward(); opts[meth].step()
            net.eval()
        k = i + 1
        if k in CHECK:
            meta = dict(arch=arch, m=m, V=V, seed=seed, k=k, base=base)
            res.append(dict(meta, method="none", **score("image", model, None, k)))
            for meth, net in mm.items():
                res.append(dict(meta, method=meth, **score("image", net, None, k)))
            res.append(dict(meta, method="write", **score("write", None, stored, k)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp,cnn,slot")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--V", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--outdir", default="results/e15")
    ap.add_argument("--base", default="e2", help="e2, or singleN: train with singletons of N new glyphs first")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        for arch in a.archs.split(","):
            fn = os.path.join(a.outdir, f"{arch}_m{a.m}_V{a.V}" + ("" if a.base == "e2" else f"_{a.base}") + f"_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(arch, a.m, a.V, seed, a.steps, base=a.base)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
            for x in r:
                print(arch, seed, x["k"], x["method"], {k: round(v, 3) for k, v in x.items() if k in ("role1", "role2", "newnew", "heldout")}, flush=True)
