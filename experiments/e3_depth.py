"""E3: recoverability across depth (productivity, Prop. 5).

A shared recursive composer C: R^w x R^w -> R^w and a shared decomposer
D: R^w -> R^w x R^w (a RAAM; Pollack 1990) are trained end to end to encode
balanced binary trees of depth 1-2 whose leaves are parts in R^d, and to decode
every leaf by applying D recursively.  We test depths 1-5 with leaves from the
training vocabulary (identification among V candidates) and fresh leaves
(variance explained, R^2).  References: recursive TPR (exact, width d 2^n) and
HRR with fixed width (noisy unbinding; identification with clean-up).
Prop. 5 bounds the fresh-leaf R^2 of any fixed-width linear composer by
w / (2^n d); a vocabulary can keep identification high beyond that bound.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

D_PART = 8


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def pop_readout(encode, n, rng, n_fit=6000, n_test=2000):
    """Definition 1 for a depth-n state: a ridge readout fitted on states of trees whose
    leaves are fresh draws (every leaf distinct) and evaluated on other fresh trees; R^2
    against the population variance of the leaves (unit per coordinate)."""
    from sklearn.linear_model import RidgeCV
    Lf = rng.normal(size=(n_fit, 2 ** n, D_PART)); Lt = rng.normal(size=(n_test, 2 ** n, D_PART))
    Hf, Ht = encode(Lf), encode(Lt)
    mu = Hf.mean(0)
    reg = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(Hf - mu, Lf.reshape(n_fit, -1))
    p = reg.predict(Ht - mu)
    return float(1 - ((p - Lt.reshape(n_test, -1)) ** 2).sum() / (n_test * 2 ** n * D_PART))


def ident(p, t_idx, cands):
    d = ((p[:, None] - cands[None]) ** 2).sum(-1)
    return float((d.argmin(1) == t_idx).mean())


class RAAM(nn.Module):
    def __init__(self, w, d=D_PART, h=256):
        super().__init__()
        self.E = nn.Linear(d, w)
        self.R = nn.Linear(w, d)
        self.C = nn.Sequential(nn.Linear(2 * w, h), nn.Tanh(), nn.Linear(h, w))
        self.D = nn.Sequential(nn.Linear(w, h), nn.Tanh(), nn.Linear(h, 2 * w))
        self.w = w

    def encode(self, leaves):  # leaves: B x 2^n x d
        x = self.E(leaves)
        while x.shape[1] > 1:
            x = self.C(x.reshape(len(x), x.shape[1] // 2, 2 * self.w))
        return x[:, 0]

    def decode(self, h, n):
        x = h[:, None]
        for _ in range(n):
            x = self.D(x).reshape(len(x), x.shape[1] * 2, self.w)
        return self.R(x)


def run_raam(w, V, seed, steps, train_depths=(1, 2), test_depths=(1, 2, 3, 4, 5)):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = torch.tensor(rng.normal(size=(V, D_PART)), dtype=torch.float32)
    fresh = torch.tensor(rng.normal(size=(256, D_PART)), dtype=torch.float32)
    model = RAAM(w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        n = train_depths[step % len(train_depths)]
        idx = torch.randint(0, V, (128, 2 ** n))
        leaves = vocab[idx]
        loss = ((model.decode(model.encode(leaves), n) - leaves) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    res = []
    with torch.no_grad():
        for n in test_depths:
            idx = torch.randint(0, V, (512, 2 ** n))
            out = model.decode(model.encode(vocab[idx]), n).reshape(-1, D_PART).numpy()
            fam_id = ident(out, idx.reshape(-1).numpy(), vocab.numpy())
            fam_r2 = r2(out, vocab[idx].reshape(-1, D_PART).numpy())
            fidx = torch.randint(0, 256, (512, 2 ** n))
            fout = model.decode(model.encode(fresh[fidx]), n).reshape(-1, D_PART).numpy()
            fr_r2 = r2(fout, fresh[fidx].reshape(-1, D_PART).numpy())
            # identification of fresh leaves among V fresh candidates (chance-matched to fam_id)
            cand = fresh[:V]
            cidx = torch.randint(0, V, (512, 2 ** n))
            cout = model.decode(model.encode(cand[cidx]), n).reshape(-1, D_PART).numpy()
            fr_id = ident(cout, cidx.reshape(-1).numpy(), cand.numpy())
            enc = lambda L: model.encode(torch.tensor(L, dtype=torch.float32)).numpy()
            res.append(dict(model="raam", w=w, V=V, seed=seed, depth=n, fam_id=fam_id, fam_r2=fam_r2,
                            fresh_r2=fr_r2, fresh_id=fr_id, bound=min(1.0, w / (2 ** n * D_PART)),
                            pop_r2=pop_readout(enc, n, np.random.default_rng(1000 + seed))))
    return res


def cconv(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a, axis=-1) * np.fft.fft(b, axis=-1), axis=-1))


def ccorr(a, b):  # approximate unbinding of b from a
    return np.real(np.fft.ifft(np.conj(np.fft.fft(b, axis=-1)) * np.fft.fft(a, axis=-1), axis=-1))


def run_hrr(w, V, seed, test_depths=(1, 2, 3, 4, 5)):
    rng = np.random.default_rng(seed)
    vocab = rng.normal(size=(V, D_PART))
    fresh = rng.normal(size=(V, D_PART))
    P = rng.normal(size=(D_PART, w)) / np.sqrt(w)  # lift parts into w dims, pseudo-inverse to read out
    Pinv = np.linalg.pinv(P)
    # unitary roles (unit-modulus spectrum), so that correlation is the exact inverse of binding
    def unitary():
        f = np.fft.fft(rng.normal(size=w))
        return np.real(np.fft.ifft(f / np.abs(f)))
    # one pair of roles per height: with a single pair, circular convolution commutes and
    # role vectors cancel across levels, leaking whole leaves into other leaves' readouts
    roles = [(unitary(), unitary()) for _ in range(max(test_depths))]

    def compose(leaves):  # leaves: B x 2^n x D_PART -> B x w
        x = leaves @ P
        ht = 0
        while x.shape[1] > 1:
            rL, rR = roles[ht]
            x = (cconv(x[:, 0::2], rL) + cconv(x[:, 1::2], rR)) / np.sqrt(2)
            ht += 1
        return x[:, 0]

    res = []
    for n in test_depths:
        pop = pop_readout(compose, n, np.random.default_rng(2000 + seed))
        # correlation unbinding on a fixed pool of fresh leaves, R^2 against the population variance
        Lt = rng.normal(size=(2000, 2 ** n, D_PART))
        x = compose(Lt)[:, None]
        for ht in reversed(range(n)):
            rL, rR = roles[ht]
            x = np.stack([ccorr(x, rL), ccorr(x, rR)], 2).reshape(len(x), -1, w) * np.sqrt(2)
        unb = float(1 - ((x.reshape(-1, w) @ Pinv - Lt.reshape(-1, D_PART)) ** 2).sum() / Lt.size)
        res.append(dict(model="hrr", w=w, V=V, seed=seed, depth=n, split="pool", pop_r2=pop, r2=unb,
                        crosstalk_pred=1 - (2 ** n - 1) * D_PART / w))
        for split, cands in (("fam", vocab), ("fresh", fresh)):
            idx = rng.integers(0, V, (512, 2 ** n))
            x = cands[idx] @ P
            ht = 0
            while x.shape[1] > 1:
                rL, rR = roles[ht]
                x = (cconv(x[:, 0::2], rL) + cconv(x[:, 1::2], rR)) / np.sqrt(2)
                ht += 1
            for ht in reversed(range(n)):
                rL, rR = roles[ht]
                x = np.stack([ccorr(x, rL), ccorr(x, rR)], 2).reshape(len(x), -1, w) * np.sqrt(2)
            out = x.reshape(-1, w) @ Pinv
            tgt = cands[idx].reshape(-1, D_PART)
            res.append(dict(model="hrr", w=w, V=V, seed=seed, depth=n, split=split,
                            r2=r2(out, tgt), id=ident(out, idx.reshape(-1), cands)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="16+32+64+128")
    ap.add_argument("--Vs", default="16")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hrr-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(a.outdir, exist_ok=True)
    if a.hrr_only:
        rows = [r for s in map(int, a.seeds.split("+")) for V in map(int, a.Vs.split("+"))
                for w in map(int, a.ws.split("+")) for r in run_hrr(w, V, s)]
        with open(os.path.join(a.outdir, "hrr_unitary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        raise SystemExit
    for seed in map(int, a.seeds.split("+")):
        for V in map(int, a.Vs.split("+")):
            for w in map(int, a.ws.split("+")):
                fn = os.path.join(a.outdir, f"w{w}_V{V}_s{seed}.json")
                if os.path.exists(fn):
                    continue
                r = run_raam(w, V, seed, a.steps) + run_hrr(w, V, seed)
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(r, f, indent=1)
                print(fn, [(x["depth"], round(x["fam_id"], 2), round(x["fresh_r2"], 2)) for x in r if x["model"] == "raam"], flush=True)
