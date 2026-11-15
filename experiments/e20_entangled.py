"""E20/E22/E25: glyph networks with inputs whose parts are not already linearly available.

The paper's toy has a weakness: the input pixels recover held-out glyphs at 0.91, and untrained
encoders at 0.82-0.83, so recoverability there is inherited from the input.  Here a fixed nonlinear
mix (orthogonal maps and tanh at gain g, `layers` times) entangles the input composite.  Targets stay
in pixels, so the encoder has to learn to unmix.  gain = 0 leaves the input untouched and must
reproduce the paper's numbers.

Measured on held-out glyphs (rin = 1, as the paper): the network's own output, the population and
vocabulary readouts of its state, the same readout of the untrained encoder and of the input, and
a new decoder trained on composites of 64 other glyphs, from the trained and from the untrained encoder.
kappa uses the concatenation and sum references at the same width.

Objective arms (E22):
  base      decoder loss only
  auxvocab  plus a linear head from h to both glyphs' pixels, on the training pairs
  auxpool   the head also fits composites of 64 other glyphs; the decoder still sees only the vocabulary
For aux arms, `through` scores the fixed render of the head's output as the system's procedure.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
import e2_glyph as E

Dpix = 128


class Mix:
    def __init__(self, world, seed, gain, layers):
        self.gain, self.layers = gain, layers
        if gain == 0:
            return
        g = np.random.default_rng(50_000 + seed)
        self.Q = [np.linalg.qr(g.normal(size=(Dpix, Dpix)))[0] for _ in range(layers)]
        self.b = [g.uniform(-1, 1, size=Dpix) for _ in range(layers)]
        z = g.normal(size=(4000, world.q)); y = g.normal(size=(4000, world.q))
        X = np.concatenate([world.render(z, y, r).reshape(4000, -1) for r in (0, 1)])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3

    def __call__(self, x):
        if self.gain == 0:
            return x
        s = x.shape
        z = (x.reshape(-1, Dpix) - self.mu) / self.sd
        for Q, b in zip(self.Q, self.b):
            z = np.tanh(self.gain * z @ Q + b)
        return z.reshape(s)


def readout(Hfit, tgts, Hev, evs):
    mu, sd = Hfit.mean(0), Hfit.std(0) + 1e-6
    out, preds = [], []
    for gt, tt in zip(tgts, evs):
        lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((Hfit - mu) / sd, gt)
        p = lin.predict((Hev - mu) / sd)
        out.append(E.r2(p, tt)); preds.append(p)
    return float(np.mean(out)), preds


def render_pixels(g1, g2, rout):
    """The fixed renderer applied to predicted glyph pixels (additive relations only)."""
    a, b = g1.reshape(-1, 8, 8), g2.reshape(-1, 8, 8)
    out = np.zeros((len(a), 8, 16))
    if rout == 0:
        out[:, :, :8], out[:, :, 8:] = a, b
    else:
        out[:, :, 4:12] = a + b[:, :, ::-1]
    return out


def run(arch, m, V, seed, steps, gain, layers, arm="base", redecode=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = E.World(seed, E.Q)
    mix = Mix(world, seed, gain, layers)
    vocab = rng.normal(size=(V, E.Q))
    rng.normal(size=(64, E.Q))
    train, heldout, strong = E.splits(V, rng)
    model = E.make_model(arch, m)
    init_model = copy.deepcopy(model)
    nrng = np.random.default_rng(30_000 + seed)
    Pn = nrng.normal(size=(64, E.Q))
    npairs = np.array([(i, j) for i in range(64) for j in range(64) if i != j])
    G = lambda z: world.glyph(z).reshape(len(z), -1)
    T = lambda a: torch.tensor(a, dtype=torch.float32)

    head = torch.nn.Linear(m, Dpix) if arm != "base" else None
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Y = {r: T(world.render(vocab[train[:, 0]], vocab[train[:, 1]], r)) for r in (0, 1)}
    X = {r: T(mix(world.render(vocab[train[:, 0]], vocab[train[:, 1]], r))) for r in (0, 1)}
    GG = T(np.c_[G(vocab[train[:, 0]]), G(vocab[train[:, 1]])])
    if arm == "auxpool":
        Xn = {r: T(mix(world.render(Pn[npairs[:, 0]], Pn[npairs[:, 1]], r))) for r in (0, 1)}
        GGn = T(np.c_[G(Pn[npairs[:, 0]]), G(Pn[npairs[:, 1]])])
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, len(train), (256,))
        rin, rout = torch.randint(0, 2, (256,)), torch.randint(0, 2, (256,))
        x = torch.where(rin.view(-1, 1, 1) == 0, X[0][idx], X[1][idx])
        y = torch.where(rout.view(-1, 1, 1) == 0, Y[0][idx], Y[1][idx])
        h = model.encode(x)
        loss = F.mse_loss(model.decode(h, rout), y)
        total = loss
        if arm == "auxvocab":
            total = total + F.mse_loss(head(h), GG[idx])
        if arm == "auxpool":
            jdx = torch.randint(0, len(npairs), (256,))
            rj = torch.randint(0, 2, (256,))
            xn = torch.where(rj.view(-1, 1, 1) == 0, Xn[0][jdx], Xn[1][jdx])
            total = total + 0.5 * F.mse_loss(head(h), GG[idx]) + 0.5 * F.mse_loss(head(model.encode(xn)), GGn[jdx])
        opt.zero_grad(); total.backward(); opt.step(); sched.step()
    model.eval(); init_model.eval()

    prng = np.random.default_rng(20_000 + seed)  # the same streams as e2_glyph.run
    pool = prng.normal(size=(400, E.Q))
    pool_pairs = prng.integers(0, 400, (2000, 2))
    fresh_b, fresh_p = prng.normal(size=(256, E.Q)), prng.normal(size=(256, E.Q))
    fp = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
    fp = np.concatenate([fp, fp[:, ::-1]])
    rin = 1

    def enc(za, zb, mdl=model):
        with torch.no_grad():
            return mdl.encode(T(mix(world.render(za, zb, rin)))).numpy()

    def own(mdl, P, pr):
        za, zb = P[pr[:, 0]], P[pr[:, 1]]
        with torch.no_grad():
            h = mdl.encode(T(mix(world.render(za, zb, rin))))
            return float(np.mean([E.r2(mdl.decode(h, torch.full((len(h),), o, dtype=torch.long)).numpy(), world.render(za, zb, o)) for o in (0, 1)]))

    res = dict(arch=arch, m=m, V=V, seed=seed, steps=steps, gain=gain, layers=layers, arm=arm, final_loss=float(loss),
               train_secs=time.time() - t0, own_heldout=own(model, fresh_b, fp), own_newpairs=own(model, vocab, heldout))
    pa, pb = pool[pool_pairs[:, 0]], pool[pool_pairs[:, 1]]
    fa, fb = fresh_p[fp[:, 0]], fresh_p[fp[:, 1]]
    gfa, gfb = G(fa), G(fb)
    res["read_trained"], preds = readout(enc(pa, pb), (G(pa), G(pb)), enc(fa, fb), (gfa, gfb))
    res["read_untrained"], _ = readout(enc(pa, pb, init_model), (G(pa), G(pb)), enc(fa, fb, init_model), (gfa, gfb))
    Xp, Xf = mix(world.render(pa, pb, rin)).reshape(len(pa), -1), mix(world.render(fa, fb, rin)).reshape(len(fa), -1)
    res["read_input"], _ = readout(Xp, (G(pa), G(pb)), Xf, (gfa, gfb))
    ta, tb = vocab[train[:, 0]], vocab[train[:, 1]]
    res["read_vocab"], _ = readout(enc(ta, tb), (G(ta), G(tb)), enc(fa, fb), (gfa, gfb))
    # the population readout used as a procedure: render what it recovers
    res["read_rendered"] = float(np.mean([E.r2(render_pixels(preds[0], preds[1], o), world.render(fa, fb, o)) for o in (0, 1)]))
    # references at this width, for kappa
    proj = np.random.default_rng(31_000 + seed)
    Pc = proj.normal(size=(Dpix, m)) / np.sqrt(Dpix); Ps = proj.normal(size=(64, m)) / np.sqrt(64)
    res["ref_concat"], _ = readout(np.c_[G(pa), G(pb)] @ Pc, (G(pa), G(pb)), np.c_[gfa, gfb] @ Pc, (gfa, gfb))
    res["ref_sum"], _ = readout((G(pa) + G(pb)) @ Ps, (G(pa), G(pb)), (gfa + gfb) @ Ps, (gfa, gfb))
    if head is not None:
        with torch.no_grad():
            o = head(T(enc(fresh_b[fp[:, 0]], fresh_b[fp[:, 1]]))).numpy()
        ba, bb = fresh_b[fp[:, 0]], fresh_b[fp[:, 1]]
        res["through"] = float(np.mean([E.r2(render_pixels(o[:, :64], o[:, 64:], k), world.render(ba, bb, k)) for k in (0, 1)]))
    if redecode:
        for tag, mdl0 in (("trained", model), ("untrained", init_model)):
            torch.manual_seed(10_000 + seed)
            mdl = copy.deepcopy(mdl0)
            dec = mdl.gdec if isinstance(mdl, E.SlotModel) else mdl.dec
            for mod in dec.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()
            for p in mdl.enc.parameters():
                p.requires_grad_(False)
            Yn = {r: T(world.render(Pn[npairs[:, 0]], Pn[npairs[:, 1]], r)) for r in (0, 1)}
            with torch.no_grad():
                hc = {r: mdl.encode(T(mix(world.render(Pn[npairs[:, 0]], Pn[npairs[:, 1]], r)))) for r in (0, 1)}
            o2 = torch.optim.Adam(dec.parameters(), lr=1e-3)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, steps)
            for _ in range(steps):
                idx = torch.randint(0, len(npairs), (256,))
                ri, ro = torch.randint(0, 2, (256,)), torch.randint(0, 2, (256,))
                h = torch.where(ri.view(-1, 1) == 0, hc[0][idx], hc[1][idx])
                y = torch.where(ro.view(-1, 1, 1) == 0, Yn[0][idx], Yn[1][idx])
                l2 = F.mse_loss(mdl.decode(h, ro), y)
                o2.zero_grad(); l2.backward(); o2.step(); s2.step()
            mdl.eval()
            res[f"redec_{tag}"] = own(mdl, fresh_b, fp)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp+cnn+slot+tf")
    ap.add_argument("--ms", default="64")
    ap.add_argument("--Vs", default="8")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--arms", default="base")
    ap.add_argument("--no-redecode", action="store_true")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E.SECOND["rel"] = "over"
    os.makedirs(a.outdir, exist_ok=True)
    for arm in a.arms.split("+"):
        for arch in a.archs.split("+"):
            for V in map(int, a.Vs.split("+")):
                for m in map(int, a.ms.split("+")):
                    fn = os.path.join(a.outdir, f"{arch}_m{m}_V{V}_g{a.gain:g}_l{a.layers}_{arm}_st{a.steps}_s{a.seed}.json")
                    if os.path.exists(fn):
                        continue
                    r = run(arch, m, V, a.seed, a.steps, a.gain, a.layers, arm, not a.no_redecode)
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump(r, f, indent=1)
                    print(fn, {k: round(v, 3) for k, v in r.items() if isinstance(v, float) and k != "train_secs"}, flush=True)
