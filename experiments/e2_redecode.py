"""Audit: does the state of a network trained on few glyphs suffice for its own decoder?

A network trained end to end on V = 8 glyphs renders fresh glyphs poorly, while a
population readout of its state recovers them.  Here we freeze the encoder and train a
new decoder of the same architecture on composites of 64 other glyphs.  If the state
holds the parts, the new decoder renders fresh glyphs about as well as a network trained
end to end on 64 glyphs.  Controls: the untrained encoder with a new decoder on the same
64 glyphs, and the trained encoder with a new decoder on the original training pairs.
"""
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn.functional as F
import e2_glyph as E


def fresh_behavior(model, world, P, pairs):
    """Output R^2 on composites of fresh parts, for each input and output relation."""
    za, zb = P[pairs[:, 0]], P[pairs[:, 1]]
    out = {}
    with torch.no_grad():
        for rin in (0, 1):
            h = model.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32))
            for rout in (0, 1):
                o = model.decode(h, torch.full((len(h),), rout, dtype=torch.long)).numpy()
                out[f"r{rin}{rout}"] = E.r2(o, world.render(za, zb, rout))
    return out


def retrain_decoder(model, world, P, pairs, steps, seed):
    """Reinitialize the decoder, freeze the encoder, and train the decoder on composites of P."""
    torch.manual_seed(10_000 + seed)
    model = copy.deepcopy(model)
    dec = model.gdec if isinstance(model, E.SlotModel) else model.dec
    for mod in dec.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    enc = model.enc
    for p in enc.parameters():
        p.requires_grad_(False)
    cache = {rin: torch.tensor(world.render(P[pairs[:, 0]], P[pairs[:, 1]], rin), dtype=torch.float32) for rin in (0, 1)}
    with torch.no_grad():
        hc = {rin: model.encode(cache[rin]) for rin in (0, 1)}
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(pairs), (256,))
        rin = torch.randint(0, 2, (256,))
        rout = torch.randint(0, 2, (256,))
        h = torch.where(rin.view(-1, 1) == 0, hc[0][idx], hc[1][idx])
        y = torch.where(rout.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        loss = F.mse_loss(model.decode(h, rout), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model, float(loss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp,cnn,slot,tf")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--Vnew", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--outdir", default="results/e2redec")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E.SECOND["rel"] = "over"
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        for arch in a.archs.split(","):
            fn = os.path.join(a.outdir, f"{arch}_m{a.m}_V{a.V}_s{seed}.json")
            if os.path.exists(fn):
                continue
            ctx = E.train_model(arch, a.m, a.V, seed, a.steps)
            world, vocab, train = ctx["world"], ctx["vocab"], ctx["train"]
            # the same fresh glyphs as the main runs (behaviour set of e2_glyph.run)
            prng = np.random.default_rng(20_000 + seed)
            prng.normal(size=(400, E.Q)); prng.integers(0, 400, (2000, 2))
            fresh_b = prng.normal(size=(256, E.Q))
            fp = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
            fp = np.concatenate([fp, fp[:, ::-1]])
            # new glyphs for decoder retraining, disjoint from vocabulary, pool and fresh sets
            nrng = np.random.default_rng(30_000 + seed)
            Pn = nrng.normal(size=(a.Vnew, E.Q))
            npairs = np.array([(i, j) for i in range(a.Vnew) for j in range(a.Vnew) if i != j])
            res = dict(arch=arch, m=a.m, V=a.V, Vnew=a.Vnew, seed=seed, steps=a.steps,
                       orig=fresh_behavior(ctx["model"], world, fresh_b, fp))
            for tag, mdl, P, pr in (("redec_trained", ctx["model"], Pn, npairs),
                                    ("redec_init", ctx["init_model"], Pn, npairs),
                                    ("redec_same", ctx["model"], vocab, train)):
                m2, loss = retrain_decoder(mdl, world, P, pr, a.steps, seed)
                res[tag] = fresh_behavior(m2, world, fresh_b, fp)
                res[tag + "_loss"] = loss
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            print(fn, {k: round(np.mean(list(v.values())), 3) for k, v in res.items() if isinstance(v, dict)}, flush=True)
