"""E12 (remedy S1): learn parts from singletons, relations from composites.

Proposition span says a procedure fitted on composites of V parts is undetermined outside their
span, and a decoder trained on 8 glyphs fails on fresh glyphs while a new decoder trained on 64
reads them from the same state (E2 redecode).  The span requirement is over parts, so part
diversity might come from cheap single-part data.  Each network trains end to end on composites
of V = 8 glyphs, as in E2, and on a share of singletons: one glyph in one role with the other role
blank, rendered under the same relations.  Conditions:
  base      composites of the 8 glyphs only (E2)
  single8   plus singletons of the same 8 glyphs (the singleton format without new parts)
  singleN   plus singletons of N other glyphs that never appear in a composite
Scored on composites of fresh glyphs (the E2 behavior set) and on held-out pairs of the 8 glyphs.
"""
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn.functional as F
import e2_glyph as E
from e2_redecode import fresh_behavior


def render_single(world, z, role, rel):
    """One glyph in `role` (0 or 1) with the other role blank, under relation `rel`."""
    g = world.glyph(z)
    blank = np.zeros_like(g)
    ga, gb = (g, blank) if role == 0 else (blank, g)
    out = np.zeros((len(z), *E.CANVAS))
    if rel == 0:
        out[..., :, :8], out[..., :, 8:] = ga, gb
    else:
        out[..., :, 4:12] = ga + gb[..., ::-1]
    return out


def train(arch, m, V, seed, steps, single, nsingle, share, npairs=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = E.World(seed)
    vocab = rng.normal(size=(V, E.Q))
    rng.normal(size=(64, E.Q))  # E2's fresh draw, kept so that the training split matches E2
    tr, heldout, _ = E.splits(V, rng)
    if npairs is not None:  # fixed number of distinct training pairs, as in E2's fixed-pairs control
        tr = tr[:npairs]
    model = E.make_model(arch, m)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    cache = {r: torch.tensor(world.render(vocab[tr[:, 0]], vocab[tr[:, 1]], r), dtype=torch.float32) for r in (0, 1)}
    if single == "base":
        S = None
    else:
        Ps = vocab if single == "same" else np.random.default_rng(40_000 + seed).normal(size=(nsingle, E.Q))
        # S[role][rel]: the singleton images of every part
        S = {ro: {r: torch.tensor(render_single(world, Ps, ro, r), dtype=torch.float32) for r in (0, 1)} for ro in (0, 1)}
    bs = 256
    ns = int(round(share * bs)) if S is not None else 0
    for _ in range(steps):
        idx = torch.randint(0, len(tr), (bs - ns,))
        rin = torch.randint(0, 2, (bs - ns,)); rout = torch.randint(0, 2, (bs - ns,))
        x = torch.where(rin.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        y = torch.where(rout.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        ro_all = torch.zeros(bs - ns, dtype=torch.long)
        if ns:
            j = torch.randint(0, len(S[0][0]), (ns,))
            ro = torch.randint(0, 2, (ns,)); ri = torch.randint(0, 2, (ns,)); rt = torch.randint(0, 2, (ns,))
            pick = lambda rel: torch.where(ro.view(-1, 1, 1) == 0, S[0][0][j] if rel == 0 else S[0][1][j],
                                           S[1][0][j] if rel == 0 else S[1][1][j])
            xs = torch.where(ri.view(-1, 1, 1) == 0, pick(0), pick(1))
            ys = torch.where(rt.view(-1, 1, 1) == 0, pick(0), pick(1))
            x, y, rout = torch.cat([x, xs]), torch.cat([y, ys]), torch.cat([rout, rt])
        loss = F.mse_loss(model.decode(model.encode(x), rout), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model, world, vocab, heldout


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp,cnn,slot,tf")
    ap.add_argument("--conds", default="base,single8,single64,single256")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--outdir", default="results/e12")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--npairs", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E.SECOND["rel"] = "over"
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        for arch in a.archs.split(","):
            for cond in a.conds.split(","):
                fn = os.path.join(a.outdir, f"{arch}_{cond}_m{a.m}_V{a.V}_sh{a.share}" + (f"_p{a.npairs}" if a.npairs else "") + f"_s{seed}.json")
                if os.path.exists(fn):
                    continue
                single, n = ("base", 0) if cond == "base" else (("same", a.V) if cond == f"single{a.V}" else ("new", int(cond[6:])))
                model, world, vocab, heldout = train(arch, a.m, a.V, seed, a.steps, single, n, a.share, a.npairs)
                prng = np.random.default_rng(20_000 + seed)  # E2's fresh behavior set
                prng.normal(size=(400, E.Q)); prng.integers(0, 400, (2000, 2))
                fresh_b = prng.normal(size=(256, E.Q))
                fp = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
                fp = np.concatenate([fp, fp[:, ::-1]])
                res = dict(arch=arch, cond=cond, m=a.m, V=a.V, npairs=a.npairs, share=a.share, seed=seed, steps=a.steps,
                           fresh=fresh_behavior(model, world, fresh_b, fp), heldout=fresh_behavior(model, world, vocab, heldout))
                if a.diagnose:  # where the remaining gap lies: single fresh glyphs, and one fresh glyph beside a trained one
                    with torch.no_grad():
                        for ro in (0, 1):
                            ys = render_single(world, fresh_b[:128], ro, 1)
                            o = model.decode(model.encode(torch.tensor(ys, dtype=torch.float32)), torch.ones(128, dtype=torch.long)).numpy()
                            res[f"single_fresh_role{ro + 1}"] = E.r2(o, ys)
                    PM = np.concatenate([vocab, fresh_b[:128]])
                    vi = np.arange(128) % len(vocab)
                    res["half_fresh_role1"] = fresh_behavior(model, world, PM, np.stack([len(vocab) + np.arange(128), vi], 1))
                    res["half_fresh_role2"] = fresh_behavior(model, world, PM, np.stack([vi, len(vocab) + np.arange(128)], 1))
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)
                print(fn, {k: round(np.mean(list(res[k].values())), 3) for k in ("fresh", "heldout")}, flush=True)
