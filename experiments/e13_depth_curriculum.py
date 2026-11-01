"""E13 (remedy S4): train the decomposer on the composer's own states at every depth.

E3 trains a RAAM end to end on trees of depth 1-2; its decomposer fails on fresh leaves at
depth 3 while a population readout recovers them from the same states.  Here the composer
trains exactly as in E3, and the decomposer receives extra training on deeper trees built by
that composer, with the composer's gradient cut so the states it builds stay E3's.  Conditions:
  base       E3: depths 1-2, leaves from the vocabulary
  fresh12    depths 1-2 with Gaussian leaves (part diversity without depth)
  self       base, plus decomposer-only loss on depths 3-5 with vocabulary leaves
  selffresh  base, plus decomposer-only loss on depths 3-5 with Gaussian leaves
  jointfresh base, plus a loss on depths 3-5 with Gaussian leaves that also trains the composer
Scored on fresh leaves at depths 1-6 (depth 6 lies past every training depth), against the
population readout of the same states.
"""
import argparse, json, os
import numpy as np
import torch
from e3_depth import RAAM, D_PART, r2, pop_readout

CONDS = ("base", "fresh12", "self", "selffresh", "jointfresh")


def run(cond, w, V, seed, steps, test_depths=(1, 2, 3, 4, 5, 6), deep_bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = torch.tensor(rng.normal(size=(V, D_PART)), dtype=torch.float32)
    fresh = torch.tensor(rng.normal(size=(256, D_PART)), dtype=torch.float32)
    model = RAAM(w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        n = (1, 2)[step % 2]
        leaves = torch.randn(128, 2 ** n, D_PART) if cond == "fresh12" else vocab[torch.randint(0, V, (128, 2 ** n))]
        loss = ((model.decode(model.encode(leaves), n) - leaves) ** 2).mean()
        if cond in ("self", "selffresh", "jointfresh"):
            m = 3 + step % 3
            deep = vocab[torch.randint(0, V, (deep_bs, 2 ** m))] if cond == "self" else torch.randn(deep_bs, 2 ** m, D_PART)
            with torch.set_grad_enabled(cond == "jointfresh"):
                h = model.encode(deep)
            loss = loss + ((model.decode(h, m) - deep) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    res = []
    with torch.no_grad():
        enc = lambda L: model.encode(torch.tensor(L, dtype=torch.float32)).numpy()
        for n in test_depths:
            fidx = torch.randint(0, 256, (512, 2 ** n))
            fout = model.decode(model.encode(fresh[fidx]), n).reshape(-1, D_PART).numpy()
            idx = torch.randint(0, V, (512, 2 ** n))
            out = model.decode(model.encode(vocab[idx]), n).reshape(-1, D_PART).numpy()
            res.append(dict(cond=cond, w=w, V=V, seed=seed, depth=n, steps=steps, deep_bs=deep_bs,
                            fresh_r2=r2(fout, fresh[fidx].reshape(-1, D_PART).numpy()),
                            fam_r2=r2(out, vocab[idx].reshape(-1, D_PART).numpy()),
                            bound=min(1.0, w / (2 ** n * D_PART)),
                            pop_r2=pop_readout(enc, n, np.random.default_rng(1000 + seed))))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--ws", default="64,256,1024")
    ap.add_argument("--V", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--outdir", default="results/e13")
    ap.add_argument("--deep-bs", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        for w in map(int, a.ws.split(",")):
            for cond in a.conds.split(","):
                fn = os.path.join(a.outdir, f"{cond}_w{w}_V{a.V}_s{seed}.json")
                if os.path.exists(fn):
                    continue
                r = run(cond, w, a.V, seed, a.steps, deep_bs=a.deep_bs)
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(r, f, indent=1)
                print(fn, [(x["depth"], round(x["fresh_r2"], 2), round(x["pop_r2"], 2)) for x in r], flush=True)
