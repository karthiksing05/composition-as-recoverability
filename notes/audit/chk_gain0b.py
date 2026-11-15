import json, sys
import numpy as np
import torch
sys.path.insert(0, "/coc/flash10/ksingara3/compositionality-theory/experiments")
import e2_glyph as E
import e20_entangled as Z
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"
for s in (0, 1, 2):
    a = json.load(open(R + f"e2redec/mlp_m64_V8_s{s}.json", encoding="utf-8"))["orig"]
    print("stored e2redec s", s, {k: round(v, 3) for k, v in a.items()}, flush=True)
seed, steps = 0, 4000
c = E.train_model("mlp", 64, 8, seed, steps)
model, world = c["model"], c["world"]
prng = np.random.default_rng(20_000 + seed)
prng.normal(size=(400, E.Q)); prng.integers(0, 400, (2000, 2))
fb = prng.normal(size=(256, E.Q))
fp = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1); fp = np.concatenate([fp, fp[:, ::-1]])
za, zb = fb[fp[:, 0]], fb[fp[:, 1]]
for rin in (0, 1):
    with torch.no_grad():
        h = model.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32))
        for o in (0, 1):
            out = model.decode(h, torch.full((len(h),), o, dtype=torch.long)).numpy()
            print("fresh train_model", f"r{rin}{o}", round(E.r2(out, world.render(za, zb, o)), 3), flush=True)
print("train_model final loss", c["loss"], "vocab head", np.round(c["vocab"][0, :3], 3), "train head", c["train"][:3].tolist())
r = Z.run("mlp", 64, 8, seed, steps, 0, 1, redecode=False)
print("e20 run gain 0", {k: round(r[k], 3) for k in ("own_heldout", "own_newpairs", "read_trained", "read_input", "final_loss")})
