"""E5: structural consequences of role-factorized recoverability in trained networks.

For each trained system (same world and training as E2) we declare a part code e(x):
the top 16 principal components of glyph pixels, fitted on 4,000 glyphs sampled from
the part distribution (a property of the parts, not of the system).  We fit the
role-factorized model h = c + B1 e(a) + B2 e(b) by ridge on composites of a pool of
parts the system never saw, read the codes back with a ridge readout fitted on the
same pool, and test, on composites of FRESH glyphs:
  delta      residual fraction of the fit (Definition 1, Eq. 2)
  reversal   predict h(b, a) from the single state h(a, b) via Prop. 6(ii); compare the
             predicted state and its decoded output with the true reversal
  newrole    place a glyph seen only in role 1 into role 2 via Prop. 6(iii)
  swap       interchange intervention: replace the role-1 code of h(a, b) with that of a
             donor h(a', b) through B1 and decode BESIDE; success = left half shows a',
             right half keeps b; controls of the same norm: a random direction, a random
             direction inside range(B1), the role-2 interchange, and a natural direction
             h(a', b') - h(a, b); each scored against the original and donor glyphs
  newpart    fine-tune on 5 composites with a new glyph in role 1 only (with/without replay);
             measure the new glyph in role 1 and role 2 and forgetting on held-out pairs
"""
import argparse, json, os, copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, RidgeCV
import e2_glyph as E

K = 16


def r2(p, t):
    p, t = p.reshape(len(p), -1), t.reshape(len(t), -1)
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def run(arch, m, V, seed, steps, second):
    E.SECOND["rel"] = second
    ctx = E.train_model(arch, m, V, seed, steps)
    model, world, vocab, fresh, train, heldout, strong = (ctx[k] for k in
                                                          ("model", "world", "vocab", "fresh", "train", "heldout", "strong"))
    rng = np.random.default_rng(10_000 + seed)
    # declared part code: PCA of glyph pixels over the part distribution
    G = world.glyph(rng.normal(size=(4000, E.Q))).reshape(4000, -1)
    gmu = G.mean(0)
    _, _, Vt = np.linalg.svd(G - gmu, full_matrices=False)
    code = lambda z: (world.glyph(z).reshape(len(z), -1) - gmu) @ Vt[:K].T

    def enc(za, zb, rin):
        with torch.no_grad():
            return model.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32)).numpy()

    def dec(h, rout):
        with torch.no_grad():
            return model.decode(torch.tensor(h, dtype=torch.float32), torch.full((len(h),), rout, dtype=torch.long)).numpy()

    res = dict(arch=arch, m=m, V=V, seed=seed, second=second, K=K)
    nf = len(fresh)
    fa, fb = fresh[:nf // 2], fresh[nf // 2:]
    # role maps are fitted on composites of a pool of parts the system never trained on
    # (population fit), so that the fit is not limited by the span of the training parts
    pool = rng.normal(size=(400, E.Q))
    pp = rng.integers(0, 400, (2000, 2))
    G2 = G - gmu
    res["code_var"] = float((G2 @ Vt[:K].T).var(0).sum() / G2.var(0).sum())  # glyph variance in the K-dim code
    maps = {}
    for rin in (0, 1):
        za, zb = pool[pp[:, 0]], pool[pp[:, 1]]
        H = enc(za, zb, rin)
        X = np.concatenate([code(za), code(zb)], 1)
        fit = Ridge(alpha=1e-3).fit(X, H)
        c, B = fit.intercept_, fit.coef_                      # B: m x 2K
        B1, B2 = B[:, :K], B[:, K:]
        # readout of the stacked codes, fitted on the same pool composites; a regularized
        # inverse of B (the exact pseudo-inverse is unstable when B is nearly square)
        U = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(H, X)
        rd = lambda h: (U.predict(h)[:, :K], U.predict(h)[:, K:])
        # delta on fresh composites
        Hf = enc(fa, fb, rin)
        Xf = np.concatenate([code(fa), code(fb)], 1)
        res[f"delta_fresh_r{rin}"] = 1 - r2(fit.predict(Xf), Hf)
        res[f"delta_fit_r{rin}"] = 1 - r2(fit.predict(X), H)
        res[f"readout_fresh_r{rin}"] = r2(U.predict(Hf), Xf)
        # reversal from one composite (Prop. 6 ii): h_rev = c + B1 e(b) + B2 e(a)
        Hrev_true = enc(fb, fa, rin)
        e1, e2 = rd(Hf)
        Hrev_pred = c + e2 @ B1.T + e1 @ B2.T
        Hrev_edit = Hf + (e2 - e1) @ (B1 - B2).T            # same prediction, keeping the residual
        res[f"rev_state_r{rin}"] = r2(Hrev_pred, Hrev_true)
        res[f"rev_state_edit_r{rin}"] = r2(Hrev_edit, Hrev_true)
        res[f"rev_state_noswap_r{rin}"] = r2(Hf, Hrev_true)
        for rout in (0, 1):
            tgt = world.render(fb, fa, rout)
            res[f"rev_out_pred_r{rin}{rout}"] = r2(dec(Hrev_pred, rout), tgt)
            res[f"rev_out_edit_r{rin}{rout}"] = r2(dec(Hrev_edit, rout), tgt)
            res[f"rev_out_true_r{rin}{rout}"] = r2(dec(Hrev_true, rout), tgt)
            res[f"rev_out_noswap_r{rin}{rout}"] = r2(dec(Hf, rout), tgt)
        # new role assignment (Prop. 6 iii): parts never seen in role 2, placed there using
        # their code read from a composite (x, y) and the role-2 map
        Hset = np.unique(strong[:, 1])
        n_nr = 256  # sample many composites so that R^2 is not computed on V/4 points
        x = vocab[Hset[rng.integers(0, len(Hset), n_nr)]]
        y = vocab[rng.integers(0, V, n_nr)]
        u = fresh[rng.integers(0, nf, n_nr)]
        v = fresh[rng.integers(0, nf, n_nr)]
        h_xy, h_uv = enc(x, y, rin), enc(u, v, rin)
        h_ux_pred = c + rd(h_uv)[0] @ B1.T + rd(h_xy)[0] @ B2.T
        h_ux_true = enc(u, x, rin)
        res[f"newrole_state_r{rin}"] = r2(h_ux_pred, h_ux_true)
        tgt = world.render(u, x, 0)
        res[f"newrole_out_pred_r{rin}"] = r2(dec(h_ux_pred, 0), tgt)
        res[f"newrole_out_net_r{rin}"] = r2(dec(h_ux_true, 0), tgt)
        # interchange intervention on role 1: replace its code by a donor's, through B1
        a2 = fresh[rng.permutation(nf)[:nf // 2]]
        b2 = fresh[rng.permutation(nf)[:nf // 2]]
        h_a2 = enc(a2, b2, rin)  # donor shares neither part, so leakage of its role-2 part is not hidden
        h_sw = Hf + (rd(h_a2)[0] - e1) @ B1.T
        o, o0, otrue = dec(h_sw, 0), dec(Hf, 0), dec(enc(a2, fb, rin), 0)
        ga2, gb = world.glyph(a2), world.glyph(fb)
        res[f"swap_left_r{rin}"] = r2(o[:, :, :8], ga2)
        res[f"swap_right_r{rin}"] = r2(o[:, :, 8:], gb)
        res[f"swap_left_base_r{rin}"] = r2(otrue[:, :, :8], ga2)   # the network's own output for (a2, b)
        res[f"swap_right_base_r{rin}"] = r2(o0[:, :, 8:], gb)      # unswapped output, right half
        res[f"swap_right_change_r{rin}"] = float(np.abs(o[:, :, 8:] - o0[:, :, 8:]).mean() / (np.abs(o[:, :, :8] - o0[:, :, :8]).mean() + 1e-9))
        # controls, each an edit of the same norm as the role-1 edit; every output is scored
        # against the original glyphs (a, b) and the donor a2
        dlt = (rd(h_a2)[0] - e1) @ B1.T
        nrm = np.linalg.norm(dlt, axis=1, keepdims=True)
        unit = lambda v: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        ga = world.glyph(fa)
        edits = {
            "swap": dlt,                                                       # role-1 interchange
            "rand": unit(rng.normal(size=dlt.shape)) * nrm,                    # random direction
            "randB1": unit(rng.normal(size=(len(dlt), K)) @ B1.T) * nrm,       # random direction inside range(B1)
            "swapB2": (rd(h_a2)[1] - e2) @ B2.T,                               # role-2 interchange (donor b2)
            "natural": unit(enc(a2, b2, rin) - Hf) * nrm,                      # in-distribution, role-agnostic
        }
        for tag, E_ in edits.items():
            oe = dec(Hf + E_, 0)
            res[f"ctl_{tag}_left_orig_r{rin}"] = r2(oe[:, :, :8], ga)
            res[f"ctl_{tag}_right_orig_r{rin}"] = r2(oe[:, :, 8:], gb)
            res[f"ctl_{tag}_left_donor_r{rin}"] = r2(oe[:, :, :8], ga2)
            res[f"ctl_{tag}_right_donor_r{rin}"] = r2(oe[:, :, 8:], world.glyph(b2))
            res[f"ctl_{tag}_dleft_r{rin}"] = float(np.abs(oe[:, :, :8] - o0[:, :, :8]).mean())
            res[f"ctl_{tag}_dright_r{rin}"] = float(np.abs(oe[:, :, 8:] - o0[:, :, 8:]).mean())
        res[f"swap_rand_left_r{rin}"] = res[f"ctl_rand_left_donor_r{rin}"]
        # round-3 controls, on a separate stream so that the edits above are unchanged:
        # randB1cov  a direction inside range(B1) with coefficients from the code distribution
        #            (difference of two random fresh codes), not isotropic coefficients
        # natural1   the network's own change when only the role-1 glyph changes, h(a2, b) - h(a, b)
        rng2 = np.random.default_rng(50_000 + seed + 7 * rin)
        a3, a4 = rng2.normal(size=(2, len(dlt), E.Q))
        extra = {"randB1cov": unit((code(a3) - code(a4)) @ B1.T) * nrm,
                 "natural1": unit(enc(a2, fb, rin) - Hf) * nrm,
                 "natural1raw": enc(a2, fb, rin) - Hf}
        for tag, E_ in extra.items():
            oe = dec(Hf + E_, 0)
            res[f"ctl_{tag}_dleft_r{rin}"] = float(np.abs(oe[:, :, :8] - o0[:, :, :8]).mean())
            res[f"ctl_{tag}_dright_r{rin}"] = float(np.abs(oe[:, :, 8:] - o0[:, :, 8:]).mean())
        maps[rin] = (c, B1, B2, U)  # U itself: rd closes over the loop variable
    # new part, few-shot, with and without replay; control: the same fine-tuning on a
    # training glyph in role 1 (ordinary forgetting from fine-tuning on five composites)
    znew = rng.normal(size=(1, E.Q))
    zold = vocab[rng.integers(0, V, 1)]
    partners = vocab[rng.integers(0, V, 5)]
    # variant 3: the new glyph in both roles (3 composites in role 1, 2 in role 2)
    for replay in (0, 1, 2, 3):
        z1 = zold if replay == 2 else znew
        mdl = copy.deepcopy(model)
        mdl.train()
        opt = torch.optim.Adam(mdl.parameters(), lr=3e-4)
        new_pairs = (np.repeat(z1, 5, 0), partners)
        if replay == 3:
            zz = np.repeat(znew, 5, 0)
            new_pairs = (np.concatenate([zz[:3], partners[3:]]), np.concatenate([partners[:3], zz[3:]]))
        for step in range(200):
            rin = step % 2
            xs = [torch.tensor(world.render(*new_pairs, rin), dtype=torch.float32)]
            ys = [torch.tensor(world.render(*new_pairs, rin), dtype=torch.float32)]
            routs = [torch.full((5,), rin, dtype=torch.long)]
            if replay == 1:
                idx = rng.integers(0, len(train), 32)
                xs.append(torch.tensor(world.render(vocab[train[idx, 0]], vocab[train[idx, 1]], rin), dtype=torch.float32))
                ys.append(xs[-1]); routs.append(torch.full((32,), rin, dtype=torch.long))
            xb, yb, rb = torch.cat(xs), torch.cat(ys), torch.cat(routs)
            loss = F.mse_loss(mdl.decode(mdl.encode(xb), rb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval()

        def ev(mm, za, zb):
            with torch.no_grad():
                h = mm.encode(torch.tensor(world.render(za, zb, 0), dtype=torch.float32))
                return r2(mm.decode(h, torch.zeros(len(za), dtype=torch.long)).numpy(), world.render(za, zb, 0))
        others = vocab[rng.integers(0, V, 32)]
        zn = np.repeat(z1, 32, 0)
        tag = {2: "oldctl", 3: "bothroles"}.get(replay, f"replay{replay}")
        for which, mm in (("before", model), ("after", mdl)):
            if which == "before" and replay >= 1:
                continue
            t = "before" if which == "before" else tag
            res[f"newpart_role1_{t}"] = ev(mm, zn, others)
            res[f"newpart_role2_{t}"] = ev(mm, others, zn)
            res[f"newpart_heldout_{t}"] = ev(mm, vocab[heldout[:256, 0]], vocab[heldout[:256, 1]])
    # round-3: before-values for the familiar-glyph control, and the no-gradient addition of
    # the new glyph to role 2: read its code from one composite (znew, partner) and write it
    # through the role-2 map; old pairs are untouched by construction
    with torch.no_grad():
        def evm(za, zb):
            h = model.encode(torch.tensor(world.render(za, zb, 0), dtype=torch.float32))
            return r2(model.decode(h, torch.zeros(len(za), dtype=torch.long)).numpy(), world.render(za, zb, 0))
        zo = np.repeat(zold, 32, 0)
        res["newpart_role1_before_old"] = evm(zo, others)
        res["newpart_role2_before_old"] = evm(others, zo)
    c0, B10, B20, U0 = maps[0]
    rd0 = lambda h: (U0.predict(h)[:, :K], U0.predict(h)[:, K:])
    zn = np.repeat(znew, 32, 0)
    e_new = rd0(enc(zn[:1], partners[:1], 0))[0]              # code read from one composite
    h_add = c0 + rd0(enc(others, others[::-1], 0))[0] @ B10.T + e_new @ B20.T
    res["newpart_role2_addcode"] = r2(dec(h_add, 0), world.render(others, zn, 0))
    h_ref = c0 + rd0(enc(others, others[::-1], 0))[0] @ B10.T + rd0(enc(zn, others, 0))[0] @ B20.T
    res["newpart_role2_addcode_avg"] = r2(dec(h_ref, 0), world.render(others, zn, 0))  # code from 32 composites
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="slot+mlp+cnn+tf")
    ap.add_argument("--ms", default="32+64")
    ap.add_argument("--Vs", default="16+64")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--second", default="over")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split("+")):
        for V in map(int, a.Vs.split("+")):
            for arch in a.archs.split("+"):
                for m in map(int, a.ms.split("+")):
                    fn = os.path.join(a.outdir, f"{arch}_m{m}_V{V}_{a.second}_s{seed}.json")
                    if os.path.exists(fn):
                        continue
                    r = run(arch, m, V, seed, a.steps, a.second)
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump(r, f, indent=1)
                    print(fn, {k: round(v, 3) for k, v in r.items() if isinstance(v, float) and ("r1" in k or "newpart" in k)}, flush=True)
