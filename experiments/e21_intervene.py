"""E21: does the question state cause the model's answer?

Across models, the figure readout of the state at the end of the question tracks answer accuracy
(r = 0.94).  This intervenes.  For a target prompt about "a x on a y" (figure x), we run a source
prompt with the same two nouns and patch the target's state at the final token, at one layer:
  reversed   "a x under a y"   same order of mention, figure y
  swapped    "a y on a x"      other order, figure y
  samefig    "a y under a x"   other order, figure x (a state of meaning leaves the answer alone)
Patches move h_t to h_t + (h_s - h_t) P, for P the projector onto:
  whole     the entire state
  readout   the row space of the figure readout (16 dimensions), fitted on canonical sentences of fitting nouns
  pc        the top 16 principal components of the fitting states (a high-variance control)
  random    a random 16 dimensions within the top 200 components (a low-specificity control)
Two source-free interventions:
  ablate_*  sets the readout's or the random subspace to its mean on the fitting rows
  write     adds the minimum-norm change that makes the readout return the other noun's code
Answers compare the first tokens of the two nouns at the final position; pairs that share a first token
are dropped.  All test nouns are held out from the readout.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO
from e11_tokens import embed_query

PICK = {"microsoft/phi-4": (30, "phrase"), "Qwen/Qwen3-8B": (27, "carrier"), "allenai/OLMo-2-1124-13B": (30, "phrase"),
        "allenai/OLMo-2-1124-7B": (16, "carrier"), "gpt2-large": (18, "carrier"), "EleutherAI/pythia-1.4b": (12, "carrier"),
        "EleutherAI/pythia-12b": (18, "carrier")}
N_FIT, N_VAL, N_TEST, K = 1500, 300, 200, 16


def locate(mod):
    if hasattr(mod, "gpt_neox"):
        return mod.gpt_neox.layers, mod.gpt_neox.final_layer_norm
    if hasattr(mod, "transformer"):
        return mod.transformer.h, mod.transformer.ln_f
    return mod.model.layers, mod.model.norm


@torch.no_grad()
def forward(tok, mod, prompts, dev, a=None, b=None, patch=None, li=None, bs=24):
    """Logit margin of noun a over noun b (first tokens) at the last token; patch is (N, D) replacing the
    state at hidden layer li of the final token.  Without a, returns nothing but the states."""
    blocks, norm = locate(mod)
    target = norm if patch is not None and li == len(blocks) else (blocks[li - 1] if patch is not None else None)
    out = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        last = enc["attention_mask"].sum(1) - 1
        ix = torch.arange(len(last), device=dev)
        hnd = None
        if patch is not None:
            new = torch.tensor(patch[s:s + bs], device=dev)

            def hook(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                t = t.clone(); t[ix, last] = new.to(t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            hnd = target.register_forward_hook(hook)
        lg = mod(**enc).logits.float()[ix, last]
        if hnd:
            hnd.remove()
        out.append((lg[ix, torch.tensor(a[s:s + bs], device=dev)] - lg[ix, torch.tensor(b[s:s + bs], device=dev)]).cpu().numpy())
    return np.concatenate(out)


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pick_layer, cname = PICK[name]
    codes = E.embed(tok, mod, [(f"a {x}" if cname == "phrase" else f"there is a {x} here") for x in nouns], dev, name)
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    first = lambda x: tok(" " + x, add_special_tokens=False)["input_ids"][0]
    res = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp, tp = sample(fit_n, N_FIT), sample(val_n, N_VAL), sample(fr_n, N_TEST * 2)
        tp = [p for p in tp if first(p[0]) != first(p[1])][:N_TEST]
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        q = QUESTIONS[rel][0]
        P = lambda t: head + fmt(t, q)
        Hfit = embed_query(tok, mod, [P(canon.format(x=x, y=y)) for x, y in fitp], dev)
        Hval = embed_query(tok, mod, [P(canon.format(x=x, y=y)) for x, y in valp], dev)
        ptarget = [P(canon.format(x=x, y=y)) for x, y in tp]
        psrc = {"reversed": ([P(conv.format(x=y, y=x)) for x, y in tp], 1),  # "a x under a y": figure y
                "swapped": ([P(canon.format(x=y, y=x)) for x, y in tp], 1),  # "a y on a x": figure y
                "samefig": ([P(conv.format(x=x, y=y)) for x, y in tp], 0)}   # "a y under a x": figure x
        Ht = embed_query(tok, mod, ptarget, dev)
        Hs = {k: embed_query(tok, mod, v[0], dev) for k, v in psrc.items()}
        ax = np.array([first(x) for x, _ in tp]); ay = np.array([first(y) for _, y in tp])
        m0 = forward(tok, mod, ptarget, dev, ax, ay)
        src_margin = {k: forward(tok, mod, psrc[k][0], dev, ax, ay) for k in psrc}
        for (_, li) in Hfit:
            Pc = codes[f"L{li}"]
            pmu = Pc[pca].mean(0)
            _, sv, Vt = np.linalg.svd(Pc[pca] - pmu, full_matrices=False)
            Tc = (Pc - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
            F_, V_ = Hfit[("query", li)], Hval[("query", li)]
            mu = F_.mean(0)
            reg = E.RidgePath(F_ - mu, Tc[[idx[x] for x, _ in fitp]]).select(V_ - mu, Tc[[idx[x] for x, _ in valp]])
            W = reg.W  # D x K: code = (h - mu) W + ym
            B = np.linalg.svd(W, full_matrices=False)[0]
            _, _, Vf = np.linalg.svd(F_ - mu, full_matrices=False)
            pcB = Vf[:K].T
            g = np.random.default_rng(1000 + seed)
            nb = min(200, Vf.shape[0])
            rB = np.linalg.qr(Vf[:nb].T @ g.normal(size=(nb, K)))[0]
            subs = {"readout": B, "pc": pcB, "random": rB}
            h0 = Ht[("query", li)]
            pinvW = np.linalg.pinv(W)
            pred = (h0 - mu) @ W + reg.ym
            conds = []
            for sname, (_, ans) in psrc.items():
                hs = Hs[sname][("query", li)]
                conds.append((sname, "whole", hs, ans))
                for kn, Bk in subs.items():
                    conds.append((sname, kn, h0 + (hs - h0) @ Bk @ Bk.T, ans))
            for kn in ("readout", "random"):
                Bk = subs[kn]
                conds.append(("none", f"ablate_{kn}", h0 + (mu - h0) @ Bk @ Bk.T, 0))
            Ty = Tc[[idx[y] for _, y in tp]]
            conds.append(("none", "write", h0 + (Ty - pred) @ pinvW, 1))
            conds.append(("none", "noop", h0.copy(), 0))
            base = dict(model=name, seed=seed, rel=rel, layer=li, picked=li == pick_layer, code=cname, n=len(tp),
                        orig_acc=float((m0 > 0).mean()), orig_margin=float(m0.mean()), alpha=float(reg.a))
            for sname, kind, hn, ans in conds:
                mp = forward(tok, mod, ptarget, dev, ax, ay, patch=hn.astype(np.float32), li=li)
                toward = (mp > 0) if ans == 0 else (mp < 0)
                row = dict(base, source=sname, kind=kind, to_answer=float(toward.mean()), margin=float(mp.mean()),
                           kept_correct=float((mp > 0).mean()), noop_err=float(np.abs(mp - m0).mean()) if kind == "noop" else None)
                if sname in src_margin:
                    sm = src_margin[sname]
                    row["src_acc"] = float(((sm > 0) if ans == 0 else (sm < 0)).mean())
                res.append(row)
            print(name, rel, li, "orig", round(base["orig_acc"], 3),
                  {f"{r['source'][:3]}-{r['kind']}": round(r["to_answer"], 2) for r in res[-len(conds):]}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e21")
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
