"""E24: the input baseline for pretrained models.

In the glyph task the input pixels recover held-out parts at 0.91, above any network's state.  The analogue
for a sentence: a state built from the model's input embeddings alone, with no layer applied.
  input_concat  the input embeddings of the two nouns concatenated (the input keeps roles by position)
  input_avg     the average of input embeddings over the sentence (bag of words, plus position
                embeddings where the model adds them to its first hidden state)
Both are scored by E4's pipeline: same splits, whitened 16-PC part codes, penalty on validation nouns,
concatenation and sum references, and the model's own token-averaged state alongside, at every layer and code.
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E

ASYM = ["on", "left", "under", "inside", "behind"]


@torch.no_grad()
def input_states(tok, mod, texts, dev):
    W = mod.get_input_embeddings()
    out = []
    for s in range(0, len(texts), 256):
        b = tok(texts[s:s + 256], padding=True, return_tensors="pt").to(dev)
        mask = b["attention_mask"].unsqueeze(-1).float()
        h0 = mod(**b, output_hidden_states=True).hidden_states[0]
        out.append(((h0 * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = E.load(name, dev)
    codes = {"phrase": E.embed(tok, mod, [f"a {x}" for x in nouns], dev, name),
             "carrier": E.embed(tok, mod, [f"there is a {x} here" for x in nouns], dev, name)}
    with torch.no_grad():
        Wt = mod.get_input_embeddings().weight
        static = np.stack([Wt[torch.tensor(tok(" " + x, add_special_tokens=False)["input_ids"], device=Wt.device)].mean(0).float().cpu().numpy() for x in nouns])
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    res = []
    for rel in ASYM:
        tmpl = E.RELATIONS[rel]
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        allfit = sample(fit_n, 3800)
        fit_pairs, pairs_pairs = allfit[:3000], allfit[3000:]
        val_pairs, fresh_pairs = sample(val_n, 600), sample(fr_n, 1000)
        grid = [(x, y) for x in fit_n[:12] for y in fit_n[:12]]
        rev_pairs = [(y, x) for x, y in fresh_pairs]
        sets, allp, s0 = {}, [], 0
        for k, prs in (("fit", fit_pairs), ("pairs", pairs_pairs), ("val", val_pairs), ("fresh", fresh_pairs), ("grid", grid), ("rev", rev_pairs)):
            sets[k] = slice(s0, s0 + len(prs)); s0 += len(prs); allp += prs
            sets[f"{k}_pairs"] = prs
        texts = [tmpl.format(x=x, y=y) for x, y in allp]
        H = E.embed(tok, mod, texts, dev, name)
        Hav = input_states(tok, mod, texts, dev)
        Hcat = np.c_[static[[idx[x] for x, _ in allp]], static[[idx[y] for _, y in allp]]]
        cands = {"val": val_n, "fresh": fr_n}
        for layer, Hall in H.items():
            if layer == "pooled":
                continue
            for cname in ("phrase", "carrier"):
                P = codes[cname][layer]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A, B = P[[idx[x] for x, _ in allp]], P[[idx[y] for _, y in allp]]
                systems = {"model": Hall, "input_concat": Hcat, "input_avg": Hav, "ref_concat": np.c_[A, B], "ref_sum": A + B}
                for sname, Hs in systems.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname)
                    row.update(E.evaluate(Hs, T, idx, sets, cands))
                    res.append(row)
                print(name, rel, layer, cname, {r["system"]: round((r["rec_fresh_role1"] + r["rec_fresh_role2"]) / 2, 3) for r in res[-5:]}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e24")
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
