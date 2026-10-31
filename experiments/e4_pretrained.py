"""E4: recoverability in pretrained text encoders.

Composites are sentences that place two nouns in the roles of a relation
("a {x} on a {y}").  A part's own representation is the model's state for
"a {x}".  Readouts from a declared class (ridge) map a composite state to
the role-1 or role-2 part's representation; they are fit on composites of 165
fitting nouns, select the penalty on 40 validation nouns, and are evaluated on
composites of 100 nouns that the readout never saw (fresh parts for the readout;
the model has of course seen every word).  The noun split is drawn once per seed and
shared by the relations; the noun pairs are sampled per relation.
Measures, per model, layer, and relation:
  rec_fresh   R^2 of the part representation on fresh-noun composites
  id_fresh    identification of the part among the 100 fresh nouns
  role_fresh  role-1 readout closer to the role-1 part than to the role-2 part
  rev_sim     cosine(h(x,y), h(y,x)) relative to cosine between unrelated composites
  inter       two-way ANOVA interaction fraction on a 12 x 12 noun grid
  delta_*     role-factorization residual: least-squares fit h = c + B1 T(x) + B2 T(y) on the
              fitting pairs, 1 - R^2 of the state on new pairs ("pairs") and fresh nouns
Options (round 3): --wide extends the penalty grid to 1e9; --codes-from takes the part
codes (targets and references) from another model, so that a random-weight model is
scored against the trained model's codes.  --whiten scales the K_PCA components to unit
variance (each direction of the part code counts equally; without it a single outlier
direction can dominate R^2); --detok first removes from the part code a quadratic function of
the noun's number of subword tokens (fitted on the fitting nouns), a non-lexical feature.
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

NOUNS = """apple bag ball banana basket bed bell bench bicycle bird boat book bottle bowl box bread brick broom bucket
bus cake camera candle car carrot cat chair clock coat cookie cow cup desk dog doll door drum duck egg fan fence fish
flag flower fork fox frog glove goat guitar hammer hat horse jar kettle key ladder lamp leaf lemon lion lock mirror
monkey mouse mug nail necklace onion orange owl pan pear pen pencil piano pig pillow pineapple plate pot pumpkin
rabbit radio ring robot rock rope sandwich scarf shell shoe sock spoon star stone sofa suitcase table teapot tent
tiger tomato towel tractor train tree truck umbrella vase violin wagon watch whale wheel window
anchor ant apron arrow axe badge balloon barrel basin bat bean bear beaver bee beetle belt bin blanket boot bottlecap
bracelet branch brush bubble bull butterfly button cabbage cactus camel can canoe cap cart castle caterpillar chain
cheese cherry chicken chimney cloud coin comb compass cone crab crayon crown cucumber cushion deer dice dish dolphin
donkey dragon drawer eagle elephant envelope feather fern flute frame garlic giraffe glass globe goose grape hammock
harp helmet hen hippo hook horn hose iron jacket jeep kangaroo kite knife koala lantern lettuce lizard lobster magnet
map marble mask medal melon microphone mitten moon moose motorcycle mushroom napkin needle nest net octopus olive
ostrich oven paddle paintbrush panda parrot peach peacock peanut pebble penguin pepper pickle pie pin pipe pizza plane
plum pony potato puppet purse quilt raccoon rake raft rocket rug ruler saddle sailboat saw scissors scooter seal sheep
shield ship shovel skateboard sled slipper snail snake spider sponge squirrel stamp statue stool stove strawberry
submarine sunflower swan sword tablet tack teddy telescope thimble tire toaster toothbrush torch tortoise trophy
trumpet tulip turkey turtle unicorn walnut wallet wand whistle wolf worm yacht zebra""".split()

RELATIONS = {"on": "a {x} on a {y}", "left": "a {x} to the left of a {y}", "and": "a {x} and a {y}",
             "under": "a {x} under a {y}", "inside": "a {x} inside a {y}", "behind": "a {x} behind a {y}"}
CODES = ("phrase", "carrier", "static")  # declared part codes: state of "a x", of "there is a x here", input embedding


def load(name, dev):
    """'rand-<name>' loads the same architecture and tokenizer with random weights (the
    untrained control); the input embedding is then random too."""
    if name.startswith("rand-"):
        tok, mod = load(name[5:], "cpu")
        torch.manual_seed(0)
        mod = type(mod)(mod.config)
        return tok, mod.to(dev).eval()
    from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer, T5EncoderModel
    if name == "clip-b32":
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    elif name == "sd15-clip-l":
        tok = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        mod = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    elif name == "clip-h14":
        tok = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        mod = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    elif name.startswith("t5"):
        tok = AutoTokenizer.from_pretrained(name)
        mod = T5EncoderModel.from_pretrained(name)
    elif "/" in name:  # large decoder LMs (round 7): bf16 on GPU; "org/model@revision" picks a training checkpoint
        repo, _, rev = name.partition("@")
        kw = dict(revision=rev) if rev else {}
        tok = AutoTokenizer.from_pretrained(repo, **kw)
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        mod = AutoModel.from_pretrained(repo, torch_dtype=dt, **kw)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    else:
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    return tok, mod.to(dev).eval()


@torch.no_grad()
def embed(tok, mod, texts, dev, name):
    """Return dict layer -> (N, D) mean-pooled states, plus 'pooled' (CLIP EOS / last token for causal LMs)."""
    out = {}
    for s in range(0, len(texts), 256):
        b = tok(texts[s:s + 256], padding=True, return_tensors="pt").to(dev)
        o = mod(**b, output_hidden_states=True)
        hs = o.hidden_states
        mask = b["attention_mask"].unsqueeze(-1).float()
        if "/" in name and tok.bos_token_id is not None:  # a prepended BOS is an attention sink of large norm whose
            mask[:, 0] *= (b["input_ids"][:, 0] != tok.bos_token_id).float().unsqueeze(-1)  # share of the mean varies with length
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            v = ((hs[li] * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
            out.setdefault(f"L{li}", []).append(v)
        last = b["attention_mask"].sum(1) - 1
        if hasattr(o, "pooler_output") and o.pooler_output is not None and "clip" in name:
            pooled = o.pooler_output
        else:
            pooled = hs[-1][torch.arange(len(last)), last]
        out.setdefault("pooled", []).append(pooled.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def cosmat(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def anova(Hg):
    mu = Hg.mean((0, 1), keepdims=True)
    resid = Hg - Hg.mean(1, keepdims=True) - Hg.mean(0, keepdims=True) + mu
    return float((resid ** 2).sum() / ((Hg - mu) ** 2).sum())


N_FIT, N_VAL, N_FRESH, K_PCA = 165, 40, 100, 16


ALPHAS = np.logspace(-3, 5, 17)
WHITEN, DETOK = False, False


class RidgePath:
    """Ridge readouts for every penalty in ALPHAS from one SVD; the penalty is chosen on
    validation NOUNS (not composites), so that selection is leave-nouns-out."""

    def __init__(self, X, Y):  # X centred; the intercept is the mean target
        self.ym = Y.mean(0)
        try:
            self.U, self.s, self.Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:  # the divide-and-conquer driver fails on some nodes' LAPACK
            import scipy.linalg
            self.U, self.s, self.Vt = scipy.linalg.svd(X, full_matrices=False, lapack_driver="gesvd")
        self.UtY = self.U.T @ (Y - self.ym)
        self.a = None

    def coef(self, a):
        return self.Vt.T @ ((self.s / (self.s ** 2 + a))[:, None] * self.UtY)

    def select(self, Xv, Yv):
        sc = [r2(Xv @ self.coef(a) + self.ym, Yv) for a in ALPHAS]
        self.a = ALPHAS[int(np.argmax(sc))]
        self.W = self.coef(self.a)
        return self

    def predict(self, X):
        return X @ self.W + self.ym


def evaluate(Hs, T, idx, sets, cands):
    """Linear readouts (the declared class) from composite states to each role's part code,
    expressed in K_PCA principal components of the part space (fitted on the fitting nouns).
    Fitted on sentences of fitting nouns, with the penalty chosen on validation nouns;
    evaluated on unseen pairs of fitting nouns ("pairs"), on validation nouns (also used to
    choose layer and code), and on fresh nouns."""
    Hfit = Hs[sets["fit"]]
    mu = Hfit.mean(0)
    row = {}
    for role in (0, 1):
        ytr = T[[idx[p[role]] for p in sets["fit_pairs"]]]
        yv = T[[idx[q[role]] for q in sets["val_pairs"]]]
        reg = RidgePath(Hfit - mu, ytr).select(Hs[sets["val"]] - mu, yv)
        tag = f"role{role + 1}"
        row[f"alpha_{tag}"] = float(reg.a)
        for split in ("pairs", "val", "fresh"):
            prs = sets[f"{split}_pairs"]
            p = reg.predict(Hs[sets[split]] - mu)
            y = T[[idx[q[role]] for q in prs]]
            row[f"rec_{split}_{tag}"] = r2(p, y)
            if split in ("val", "fresh"):
                tgt = np.array([cands[split].index(q[role]) for q in prs])
                C = T[[idx[x] for x in cands[split]]]
                row[f"id_{split}_{tag}"] = float((cosmat(p, C).argmax(1) == tgt).mean())
                yo = T[[idx[q[1 - role]] for q in prs]]
                row[f"role_{split}_{tag}"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
    Hfr, Hrev = Hs[sets["fresh"]], Hs[sets["rev"]]
    c_rev = (cosmat(Hfr, Hrev).diagonal()).mean()
    c_rand = cosmat(Hfr[:200], Hfr[200:400]).mean()
    row["rev_sim"] = float((c_rev - c_rand) / (1.0 - c_rand))
    row["inter"] = anova(Hs[sets["grid"]].reshape(12, 12, -1))
    X = lambda prs: np.concatenate([T[[idx[x] for x, _ in prs]], T[[idx[y] for _, y in prs]]], 1)
    Xf_ = X(sets["fit_pairs"]); xm = Xf_.mean(0)
    Wd, *_ = np.linalg.lstsq(Xf_ - xm, Hfit - mu, rcond=None)
    for split in ("pairs", "fresh"):
        row[f"delta_{split}"] = 1 - r2((X(sets[f"{split}_pairs"]) - xm) @ Wd + mu, Hs[sets[split]])
    return row


def run(name, seed, dev, n_fit=N_FIT, codes_from=None):
    """n_fit < N_FIT fits the readouts on the first n_fit fitting nouns only (the pool-size
    curve); validation and fresh nouns, and the target space, do not change with n_fit."""
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(NOUNS))
    rng.shuffle(nouns)
    fit_n = nouns[:n_fit]
    pca_idx = list(range(N_FIT))  # the part-code PCA is fitted on the 165 fitting nouns only
    val_n = nouns[N_FIT:N_FIT + N_VAL]
    fr_n = nouns[N_FIT + N_VAL:N_FIT + N_VAL + N_FRESH]
    used = nouns
    def part_codes(cname_):
        tk, md = load(cname_, dev)
        cd = {"phrase": embed(tk, md, [f"a {x}" for x in used], dev, cname_),
              "carrier": embed(tk, md, [f"there is a {x} here" for x in used], dev, cname_)}
        with torch.no_grad():  # static input embedding, mean over the word's subword tokens
            W = md.get_input_embeddings().weight
            st = np.stack([W[torch.tensor(tk(x, add_special_tokens=False)["input_ids"])].mean(0).float().cpu().numpy() for x in used])
        cd["static"] = {layer: st for layer in cd["phrase"]}
        return tk, md, cd
    tok, mod, codes = part_codes(name)
    ctok = tok
    if codes_from:  # targets and references from another model's codes (same layers)
        ctok, _, codes = part_codes(codes_from)
    ntok = np.array([len(ctok(x, add_special_tokens=False)["input_ids"]) for x in used], float)
    XT = np.c_[np.ones(len(used)), ntok, ntok ** 2]
    idx = {x: i for i, x in enumerate(used)}
    res = []
    for rel, tmpl in RELATIONS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        allfit = sample(fit_n, 3800)
        n_tr = min(3000, int(0.8 * len(allfit)))
        fit_pairs, pairs_pairs = allfit[:n_tr], allfit[n_tr:]
        val_pairs, fresh_pairs = sample(val_n, 600), sample(fr_n, 1000)
        grid = [(x, y) for x in fit_n[:12] for y in fit_n[:12]]
        rev_pairs = [(y, x) for x, y in fresh_pairs]
        order = [("fit", fit_pairs), ("pairs", pairs_pairs), ("val", val_pairs), ("fresh", fresh_pairs), ("grid", grid), ("rev", rev_pairs)]
        sets, allp, s0 = {}, [], 0
        for k, prs in order:
            sets[k] = slice(s0, s0 + len(prs)); s0 += len(prs); allp += prs
            sets[f"{k}_pairs"] = prs
        H = embed(tok, mod, [tmpl.format(x=x, y=y) for x, y in allp], dev, name)
        cands = {"val": val_n, "fresh": fr_n}
        for layer, Hall in H.items():
            for cname in CODES:
                P = codes[cname][layer]
                if DETOK:  # remove the token-count component, fitted on the fitting nouns
                    bt, *_ = np.linalg.lstsq(XT[pca_idx], P[pca_idx], rcond=None)
                    P = P - XT @ bt
                pmu = P[pca_idx].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca_idx] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:K_PCA].T
                if WHITEN:
                    T = T / (sv[:K_PCA] / np.sqrt(len(pca_idx)))
                A = P[[idx[x] for x, _ in allp]]
                B = P[[idx[y] for _, y in allp]]
                systems = {"model": Hall, "ref_concat": np.concatenate([A, B], 1), "ref_sum": A + B}
                for sname, Hs in systems.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname, dim=int(Hs.shape[1]),
                               n_fit=n_fit, n_fit_pairs=len(fit_pairs), codes_from=codes_from or name, alpha_max=float(ALPHAS[-1]),
                               whiten=WHITEN, detok=DETOK)
                    row.update(evaluate(Hs, T, idx, sets, cands))
                    res.append(row)
                print(name, rel, layer, cname, {k: round(v, 3) for k, v in res[-3].items()
                                                 if k.startswith("rec_") and k.endswith("role2")}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip-b32")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nfit", type=int, default=N_FIT)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--codes-from", default=None)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--detok", action="store_true")
    a = ap.parse_args()
    WHITEN, DETOK = a.whiten, a.detok
    if a.wide:
        ALPHAS = np.logspace(-3, 9, 25)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            cf = f"_codes-{a.codes_from}" if a.codes_from else ""
            fn = os.path.join(a.outdir, f"{name.replace('/', '_')}{cf}_n{a.nfit}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev, a.nfit, a.codes_from)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
