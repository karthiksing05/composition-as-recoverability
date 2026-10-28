"""E4: recoverability in pretrained text encoders.

Composites are sentences that place two nouns in the roles of a relation
("a {x} on a {y}").  A part's own representation is the model's state for
"a {x}".  Readouts from a declared class (ridge, MLP) map a composite state to
the role-1 or role-2 part's representation; they are fit on composites of 80
training nouns and evaluated on composites of 40 nouns that the readout never
saw (fresh parts for the readout; the model has of course seen every word).
Measures, per model, layer, and relation:
  rec_fresh   R^2 of the part representation on fresh-noun composites
  id_fresh    identification of the part among the 40 fresh nouns
  role_fresh  role-1 readout closer to the role-1 part than to the role-2 part
  rev_sim     cosine(h(x,y), h(y,x)) relative to cosine between unrelated composites
  inter       two-way ANOVA interaction fraction on a 12 x 12 noun grid
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

RELATIONS = {"on": "a {x} on a {y}", "left": "a {x} to the left of a {y}", "and": "a {x} and a {y}"}


def load(name, dev):
    from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer, T5EncoderModel
    if name == "clip-b32":
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    elif name == "sd15-clip-l":
        tok = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        mod = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    elif name.startswith("t5"):
        tok = AutoTokenizer.from_pretrained(name)
        mod = T5EncoderModel.from_pretrained(name)
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
        L = len(hs) - 1
        for li in sorted({L // 2, L}):
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


N_TRAIN, N_FRESH, K_PCA = 170, 80, 32


def evaluate(Hall, P, idx, tr_pairs, fr_pairs, fr_n, n1, n2, n3, pca):
    """Linear readouts (the declared class) from composite states to each role's part,
    in the part space reduced to K_PCA principal components fitted on training nouns."""
    Htr, Hfr, Hg, Hrev = Hall[:n1], Hall[n1:n1 + n2], Hall[n1 + n2:n1 + n2 + n3], Hall[n1 + n2 + n3:]
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    z = lambda h: (h - mu) / sd
    T = pca(P)
    row = {}
    fr_cands = T[[idx[x] for x in fr_n]]
    for role in (0, 1):
        ytr = T[[idx[p[role]] for p in tr_pairs]]
        yfr = T[[idx[p[role]] for p in fr_pairs]]
        yother = T[[idx[p[1 - role]] for p in fr_pairs]]
        reg = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(z(Htr), ytr)
        p, ptr = reg.predict(z(Hfr)), reg.predict(z(Htr[:500]))
        tag = f"role{role + 1}"
        row[f"rec_train_{tag}"] = r2(ptr, ytr[:500])
        row[f"rec_fresh_{tag}"] = r2(p, yfr)
        tgt = np.array([fr_n.index(pp[role]) for pp in fr_pairs])
        row[f"id_fresh_{tag}"] = float((cosmat(p, fr_cands).argmax(1) == tgt).mean())
        row[f"role_fresh_{tag}"] = float((((p - yfr) ** 2).sum(1) < ((p - yother) ** 2).sum(1)).mean())
    c_rev = (cosmat(Hfr, Hrev).diagonal()).mean()
    c_rand = cosmat(Hfr[:200], Hfr[200:400]).mean()
    row["rev_sim"] = float((c_rev - c_rand) / (1.0 - c_rand))
    row["inter"] = anova(Hg.reshape(12, 12, -1))
    return row


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(NOUNS))
    rng.shuffle(nouns)
    tr_n, fr_n = nouns[:N_TRAIN], nouns[N_TRAIN:N_TRAIN + N_FRESH]
    used = nouns[:N_TRAIN + N_FRESH]
    tok, mod = load(name, dev)
    parts = embed(tok, mod, [f"a {x}" for x in used], dev, name)
    idx = {x: i for i, x in enumerate(used)}
    res = []
    for rel, tmpl in RELATIONS.items():
        tr_pairs = [(x, y) for x in tr_n for y in tr_n if x != y]
        tr_pairs = [tr_pairs[i] for i in rng.choice(len(tr_pairs), 3000, replace=False)]
        fr_pairs = [(x, y) for x in fr_n for y in fr_n if x != y]
        fr_pairs = [fr_pairs[i] for i in rng.choice(len(fr_pairs), 1000, replace=False)]
        g = tr_n[:12]
        grid = [(x, y) for x in g for y in g]
        texts = [tmpl.format(x=x, y=y) for x, y in tr_pairs + fr_pairs + grid + [(y, x) for x, y in fr_pairs]]
        H = embed(tok, mod, texts, dev, name)
        n1, n2, n3 = len(tr_pairs), len(fr_pairs), len(grid)
        all_pairs = tr_pairs + fr_pairs + grid + [(y, x) for x, y in fr_pairs]
        for layer, Hall in H.items():
            P = parts[layer]
            Ptr = P[[idx[x] for x in tr_n]]
            pmu = Ptr.mean(0)
            _, _, Vt = np.linalg.svd(Ptr - pmu, full_matrices=False)
            pca = lambda X, Vt=Vt, pmu=pmu: (X - pmu) @ Vt[:K_PCA].T
            # reference operators applied to the model's own part representations
            A = P[[idx[x] for x, _ in all_pairs]]
            B = P[[idx[y] for _, y in all_pairs]]
            systems = {"model": Hall, "ref_concat": np.concatenate([A, B], 1), "ref_sum": A + B, "ref_weighted": A + 0.5 * B}
            for sname, Hs in systems.items():
                row = dict(model=name, seed=seed, rel=rel, layer=layer, system=sname, dim=int(Hs.shape[1]))
                row.update(evaluate(Hs, P, idx, tr_pairs, fr_pairs, fr_n, n1, n2, n3, pca))
                res.append(row)
                print(name, rel, layer, sname, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip-b32")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
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
