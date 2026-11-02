"""E18: the behavior of text-only encoders on the sentences whose states E7 reads.

A text encoder's own procedure is the similarity of its sentence embeddings, the use that
retrieval and paraphrase matching make of it: the projected end-of-text state for CLIP, the
pooled state for the Stable Diffusion text encoder (it has no projection), and the mean of the
last layer for T5 and BERT.  For a fresh pair (x, y) of E7's fresh nouns and an anchor sentence
in one phrasing (canonical "a x on a y" or converse "a y under a x"):
  roles   the anchor is closer to its paraphrase in the other phrasing than to the paraphrase with
          the two nouns' roles swapped.  The two candidates contain the same words, so a bag of
          words scores 0.5, and the swapped candidate mentions the nouns in the anchor's order,
          so matching the order of mention scores 0 and matching meaning scores 1
  order   the same choice within one phrasing, against the reversal ("the x on the y" against
          "the y on the x" for the anchor "a x on a y"): sensitivity to the order of mention
  identity  the anchor is closer to its paraphrase than to the paraphrase with the second noun
          replaced by another fresh noun (the control: the encoder tells the nouns apart)
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS

MODELS = ("clip-b32", "sd15-clip-l", "clip-h14", "t5-small", "t5-base", "t5-large", "bert-base-uncased", "bert-large-uncased")
PROJ = {"clip-b32": "openai/clip-vit-base-patch32", "clip-h14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
N_PAIRS = 1000


def load(name, dev):
    if name in PROJ:
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        return CLIPTokenizer.from_pretrained(PROJ[name]), CLIPTextModelWithProjection.from_pretrained(PROJ[name]).to(dev).eval()
    return E.load(name, dev)


@torch.no_grad()
def sentence_emb(tok, mod, texts, dev, name):
    out = []
    for s in range(0, len(texts), 256):
        b = tok(texts[s:s + 256], padding=True, return_tensors="pt").to(dev)
        o = mod(**b)
        if name in PROJ:
            e = o.text_embeds
        elif name.startswith("sd15"):
            e = o.pooler_output
        else:
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (o.last_hidden_state * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def run(name, seed, dev):
    rng = np.random.default_rng(seed)  # E7's noun split: the same draws as e7_roles.run
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fr_n = nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    tok, mod = load(name, dev)
    rng2 = np.random.default_rng(20_000 + seed)
    res = []
    for rel, (canon, conv) in PAIRS.items():
        tpl = (canon, conv)
        the = tuple(t.replace("a {", "the {") for t in tpl)
        xs = rng2.choice(fr_n, N_PAIRS); ys = np.array([rng2.choice([n for n in fr_n if n != x]) for x in xs])
        zs = np.array([rng2.choice([n for n in fr_n if n not in (x, y)]) for x, y in zip(xs, ys)])
        ph = np.arange(N_PAIRS) % 2  # anchor phrasing: 0 canonical, 1 converse
        f = lambda t, x, y: t.format(x=x, y=y)
        anchor = [f(tpl[p], x, y) for p, x, y in zip(ph, xs, ys)]
        para = [f(tpl[1 - p], x, y) for p, x, y in zip(ph, xs, ys)]
        swap = [f(tpl[1 - p], y, x) for p, x, y in zip(ph, xs, ys)]
        same = [f(the[p], x, y) for p, x, y in zip(ph, xs, ys)]
        rev = [f(the[p], y, x) for p, x, y in zip(ph, xs, ys)]
        wrong = [f(tpl[1 - p], x, z) for p, x, z in zip(ph, xs, zs)]
        A, P, S, M, R, W = [sentence_emb(tok, mod, t, dev, name) for t in (anchor, para, swap, same, rev, wrong)]
        sp, ss, sm, sr, sw = [(A * B).sum(1) for B in (P, S, M, R, W)]
        res.append(dict(model=name, seed=seed, rel=rel, roles=float((sp > ss).mean()),
                        roles_can=float((sp > ss)[ph == 0].mean()), roles_conv=float((sp > ss)[ph == 1].mean()),
                        order=float((sm > sr).mean()), identity=float((sp > sw).mean()),
                        margin_roles=float((sp - ss).mean()), margin_order=float((sm - sr).mean())))
        print(res[-1], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--outdir", default="results/e18")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split(","):
        for seed in map(int, a.seeds.split(",")):
            fn = os.path.join(a.outdir, f"{name}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
