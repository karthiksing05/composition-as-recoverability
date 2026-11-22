"""E35 text side (E31 --part a on the caption encoders of the transformer denoisers).
A ridge readout of the figure noun's part code from the caption state, fit on canonical captions of fitting nouns, read on
canonical (control) and converse captions of held-out nouns; 3 split seeds; same nouns, pair draws, targets (PCA-16 of the
"a x" states of fitting nouns) and scores as e31_state.part_a, which ran this on SD 1.5's CLIP-L.
  pixart-t5     T5-XXL encoder of PixArt-Sigma (the denoiser reads its last layer, L24)
  lumina-gemma  Gemma-2-2B of Lumina-Image-2.0, prompted with the pipeline's system prompt (the denoiser reads L25 of 26);
                the state is the mean over the caption's own tokens, after the system prompt
State = mean over the caption's tokens at layers L/4, L/2, 3L/4, L-1 and L.  Output results/e31/d2a_text_<encoder>.json."""
import argparse, json
import numpy as np, torch
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True, choices=["pixart-t5", "lumina-gemma"])
a = ap.parse_args()
dev = "cuda"
SYS = None
if a.enc == "pixart-t5":
    from transformers import T5EncoderModel, AutoTokenizer
    R = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    tok = AutoTokenizer.from_pretrained(R, subfolder="tokenizer")
    mod = T5EncoderModel.from_pretrained(R, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
else:
    from transformers import AutoModel, AutoTokenizer
    R = "Alpha-VLLM/Lumina-Image-2.0"
    tok = AutoTokenizer.from_pretrained(R, subfolder="tokenizer")
    mod = AutoModel.from_pretrained(R, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    SYS = ("You are an assistant designed to generate superior images with the superior degree of image-text alignment based on "
           "textual prompts or user prompts. <Prompt Start> ")
    tok.padding_side = "right"


@torch.no_grad()
def embed(texts):
    out = {}
    n0 = 0
    if SYS:
        n0 = len(tok(SYS.rstrip())["input_ids"])  # tokens of the system prompt (with BOS); the caption's tokens follow
        texts = [SYS + t for t in texts]
    for s in range(0, len(texts), 256):
        b = tok(texts[s:s + 256], padding=True, return_tensors="pt").to(dev)
        hs = mod(**b, output_hidden_states=True).hidden_states
        mask = b["attention_mask"].unsqueeze(-1).float()
        mask[:, :n0] = 0
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L - 1, L}):
            v = ((hs[li].float() * mask).sum(1) / mask.sum(1)).cpu().numpy()
            out.setdefault(f"L{li}", []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


out = []
for seed in (0, 1, 2):
    sp = split(seed)
    fit_n, val_n, fr_n, allN = sp["fit"], sp["val"], sp["fresh"], sp["all"]
    idx = {x: i for i, x in enumerate(allN)}
    Pc = embed([f"a {x}" for x in allN])
    rng = np.random.default_rng(seed + 1000)
    for rel, (canon, conv, axis) in RELS.items():
        def samp(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fit_p, val_p, fr_p = samp(fit_n, 3000), samp(val_n, 600), samp(fr_n, 1000)
        H = {}
        for nm, prs, tm in (("fit", fit_p, canon), ("val", val_p, canon), ("new_canon", fr_p, canon), ("new_conv", fr_p, conv)):
            H[nm] = embed([tm.format(x=x, y=y) for x, y in prs])
        for layer in Pc:
            P = Pc[layer]
            pidx = [idx[x] for x in fit_n]
            pmu = P[pidx].mean(0)
            _, sv, Vt = np.linalg.svd(P[pidx] - pmu, full_matrices=False)
            T = (P - pmu) @ Vt[:e4.K_PCA].T
            mu = H["fit"][layer].mean(0)
            tgt = lambda prs, j: T[[idx[p[j]] for p in prs]]
            reg = e4.RidgePath(H["fit"][layer] - mu, tgt(fit_p, 0)).select(H["val"][layer] - mu, tgt(val_p, 0))
            C = T[[idx[x] for x in fr_n]]
            tg = np.array([fr_n.index(p[0]) for p in fr_p])
            row = dict(enc=a.enc, seed=seed, rel=rel, layer=layer, alpha=float(reg.a), val_r2=e4.r2(reg.predict(H["val"][layer] - mu), tgt(val_p, 0)))
            for nm in ("new_canon", "new_conv"):
                p = reg.predict(H[nm][layer] - mu)
                y, yo = tgt(fr_p, 0), tgt(fr_p, 1)
                row[f"{nm}_r2_figure"] = e4.r2(p, y)
                row[f"{nm}_share_nearer_figure"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
                row[f"{nm}_id_figure"] = float((e4.cosmat(p, C).argmax(1) == tg).mean())
                tg2 = np.array([fr_n.index(p_[1]) for p_ in fr_p])
                row[f"{nm}_id_ground"] = float((e4.cosmat(p, C).argmax(1) == tg2).mean())
            out.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump(out, open(f"{R31}/d2a_text_{a.enc}.json", "w", encoding="utf-8"), indent=1)
