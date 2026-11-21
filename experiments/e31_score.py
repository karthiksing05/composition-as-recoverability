"""GroundingDINO detection of each named noun separately (one query per noun, per image).
Stores, per image and noun, the best box (cx,cy,w,h normalised) and its score (max sigmoid logit over the noun's tokens).
The box threshold is applied afterwards in e31_analyze.py. Resumable (jsonl)."""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)  # path to an index json (list of records)
ap.add_argument("--imgdir", default=None)  # default results/e31/images/<model>
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
mod = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda").eval()
recs = json.load(open(a.index, encoding="utf-8"))
done = set()
if os.path.exists(a.out):
    for l in open(a.out, encoding="utf-8"):
        done.add(json.loads(l)["file"])
todo = [r for r in recs if r["file"] not in done]
print(len(recs), "records", len(todo), "to do", flush=True)


def nouns_of(r):
    return [r["x"]] if r["kind"] == "single" else [r["x"], r["y"]]


@torch.no_grad()
def detect(imgs, words):
    """imgs: list of PIL (repeated per query), words: list of nouns."""
    inp = proc(images=imgs, text=[f"{w}." for w in words], return_tensors="pt", padding=True).to("cuda")
    o = mod(**inp)
    sc = o.logits.sigmoid()  # (B, 900, T)
    res = []
    for i, w in enumerate(words):
        ids = inp["input_ids"][i]
        # noun tokens = all tokens except [CLS] (first), '.' and [SEP]
        keep = [j for j in range(len(ids)) if inp["attention_mask"][i, j] and j > 0 and j < int(inp["attention_mask"][i].sum()) - 2]
        s = sc[i][:, keep].max(-1).values
        k = int(s.argmax())
        res.append(dict(score=float(s[k]), box=[float(v) for v in o.pred_boxes[i, k]]))
    return res


mode = "a"
with open(a.out, mode, encoding="utf-8") as fo:
    for b in range(0, len(todo), a.bs):
        batch = todo[b:b + a.bs]
        imgs, words, owner = [], [], []
        for ri, r in enumerate(batch):
            dirm = a.imgdir or f"{R31}/images/{r['model']}"
            im = Image.open(f"{dirm}/{r['file']}.png").convert("RGB")
            for w in nouns_of(r):
                imgs.append(im); words.append(w); owner.append(ri)
        out = detect(imgs, words)
        det = [dict() for _ in batch]
        for ri, w, d in zip(owner, words, out):
            det[ri][w] = d
        for r, d in zip(batch, det):
            fo.write(json.dumps(dict(file=r["file"], det=d)) + "\n")
        fo.flush()
        if (b // a.bs) % 20 == 0:
            print(b, "/", len(todo), flush=True)
