"""E36: test-time harnesses on E9's role questions.  Where does the decomposition live: in the network or in the harness's code?

Items, nouns (fresh nouns of each seed's split), demonstrations (fitting nouns, one per condition) and the rng order are E9's,
so the "direct" condition reproduces E9's choice score.  Three procedures answer the same item:
  direct  E9: the few-shot prompt asks the role question; the answer noun's continuation must beat the other noun's.
  verify  the harness splits the choice into one yes/no check per noun ("Is the cake on top?"), each a separate call with
          its own 4 demonstrations, and returns the noun with the larger log P(yes) - log P(no).  The network still assigns
          the roles; the harness only decomposes the question.
  parse   the network copies the sentence into three fields (first object, relation, second object; 4 demonstrations,
          greedy decoding), and code assigns the roles from a lexicon fixed in advance that lists, for each training
          relation phrase, which mentioned object is the figure.  An unparsed output or a phrase missing from the lexicon
          falls back to the direct answer (the fallback share is reported).
Held-out relation phrases (NEW) test the lexicon itself: atop / beneath, within / holding, struck / was struck by,
higher than / lower than.  They keep E9's questions and take their demonstrations from the same phrases, as E9 does.
usage: e36_harness_lm.py --models a+b --seeds 0+1+2 --outdir results/e36"""
import argparse, json, os, re
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, seq_logprob, N_PAIRS, N_DEMO

NEW = {  # held-out phrases; x is the figure; family -> E9 questions
    "atop": ("a {x} atop a {y}", "a {y} beneath a {x}", "on"),
    "within": ("a {x} within a {y}", "a {y} holding a {x}", "inside"),
    "struck": ("a {x} struck a {y}", "a {y} was struck by a {x}", "hit"),
    "higher": ("a {x} higher than a {y}", "a {y} lower than a {x}", "on"),
}
VERIFY = {"on": ("Is the {n} on top?", "Is the {n} underneath?"), "left": ("Is the {n} on the left?", "Is the {n} on the right?"),
          "inside": ("Is the {n} inside the other?", "Does the {n} hold the other?"), "behind": ("Is the {n} in the back?", "Is the {n} in front?"),
          "hit": ("Did the {n} do the hitting?", "Was the {n} hit?")}
# the harness's lexicon: relation phrase of a TRAINING relation -> index of the figure among (first, second) mentioned object
LEXICON = {"on": 0, "under": 1, "to the left of": 0, "to the right of": 1, "inside": 0, "containing": 1,
           "behind": 0, "in front of": 1, "hit": 0, "was hit by": 1}


def phr_parts(tmpl):
    """'a {y} under a {x}' -> relation words between the two nouns ('under')."""
    m = re.match(r"a \{(\w)\} (.+) a \{(\w)\}$", tmpl)
    return m.group(2)


@torch.no_grad()
def greedy(tok, mod, prompts, dev, max_new=24, bs=16):
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), bs):
        b = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(dev)
        g = mod.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return out


@torch.no_grad()
def yes_minus_no(tok, mod, prompts, dev, bs=32):
    iy, ino = tok(" yes", add_special_tokens=False)["input_ids"][0], tok(" no", add_special_tokens=False)["input_ids"][0]
    out = []
    for s in range(0, len(prompts), bs):
        b = tok(prompts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        lp = torch.log_softmax(mod(**b).logits.float(), -1)
        last = b["attention_mask"].sum(1) - 1
        v = lp[torch.arange(len(last)), last]
        out += (v[:, iy] - v[:, ino]).cpu().tolist()
    return np.array(out)


def fields(sent, n1, rel, n2):
    return f"Sentence: {sent}.\nFirst object: {n1}\nRelation: {rel}\nSecond object: {n2}\n\n"


def run(name, seed, dev, tok, mod):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    rels = [(r, PAIRS[r], r, "train") for r in PAIRS] + [(r, NEW[r][:2], NEW[r][2], "new") for r in NEW]
    res = []
    for rel, templ, fam, kind in rels:
        # E9's draws, in E9's order, for the training relations (E9 draws for its relations only; the new phrases follow)
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        conds = [(p, q) for p in (0, 1) for q in (0, 1)]
        itm = lambda p, q, x, y: (templ[p].format(x=x, y=y), QUESTIONS[fam][q], (x, y)[q], (y, x)[q])
        demo, vdemo, pdemo = [], [], []
        for k, (p, q) in enumerate(conds):
            x, y = dn[2 * k], dn[2 * k + 1]
            s, qq, ans, oth = itm(p, q, x, y)
            demo.append(fmt(s, qq) + f" {ans}\n\n")
            vdemo.append(f"Sentence: {s}.\nQuestion: {VERIFY[fam][q].format(n=(ans if k % 2 == 0 else oth))}\nAnswer: {'yes' if k % 2 == 0 else 'no'}\n\n")
            n1, n2 = (x, y) if p == 0 else (y, x)
            pdemo.append(fields(s, n1, phr_parts(templ[p]), n2))
        perm = rng.permutation(len(demo))
        demo = [demo[i] for i in perm]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo)
        vhead = "Read each sentence and answer the question with yes or no.\n\n" + "".join(vdemo[i] for i in perm)
        phead = "Copy each sentence into its parts.\n\n" + "".join(pdemo[i] for i in perm)
        prs = [(x, y) for x in fr_n for y in fr_n if x != y]
        prs = [prs[i] for i in rng.choice(len(prs), N_PAIRS, replace=False)]
        # parse: one extraction per (pair, phrasing), shared by both questions
        parsed = {}
        for p in (0, 1):
            sents = [templ[p].format(x=x, y=y) for x, y in prs]
            outs = greedy(tok, mod, [phead + f"Sentence: {s}.\nFirst object:" for s in sents], dev)
            for (x, y), o in zip(prs, outs):
                m = re.match(r"\s*(.+?)\s*\nRelation:\s*(.+?)\s*\nSecond object:\s*([^\n]+)", o)
                parsed[(p, x, y)] = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else None
        for p, q in conds:
            its = [itm(p, q, x, y) for x, y in prs]
            prompts = [head + fmt(s, qq) for s, qq, _, _ in its]
            la = seq_logprob(tok, mod, prompts, [" " + a for _, _, a, _ in its], dev)
            lo = seq_logprob(tok, mod, prompts, [" " + o for _, _, _, o in its], dev)
            direct = la > lo
            va = yes_minus_no(tok, mod, [vhead + f"Sentence: {s}.\nQuestion: {VERIFY[fam][q].format(n=a)}\nAnswer:" for s, _, a, _ in its], dev)
            vo = yes_minus_no(tok, mod, [vhead + f"Sentence: {s}.\nQuestion: {VERIFY[fam][q].format(n=o)}\nAnswer:" for s, _, _, o in its], dev)
            verify = va > vo
            pc, fb, okparse, okrel = [], 0, 0, 0
            for (x, y), (s, qq, a, o), d in zip(prs, its, direct):
                f = parsed[(p, x, y)]
                n1, n2 = (x, y) if p == 0 else (y, x)
                if f and f[0] == n1 and f[2] == n2:
                    okparse += 1
                if f and f[1] == phr_parts(templ[p]):
                    okrel += 1
                if f and f[1] in LEXICON and {f[0], f[2]} == {x, y}:
                    fig = (f[0], f[2])[LEXICON[f[1]]]
                    pick = fig if q == 0 else ({f[0], f[2]} - {fig}).pop()
                    pc.append(pick == a)
                else:
                    fb += 1
                    pc.append(bool(d))
            row = dict(model=name, seed=seed, rel=rel, family=fam, kind=kind, phrasing=["canonical", "converse"][p], question=["figure", "ground"][q],
                       direct=float(direct.mean()), verify=float(np.mean(verify)), parse=float(np.mean(pc)), parse_fallback=fb / len(prs),
                       copy_exact=okparse / len(prs), rel_exact=okrel / len(prs), n=len(prs))
            res.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.outdir, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    for name in a.models.split("+"):
        todo = [s for s in map(int, a.seeds.split("+")) if not os.path.exists(os.path.join(a.outdir, f"{name.replace('/', '_')}_s{s}.json"))]
        if not todo:
            continue
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        for seed in todo:
            r = run(name, seed, dev, tok, mod)
            with open(os.path.join(a.outdir, f"{name.replace('/', '_')}_s{seed}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
        del mod; torch.cuda.empty_cache()
