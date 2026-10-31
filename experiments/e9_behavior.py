"""E9: behavioral role binding in language models, on the nouns and phrasings of E7.

E4/E6/E7 read a model's state with fitted readouts (representation).  Here the model itself
answers: a few-shot prompt gives a sentence and a question about a semantic role that names
neither noun ("Which object is on top?"), and we score the model's own continuation.  Each
relation keeps E7's canonical and converse phrasings, so the answer is the first-mentioned noun in
half of the prompts and the second-mentioned noun in the other half, and a word-order shortcut
scores 0.5 overall.  The demonstrations use fitting nouns; the test sentences use the 100 fresh
nouns of the seed's split (the nouns E4/E7 readouts never saw).  Measures, per relation and
condition (phrasing x question):
  choice  the answer noun's full continuation is more probable than the other noun's
  ident   the answer's first token beats the first tokens of all 100 fresh nouns (ties share credit)
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS

QUESTIONS = {  # relation: (question answered by the figure, question answered by the ground)
    "on": ("Which object is on top?", "Which object is underneath?"),
    "left": ("Which object is on the left?", "Which object is on the right?"),
    "inside": ("Which object is inside the other?", "Which object holds the other?"),
    "behind": ("Which object is in the back?", "Which object is in front?"),
    "hit": ("Which object did the hitting?", "Which object was hit?"),
}
N_PAIRS, N_DEMO = 200, 4


def item(rel, x, y, phr, q):
    """x is the figure.  Returns (sentence, question, answer, other, answer mentioned first)."""
    sent = PAIRS[rel][phr].format(x=x, y=y)
    ans, oth = (x, y) if q == 0 else (y, x)
    return sent, QUESTIONS[rel][q], ans, oth, (phr == 0) == (q == 0)


def fmt(sent, q):
    return f"Sentence: {sent}.\nQuestion: {q}\nAnswer: the"


@torch.no_grad()
def seq_logprob(tok, mod, prompts, conts, dev, bs=32):
    """Sum of log-probabilities of each continuation given its prompt."""
    out = []
    for s in range(0, len(prompts), bs):
        P = [tok(p)["input_ids"] for p in prompts[s:s + bs]]
        C = [tok(c, add_special_tokens=False)["input_ids"] for c in conts[s:s + bs]]
        seqs = [p + c for p, c in zip(P, C)]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q); att[i, :len(q)] = 1
        lp = torch.log_softmax(mod(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(), -1)
        for i, (p, c) in enumerate(zip(P, C)):
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1)
            out.append(float(lp[i, pos, torch.tensor(c)].sum()))
    return np.array(out)


@torch.no_grad()
def first_token_ident(tok, mod, prompts, answers, cands, dev, bs=32):
    first = {n: tok(" " + n, add_special_tokens=False)["input_ids"][0] for n in cands}
    ids_c = torch.tensor([first[n] for n in cands])
    out = []
    for s in range(0, len(prompts), bs):
        b = tok(prompts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        lg = mod(**b).logits.float()
        last = b["attention_mask"].sum(1) - 1
        sc = lg[torch.arange(len(last)), last][:, ids_c.to(dev)].cpu().numpy()
        for row, a in zip(sc, answers[s:s + bs]):
            t = row[cands.index(a)]
            top = row.max()
            out.append(0.0 if t < top else 1.0 / (row == top).sum())
    return np.array(out)


def run(name, seed, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    repo, _, rev = name.partition("@")
    kw = dict(revision=rev) if rev else {}
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    mod = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, **kw).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    res = []
    for rel in PAIRS:
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        conds = [(p, q) for p in (0, 1) for q in (0, 1)]
        demo = []
        for k, (p, q) in enumerate(conds):  # one demonstration per condition, in a shuffled order
            s, qq, a, _, _ = item(rel, dn[2 * k], dn[2 * k + 1], p, q)
            demo.append(fmt(s, qq) + f" {a}\n\n")
        demo = [demo[i] for i in rng.permutation(len(demo))]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo)
        prs = [(x, y) for x in fr_n for y in fr_n if x != y]
        prs = [prs[i] for i in rng.choice(len(prs), N_PAIRS, replace=False)]
        for p, q in conds:
            its = [item(rel, x, y, p, q) for x, y in prs]
            prompts = [head + fmt(s, qq) for s, qq, *_ in its]
            la = seq_logprob(tok, mod, prompts, [" " + a for _, _, a, _, _ in its], dev)
            lo = seq_logprob(tok, mod, prompts, [" " + o for _, _, _, o, _ in its], dev)
            ident = first_token_ident(tok, mod, prompts, [a for _, _, a, _, _ in its], fr_n, dev)
            row = dict(model=name, seed=seed, rel=rel, phrasing=["canonical", "converse"][p], question=["figure", "ground"][q],
                       answer_first=bool(its[0][4]), choice=float((la > lo).mean()), ident=float(ident.mean()),
                       margin=float((la - lo).mean()))
            res.append(row)
            print(name, rel, row["phrasing"], row["question"], round(row["choice"], 3), round(row["ident"], 3), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
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
