"""Case studies for the language-model figure: individual held-out sentences, relation "on", seed 0.

Reruns the E19/E11 ridge figure readout (same random draws, same layer and part code per model as
E11/E19/E21 picked) and the E9 answers, and saves, besides the aggregates, 8 fixed pairs of
held-out nouns in canonical ("a x on a y") and converse ("a y under a x") phrasing, with:
  answer    the model's log-probabilities of " x" and " y" after the question-state prompt
            (E19's four demonstrations, the sentence, "Which object is on top?", "Answer: the")
  query     the ridge figure readout of the question state: the nearer of the figure and ground
            part codes, with both Euclidean distances in the whitened 16-PC code space
  mean      the same readout class fitted on the token-averaged sentence state at the same layer
Aggregates:
  xfer_query / xfer_mean   transfer on the 500 converse held-out test sentences (E19 rows)
  ans_e9                   E9's protocol replicated for "on" (its own draws; 200 pairs x 4 conditions)
  ans_qprompt              answers to the figure question on the 1000 E19 test sentences
No existing script or result is modified.  --summary writes notes/audit/case_lm_summary.md.
"""
import argparse, glob, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO, N_PAIRS, seq_logprob
from e11_tokens import spans, embed_tokens, embed_query
from e21_intervene import PICK

REL = "on"
CASES = [("book", "bench"), ("cake", "plate"), ("lamp", "rug"), ("leaf", "rock"),
         ("key", "map"), ("hat", "sofa"), ("cherry", "pie"), ("basket", "tractor")]
HEADLINE = {  # paper (means over relations and 3 seeds): question-state transfer, answer accuracy
    "microsoft/phi-4": (0.98, 0.99), "Qwen/Qwen3-8B": (0.87, None), "allenai/OLMo-2-1124-7B": (0.20, 0.77),
    "EleutherAI/pythia-1.4b": (0.085, 0.50), "gpt2-large": (0.085, 0.50)}
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_lm(name, dev):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    # E4/E11 load "org/model" in bf16 and bare names (gpt2-large) in fp32; keep that for the states
    dt = torch.bfloat16 if (dev == "cuda" and "/" in name) else torch.float32
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok, mod


def e9_answers(tok, mod, name, seed, dev):
    """E9's draws and scoring for the first relation ("on") only."""
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
    conds = [(p, q) for p in (0, 1) for q in (0, 1)]
    demo = []
    for k, (p, q) in enumerate(conds):
        s, qq, a, _, _ = item(REL, dn[2 * k], dn[2 * k + 1], p, q)
        demo.append(fmt(s, qq) + f" {a}\n\n")
    demo = [demo[i] for i in rng.permutation(len(demo))]
    head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo)
    prs = [(x, y) for x in fr_n for y in fr_n if x != y]
    prs = [prs[i] for i in rng.choice(len(prs), N_PAIRS, replace=False)]
    out = {}
    for p, q in conds:
        its = [item(REL, x, y, p, q) for x, y in prs]
        prompts = [head + fmt(s, qq) for s, qq, *_ in its]
        la = seq_logprob(tok, mod, prompts, [" " + a for _, _, a, _, _ in its], dev)
        lo = seq_logprob(tok, mod, prompts, [" " + o for _, _, _, o, _ in its], dev)
        out[f"{['canonical', 'converse'][p]}_{['figure', 'ground'][q]}"] = float((la > lo).mean())
    out["all"] = float(np.mean(list(out.values())))
    return out


def run(name, seed, dev):
    assert list(PAIRS)[0] == REL  # the E19 draws below are those of the first relation
    rng = np.random.default_rng(seed)  # the same draws as e19_nonlinear.run / e11_tokens.run
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    for x, y in CASES:
        assert x in fr_n and y in fr_n, (x, y)
    tok, mod = load_lm(name, dev)
    li, cname = PICK[name]
    # states from the decoder without the LM head (as E4/E11 load AutoModel); the head is used only for answers
    P = E.embed(tok, mod.base_model, [(f"a {x}" if cname == "phrase" else f"there is a {x} here") for x in nouns], dev, name)[f"L{li}"]
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    pmu = P[pca].mean(0)
    _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
    T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
    canon, conv = PAIRS[REL]

    def sample(ns, k):
        prs = [(x, y) for x in ns for y in ns if x != y]
        return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
    fitp, valp, frp = sample(fit_n, 3000), sample(val_n, 600), sample(fr_n, 1000)
    texts, meta, sps = [], [], []
    for split, prs in (("fit", fitp), ("val", valp), ("fresh", frp)):
        for i, (x, y) in enumerate(prs):
            ph = i % 2
            tm = canon if ph == 0 else conv
            texts.append(tm.format(x=x, y=y)); meta.append((split, ph, x, y)); sps.append(spans(tm, x, y))
    dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
    demo = [fmt(*item(REL, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(REL, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
            for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
    head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
    for x, y in CASES:
        for ph, tm in ((0, canon), (1, conv)):
            texts.append(tm.format(x=x, y=y)); meta.append(("case", ph, x, y)); sps.append(spans(tm, x, y))
    # keep only the rows the ridge readout uses: canonical fitting/validation, all test and case rows
    keep = [i for i, m in enumerate(meta) if m[0] in ("fresh", "case") or m[1] == 0]
    texts, meta, sps = [texts[i] for i in keep], [meta[i] for i in keep], [sps[i] for i in keep]
    qprompts = [head + fmt(t, QUESTIONS[REL][0]) for t in texts]
    H = embed_tokens(tok, mod.base_model, texts, sps, dev)
    H.update(embed_query(tok, mod.base_model, qprompts, dev))
    sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
    fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
    sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
    Fc, Vc = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0)
    tests = {"can": sel(sp == "fresh", ph == 0), "conv": sel(sp == "fresh", ph == 1)}
    case_rows = sel(sp == "case")
    T_of = lambda rows, which: T[[idx[(fig if which == 0 else gnd)[i]] for i in rows]]
    agg, pred = {}, {}
    for state in ("query", "mean"):
        X = H[(state, li)]
        Xc = X - X[Fc].mean(0)
        reg = E.RidgePath(Xc[Fc], T_of(Fc, 0)).select(Xc[Vc], T_of(Vc, 0))
        for k, r in tests.items():
            p = reg.predict(Xc[r])
            agg[f"xfer_{state}_{k}"] = float((((p - T_of(r, 0)) ** 2).sum(1) < ((p - T_of(r, 1)) ** 2).sum(1)).mean())
        agg[f"ridge_alpha_{state}"] = float(reg.a)
        pred[state] = reg.predict(Xc[case_rows])
    # answers from the question-state prompt on the 1000 test sentences and the case sentences
    rows = np.r_[tests["can"], tests["conv"], case_rows]
    la = seq_logprob(tok, mod, [qprompts[i] for i in rows], [" " + fig[i] for i in rows], dev)
    lo = seq_logprob(tok, mod, [qprompts[i] for i in rows], [" " + gnd[i] for i in rows], dev)
    nc, nv = len(tests["can"]), len(tests["conv"])
    agg["ans_qprompt_can"] = float((la[:nc] > lo[:nc]).mean())
    agg["ans_qprompt_conv"] = float((la[nc:nc + nv] > lo[nc:nc + nv]).mean())
    agg["ans_qprompt_all"] = float((la[:nc + nv] > lo[:nc + nv]).mean())
    agg["ans_e9"] = e9_answers(tok, mod, name, seed, dev)
    cases = []
    for j, i in enumerate(case_rows):
        lf, lg = float(la[nc + nv + j]), float(lo[nc + nv + j])
        c = dict(figure=fig[i], ground=gnd[i], phrasing=["canonical", "converse"][ph[i]], sentence=texts[i],
                 question=QUESTIONS[REL][0], first_mentioned=fig[i] if ph[i] == 0 else gnd[i],
                 answer=dict(preferred=fig[i] if lf > lg else gnd[i], logp_figure=lf, logp_ground=lg, margin_figure_minus_ground=lf - lg))
        for state in ("query", "mean"):
            p = pred[state][j]
            df, dg = float(np.linalg.norm(p - T[idx[fig[i]]])), float(np.linalg.norm(p - T[idx[gnd[i]]]))
            c[f"readout_{state}"] = dict(returned=fig[i] if df < dg else gnd[i], dist_figure=df, dist_ground=dg)
        cases.append(c)
    return dict(model=name, seed=seed, rel=REL, layer=f"L{li}", code=cname, n_test_can=nc, n_test_conv=nv,
                prompt_head=head, aggregates=agg, headline=dict(zip(("xfer_query", "answers"), HEADLINE.get(name, (None, None)))),
                reference=reference(name, seed, li, cname), cases=cases)


def reference(name, seed, li, cname):
    """The stored E11 (query, mean) and E9 numbers for this relation and seed, for an exact check."""
    m = name.replace("/", "_")
    out = {}
    fn = os.path.join(ROOT, "results", "e11", f"{m}_s{seed}.json")
    if os.path.exists(fn):
        for r in json.load(open(fn, encoding="utf-8")):
            if r["system"] == "model" and r["rel"] == REL and r["layer"] == f"L{li}" and r["code"] == cname and r["state"] in ("query", "mean"):
                out[f"e11_xfer_{r['state']}_conv"] = r["xfer_role_conv"]
    fn = os.path.join(ROOT, "results", "e9", f"{m}_s{seed}.json")
    if os.path.exists(fn):
        d = json.load(open(fn, encoding="utf-8"))
        out["e9_choice_on"] = float(np.mean([r["choice"] for r in d if r["rel"] == REL]))
        out["e9_choice_allrels"] = float(np.mean([r["choice"] for r in d]))
    return out


def summary(outdir, path):
    L = ["# Case studies, language models (relation \"on\", seed 0)", "",
         "Written by experiments/case_lm.py --summary from experiments/results/case_lm/*.json.",
         "Headlines are means over five relations and 3 seeds; these runs are one relation and one seed,",
         "so the stored E11/E9 values for the same relation and seed are the exact check (`ref`).",
         "Flag = |rerun - headline| > 0.1.", ""]
    for fn in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        d = json.load(open(fn, encoding="utf-8"))
        a, h, ref = d["aggregates"], d["headline"], d["reference"]
        L.append(f"## {d['model']} (layer {d['layer']}, part code {d['code']})")
        L.append("")
        L.append("| measure | rerun | ref (stored, on, s0) | headline | flag |")
        L.append("|---|---|---|---|---|")
        hq, ha = h["xfer_query"], h["answers"]
        fq = "" if hq is None else ("**>0.1**" if abs(a["xfer_query_conv"] - hq) > 0.1 else "ok")
        fa = "" if ha is None else ("**>0.1**" if abs(a["ans_e9"]["all"] - ha) > 0.1 else "ok")
        f2 = lambda v: "-" if v is None else f"{v:.3f}"
        L.append(f"| question-state transfer (conv) | {a['xfer_query_conv']:.3f} | {f2(ref.get('e11_xfer_query_conv'))} | {f2(hq)} | {fq} |")
        L.append(f"| token-averaged transfer (conv) | {a['xfer_mean_conv']:.3f} | {f2(ref.get('e11_xfer_mean_conv'))} | ~0 | |")
        L.append(f"| answer accuracy, E9 protocol (on) | {a['ans_e9']['all']:.3f} | {f2(ref.get('e9_choice_on'))} | {f2(ha)} | {fa} |")
        L.append(f"| answer accuracy, figure question on test sentences | {a['ans_qprompt_all']:.3f} (can {a['ans_qprompt_can']:.2f}, conv {a['ans_qprompt_conv']:.2f}) | | | |")
        L.append("")
        L.append("| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |")
        L.append("|---|---|---|---|")
        for c in d["cases"]:
            an, q, m = c["answer"], c["readout_query"], c["readout_mean"]
            L.append(f"| {c['sentence']} | {an['preferred']} ({an['margin_figure_minus_ground']:+.2f}) | "
                     f"{q['returned']} ({q['dist_figure']:.2f} / {q['dist_ground']:.2f}) | {m['returned']} |")
        L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt2-large+EleutherAI/pythia-1.4b+allenai/OLMo-2-1124-7B+microsoft/phi-4+Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "case_lm"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "..", "notes", "audit", "case_lm_summary.md"))
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)  # as in e19_nonlinear / e11_tokens
    E.WHITEN = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models.split("+"):
        fn = os.path.join(a.outdir, f"{name.replace('/', '_')}.json")
        if os.path.exists(fn):
            continue
        r = run(name, a.seed, dev)
        print(name, json.dumps(r["aggregates"]), json.dumps(r["reference"]), flush=True)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        del r
        torch.cuda.empty_cache()
    summary(a.outdir, os.path.abspath(a.summary))
