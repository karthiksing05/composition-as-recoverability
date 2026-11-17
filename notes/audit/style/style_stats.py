"""Prose statistics of the reference papers against ours (main body only, before references/appendix).
Writes each body text to notes/audit/style/<name>.txt and prints a comparison table."""
import re, sys, collections, statistics as st
import pymupdf

OUT = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/style"
DOCS = {
    "CobwebDiff": "/coc/flash10/ksingara3/_ACS_26__Cobweb_Diffusion.pdf",
    "TRELLIS": "/coc/flash10/ksingara3/_ACS_26__TRELLIS.pdf",
    "ICLR": "/coc/flash10/ksingara3/_ICLR_2027__Diffusion_Composition_Generalization.pdf",
    "OURS": "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.pdf",
}


def body(path):
    d = pymupdf.open(path)
    paras = []
    for p in d:
        for b in p.get_text("blocks"):
            t = b[4].replace("-\n", "").replace("\n", " ").strip()
            t = re.sub(r"\s+", " ", t)
            if len(t) < 120 or re.fullmatch(r"[\d\s.]+", t):
                continue  # headers, numbers, figure ticks, line numbers
            if t.startswith(("Figure", "Table", "Under review", "Anonymous")):
                continue
            paras.append(t)
    txt = "\n".join(paras)
    m = re.search(r"\n(References|REFERENCES|Bibliography)\b", txt)
    return txt[: m.start()] if m else txt, paras


PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?(\w+ed|built|made|shown|given|taken|known|held|found|seen|done|drawn|written|chosen|set|kept|left|run|read|split|fit)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|suggests?|appears?|seems?|likely|potentially|possibly|perhaps|arguably)\b", re.I)
SLOP = re.compile(r"\b(crucial|crucially|notably|importantly|key|novel|robust|leverage|leverages|delve|landscape|pivotal|comprehensive|seamless|intricate|nuanced|underscores?|highlights?|showcases?|furthermore|moreover|additionally|in summary|overall)\b", re.I)
CONNECT = ["therefore", "so", "because", "while", "however", "thus", "then", "which", "and", "but", "whereas", "instead", "yet"]


def stats(name, txt, paras):
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", txt.replace("\n", " ")) if len(s.split()) > 3]
    L = [len(s.split()) for s in sents]
    words = re.findall(r"[A-Za-z']+", txt.lower())
    n = len(words)
    cnt = collections.Counter(words)
    per1k = lambda k: 1000 * cnt[k] / n
    plen = [len(p.split()) for p in paras if len(p.split()) > 25]
    r = dict(name=name, words=n, sents=len(L), mean_len=st.mean(L), median_len=st.median(L),
             p90_len=sorted(L)[int(.9 * len(L))], over40=sum(x > 40 for x in L) / len(L),
             under12=sum(x < 12 for x in L) / len(L),
             para_mean=st.mean(plen) if plen else 0,
             passive_per_sent=len(PASSIVE.findall(txt)) / len(L), hedge_1k=1000 * len(HEDGE.findall(txt)) / n,
             slop_1k=1000 * len(SLOP.findall(txt)) / n, we_1k=per1k("we") + per1k("our"),
             commas_per_sent=txt.count(",") / len(L), paren_1k=1000 * txt.count("(") / n,
             semicolon_1k=1000 * txt.count(";") / n, colon_1k=1000 * txt.count(":") / n,
             ttr=len(set(words[:5000])) / min(5000, n))
    for c in CONNECT:
        r["c_" + c] = per1k(c)
    starts = collections.Counter(" ".join(s.split()[:1]) for s in sents)
    r["top_starts"] = ", ".join(f"{w}:{100*c/len(L):.0f}%" for w, c in starts.most_common(8))
    return r, sents


rows = {}
for name, path in DOCS.items():
    txt, paras = body(path)
    open(f"{OUT}/{name}.txt", "w", encoding="utf-8").write(txt)
    rows[name], sents = stats(name, txt, paras)
    open(f"{OUT}/{name}_sents.txt", "w", encoding="utf-8").write("\n".join(sents))
keys = [k for k in rows["OURS"] if k not in ("name", "top_starts")]
print("%-18s" % "metric" + "".join("%12s" % n for n in rows))
for k in keys:
    print("%-18s" % k + "".join("%12.2f" % rows[n][k] for n in rows))
for n in rows:
    print(n, "starts:", rows[n]["top_starts"])
