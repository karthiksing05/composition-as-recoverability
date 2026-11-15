import json, urllib.request, urllib.parse, time, difflib
Q = """Compositionality Partee 1984
Compositionality Janssen 1997 Handbook of Logic and Language
On the proper treatment of connectionism Smolensky
The constituent structure of connectionist mental states Smolensky
Constituent structure and explanation in an integrated connectionist/symbolic cognitive architecture Smolensky
The Harmonic Mind Smolensky Legendre
Connectionism and cognitive architecture Fodor Pylyshyn
Connectionism and the problem of systematicity Fodor McLaughlin
Mapping part-whole hierarchies into connectionist networks Hinton
Tensor product variable binding Smolensky
Implementing semantic networks in parallel hardware Hinton 1981
Learning distributed representations of concepts Hinton 1986
Hyperdimensional computing: an introduction to computing in distributed representation Kanerva
Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience Gayler
Finding structure in time Elman
Distributed representations, simple recurrent networks, and grammatical structure Elman
On language and connectionism: analysis of a parallel distributed processing model Pinker Prince
The Algebraic Mind Marcus
Syntactic Structures Chomsky
Building machines that learn and think like people Lake
Linguistic generalization and compositionality in modern artificial neural networks Baroni
Representation learning: a review and new perspectives Bengio
Challenging common assumptions in the unsupervised learning of disentangled representations Locatello
beta-VAE learning basic visual concepts Higgins
Probing classifiers: promises, shortcomings, and advances Belinkov
A feature-integration theory of attention Treisman Gelade
Recursive deep models for semantic compositionality over a sentiment treebank Socher
Systematicity in connectionist language learning Hadley
Composition in distributional models of semantics Mitchell Lapata
Catastrophic interference in connectionist networks McCloskey Cohen
Catastrophic forgetting in connectionist networks French
On the binding problem in artificial neural networks Greff
Rule learning by seven-month-old infants Marcus
Concepts and compositionality: in search of the brain's language of thought Frankland Greene
Human-level concept learning through probabilistic program induction Lake""".split("\n")
for q in Q:
    u = "https://api.crossref.org/works?rows=3&select=title,author,issued,container-title,volume,page,DOI,type&query.bibliographic=" + urllib.parse.quote(q)
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "citation-check/1.0"}), timeout=30))["message"]["items"]
    except Exception as e:
        print("ERR", q, e); continue
    print("Q:", q)
    for it in d[:2]:
        a = ",".join(x.get("family", "?") for x in it.get("author", [])[:3])
        print("   ", (it.get("title") or ["?"])[0][:80], "|", a, "|", it.get("issued", {}).get("date-parts", [[None]])[0][0], "|", (it.get("container-title") or [""])[0][:40], it.get("volume"), it.get("page"), it.get("DOI"), it.get("type"))
    time.sleep(4)
