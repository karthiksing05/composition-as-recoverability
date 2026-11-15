import urllib.request, os, re, sys, fitz
D="notes/audit/web_sources"; os.makedirs(D,exist_ok=True)
U={"smolensky1990":"https://nessie.ilab.sztaki.hu/~kornai/2021/ESSLLI/smolensky_1990.pdf",
"challenge2606":"https://arxiv.org/pdf/2606.14512","core72765645":"https://core.ac.uk/download/72765645.pdf",
"aydede":"https://philarchive.org/archive/AYDLOT","chalmers":"https://philarchive.org/archive/CHACAC-2v1"}
KW={"smolensky1990":r"saturat|linearly independent|graceful|Fodor|faithful",
"default":r"coffee|weak compositional|strong compositional|causally efficacious|graceful"}
for k,u in U.items():
    try:
        b=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0 citation-check"}),timeout=60).read()
        fn=f"{D}/{k}.pdf"; open(fn,"wb").write(b)
        d=fitz.open(fn); T=[p.get_text() for p in d]
        print("=====",k,len(d),"pages",re.sub(r"\s+"," ",T[0][:200]))
        n=0
        for i,t in enumerate(T):
            t=re.sub(r"\s+"," ",t)
            for m in re.finditer(KW.get(k,KW["default"]),t):
                print(f"[p{i+1}] ...{t[max(0,m.start()-200):m.end()+220]}...")
                n+=1
                if n>=9: break
            if n>=9: break
    except Exception as e:
        print("ERR",k,e)
