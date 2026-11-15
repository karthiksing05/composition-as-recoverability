import fitz, re, unicodedata
D="notes/audit/web_sources/"
for k,kw,maxn in (("aydede",r"coffee|weak compos|strong compos|causally eff|extended sense",7),("chalmers",r"coffee|weak compos|strong compos|functional compos",4),("challenge2606",r"Smolensky|coffee",6)):
    d=fitz.open(D+k+".pdf"); n=0
    print("=====",k, sum(len(p.get_text()) for p in d))
    for i,p in enumerate(d):
        t=unicodedata.normalize("NFKC",p.get_text()); t=re.sub(r"-\s*\n","",t); t=re.sub(r"\s+"," ",t)
        last=-1000
        for m in re.finditer(kw,t,re.I):
            if m.start()-last<500: continue
            last=m.start(); n+=1
            if n<=maxn: print(f"[p{i+1}] {t[max(0,m.start()-260):m.end()+300]}\n")
