import json, urllib.request, urllib.parse, time
def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"citation-check/1.0"}),timeout=30))
for q in ["Challenging Common Assumptions Unsupervised Learning Disentangled Representations","beta-VAE Learning Basic Visual Concepts Constrained Variational Framework","On the Binding Problem in Artificial Neural Networks","Compositional Visual Generation with Composable Diffusion Models","Toy Models of Superposition"]:
    try:
        h=get("https://dblp.org/search/publ/api?format=json&h=4&q="+urllib.parse.quote(q))["result"]["hits"].get("hit",[])
        print("Q",q)
        for x in h[:3]:
            i=x["info"]; print("   ",i.get("title"),"|",i.get("venue"),"|",i.get("year"),"|",i.get("pages"),"|",i.get("volume"),"|",i.get("doi"))
    except Exception as e: print("ERR",q,e)
    time.sleep(3)
for doi in ["10.1207/s15516709cog1402_1","10.1111/j.2041-6962.1988.tb00470.x","10.1098/rstb.2019.0307"]:
    m=get("https://api.crossref.org/works/"+doi)["message"]
    print(doi,m["title"][0],[a.get("family") for a in m.get("author",[])],m["issued"]["date-parts"][0],m.get("container-title"),m.get("volume"),m.get("issue"),m.get("page"))
    time.sleep(3)
