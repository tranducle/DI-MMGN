import concurrent.futures, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from bs4 import BeautifulSoup

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
OUT=V4/"graphs"
REPORT=ROOT/"gates"/"G1c_dom_graph_generation.json"
MAX_NODES=2048
WORKERS=8
TAGS=[
"html","head","body","title","meta","link","style","script","noscript","div","span","p","a","img",
"ul","ol","li","nav","header","footer","main","section","article","aside","h1","h2","h3","h4","h5","h6",
"table","thead","tbody","tfoot","tr","th","td","form","input","button","select","option","textarea","label",
"iframe","video","audio","source","canvas","svg","path","figure","figcaption","br","hr","strong","em","b","i",
"small","pre","code","blockquote","__other__"
]
assert len(TAGS)==64
TAG_INDEX={t:i for i,t in enumerate(TAGS)}
SCHEMA={
 "tags":TAGS,
 "numeric":["depth_min32_div32","log1p_child_count_div_log65","log1p_text_len_div_log10001","attr_count_min10_div10","has_id","has_class"],
 "edge_policy":"bidirectional parent-child edges among retained nodes",
 "max_nodes":MAX_NODES,
}
SCHEMA_SHA=hashlib.sha256(json.dumps(SCHEMA,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def depth(tag):
    d=0; p=tag.parent
    while p is not None and getattr(p,"name",None) is not None and d<64:
        d+=1; p=p.parent
    return d

def build_one(r):
    src=ROOT/r["html_path"]
    raw=src.read_bytes()
    try:s=BeautifulSoup(raw,"lxml")
    except Exception:s=BeautifulSoup(raw,"html.parser")
    tags=s.find_all(True)
    truncated=len(tags)>MAX_NODES
    tags=tags[:MAX_NODES]
    if not tags:
        s=BeautifulSoup("<html><body></body></html>","lxml"); tags=s.find_all(True)[:MAX_NODES]
    idx={id(t):i for i,t in enumerate(tags)}
    x=np.zeros((len(tags),70),dtype=np.float32)
    edges=[]
    for i,t in enumerate(tags):
        name=(t.name or "").lower()
        j=TAG_INDEX.get(name,TAG_INDEX["__other__"])
        x[i,j]=1.0
        child_tags=[c for c in t.children if getattr(c,"name",None)]
        txt=t.get_text(" ",strip=True)
        x[i,64]=min(depth(t),32)/32.0
        x[i,65]=min(math.log1p(len(child_tags))/math.log(65),1.0)
        x[i,66]=min(math.log1p(len(txt))/math.log(10001),1.0)
        x[i,67]=min(len(t.attrs),10)/10.0
        x[i,68]=1.0 if "id" in t.attrs else 0.0
        x[i,69]=1.0 if "class" in t.attrs else 0.0
        p=t.parent
        if p is not None and id(p) in idx:
            a=idx[id(p)]; b=i
            edges.append((a,b)); edges.append((b,a))
    edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
    out=OUT/r["domain"]/f"{r['snapshot_id']}.pt"
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "x":torch.from_numpy(x),"edge_index":edge_index,"num_nodes":len(tags),
        "domain":r["domain"],"snapshot_id":r["snapshot_id"],"source_html":r["html_path"],
        "feature_schema_sha256":SCHEMA_SHA,"truncated":truncated,
    },out)
    return {"domain":r["domain"],"snapshot_id":r["snapshot_id"],"nodes":len(tags),"edges":edge_index.shape[1],"truncated":truncated,"path":str(out.relative_to(ROOT)).replace("\\","/")}

def main():
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]; errors=[]
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(build_one,r):r for r in rows}
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            r=futs[f]
            try:results.append(f.result())
            except Exception as e:errors.append({"domain":r["domain"],"snapshot_id":r["snapshot_id"],"error":type(e).__name__+": "+str(e)})
            if i%250==0 or i==len(rows): print("graphs",i,"/",len(rows),"errors",len(errors),flush=True)
    criteria={"all_graphs_generated":len(results)==len(rows),"no_errors":not errors,"all_shapes_valid":all(x["nodes"]>=1 for x in results)}
    verdict="PASS" if all(criteria.values()) else "FAIL_REPAIR"
    rep={"gate_id":"G1c_dom_graph_generation","verdict":verdict,"snapshots":len(rows),"generated":len(results),"errors":errors[:50],"truncated":sum(x["truncated"] for x in results),"feature_schema":SCHEMA,"feature_schema_sha256":SCHEMA_SHA,"criteria":criteria,"elapsed_seconds":time.time()-started}
    REPORT.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rep,indent=2))
    return 0 if verdict=="PASS" else 3
if __name__=="__main__": raise SystemExit(main())
