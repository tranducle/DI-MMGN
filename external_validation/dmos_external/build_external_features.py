import concurrent.futures, hashlib, json, math, pathlib, re, time
import numpy as np
import torch
from bs4 import BeautifulSoup, UnicodeDammit
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
SNAPS=ROOT/"external_snapshot_manifest.jsonl"
PAIRS=ROOT/"external_pairs.jsonl"
FEAT=ROOT/"features"
TEXT=FEAT/"text"; GRAPHS=FEAT/"graphs"; SHOTS=FEAT/"screenshots"; VIS=FEAT/"visuals"
for d in [TEXT,GRAPHS,SHOTS,VIS]: d.mkdir(parents=True,exist_ok=True)

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
MODEL_COMMIT="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CLIP_ID="openai/clip-vit-base-patch32"
CLIP_COMMIT="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MAX_LEN=512; CHAR_CAP=3072; MAX_NODES=2048
TAGS=["html","head","body","title","meta","link","style","script","noscript","div","span","p","a","img",
"ul","ol","li","nav","header","footer","main","section","article","aside","h1","h2","h3","h4","h5","h6",
"table","thead","tbody","tfoot","tr","th","td","form","input","button","select","option","textarea","label",
"iframe","video","audio","source","canvas","svg","path","figure","figcaption","br","hr","strong","em","b","i",
"small","pre","code","blockquote","__other__"]
TAG_INDEX={t:i for i,t in enumerate(TAGS)}

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def safe_key(r): return r["domain"].replace(":","_").replace("/","_"),r["snapshot_id"]

def visible_text(path):
    raw=pathlib.Path(path).read_bytes()
    try:s=BeautifulSoup(raw,"lxml")
    except Exception:s=BeautifulSoup(raw,"html.parser")
    for tag in list(s.find_all(["script","style","noscript","template"])): tag.decompose()
    for tag in list(s.find_all(True)):
        attrs=tag.attrs if isinstance(tag.attrs,dict) else {}
        style=str(attrs.get("style","")).replace(" ","").lower()
        if ("hidden" in attrs or str(attrs.get("aria-hidden","")).lower()=="true" or "display:none" in style
            or "visibility:hidden" in style or (tag.name=="input" and str(attrs.get("type","")).lower()=="hidden")):
            tag.decompose()
    return " ".join(s.get_text(separator=" ",strip=True).split())[:CHAR_CAP]

def dom_depth(tag):
    d=0;p=tag.parent
    while p is not None and getattr(p,"name",None) is not None and d<64:d+=1;p=p.parent
    return d

def build_graph(r):
    p=pathlib.Path(r["html_path"]); raw=p.read_bytes()
    try:s=BeautifulSoup(raw,"lxml")
    except Exception:s=BeautifulSoup(raw,"html.parser")
    tags=s.find_all(True)[:MAX_NODES]
    if not tags:
        s=BeautifulSoup("<html><body></body></html>","lxml"); tags=s.find_all(True)
    idx={id(t):i for i,t in enumerate(tags)}
    x=np.zeros((len(tags),70),dtype=np.float32); edges=[]
    for i,t in enumerate(tags):
        name=(t.name or "").lower(); x[i,TAG_INDEX.get(name,TAG_INDEX["__other__"])]=1.0
        child_tags=[c for c in t.children if getattr(c,"name",None)]
        txt=t.get_text(" ",strip=True)
        x[i,64]=min(dom_depth(t),32)/32.0
        x[i,65]=min(math.log1p(len(child_tags))/math.log(65),1.0)
        x[i,66]=min(math.log1p(len(txt))/math.log(10001),1.0)
        x[i,67]=min(len(t.attrs),10)/10.0
        x[i,68]=1.0 if "id" in t.attrs else 0.0
        x[i,69]=1.0 if "class" in t.attrs else 0.0
        par=t.parent
        if par is not None and id(par) in idx:
            a=idx[id(par)];edges.extend([(a,i),(i,a)])
    edge=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
    dom,sid=safe_key(r);out=GRAPHS/dom/f"{sid}.pt";out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"x":torch.from_numpy(x),"edge_index":edge,"num_nodes":len(tags),"source_html":r["html_path"]},out)
    return str(out)

CSP='<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\' data:; font-src data:; media-src data:; frame-src \'none\'; connect-src \'none\';">'
STYLE='<style>*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}html{scroll-behavior:auto!important}</style>'
def decode_html(raw):
    text=UnicodeDammit(raw,is_html=True).unicode_markup or raw.decode("utf-8",errors="replace")
    text=text.encode("utf-8",errors="replace").decode("utf-8")
    text="".join(ch for ch in text if ch in "\t\n\r" or ord(ch)>=32)
    m=re.search(r"<head\b[^>]*>",text,flags=re.I);inject=CSP+STYLE
    return text[:m.end()]+inject+text[m.end():] if m else inject+text

def main():
    snaps=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs=[json.loads(x) for x in PAIRS.read_text(encoding="utf-8").splitlines() if x.strip()]
    snaps.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started=time.time()

    # Text — exact v4 policy
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_COMMIT)
    model=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_COMMIT).eval().to(device)
    text_paths={}
    for start in range(0,len(snaps),32):
        batch=snaps[start:start+32]; texts=[visible_text(r["html_path"]) for r in batch]
        enc=tok(texts,padding="max_length",truncation=True,max_length=MAX_LEN,return_tensors="pt")
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.inference_mode(): h=model(**enc).last_hidden_state[:,0,:].detach().float().cpu().numpy()
        if h.shape!=(len(batch),384) or not np.isfinite(h).all(): raise RuntimeError("bad text output")
        for j,r in enumerate(batch):
            dom,sid=safe_key(r);out=TEXT/dom/f"{sid}.npy";out.parent.mkdir(parents=True,exist_ok=True);np.save(out,h[j].astype(np.float32))
            text_paths[r["domain"]+"|"+r["snapshot_id"]]=str(out)
        print(f"EXT_TEXT {min(start+32,len(snaps))}/{len(snaps)}",flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # DOM
    graph_paths={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(build_graph,r):r for r in snaps}
        for i,f in enumerate(concurrent.futures.as_completed(fut),1):
            r=fut[f];graph_paths[r["domain"]+"|"+r["snapshot_id"]]=f.result()
            if i%50==0 or i==len(snaps):print(f"EXT_DOM {i}/{len(snaps)}",flush=True)

    # Visual render with retries/fresh pages
    rendered={};render_errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={"width":1280,"height":720},device_scale_factor=1.0,java_script_enabled=False)
        for i,r in enumerate(snaps,1):
            dom,sid=safe_key(r);out=SHOTS/dom/f"{sid}.png";out.parent.mkdir(parents=True,exist_ok=True)
            raw=pathlib.Path(r["html_path"]).read_bytes();html=decode_html(raw);ok=False;last=None
            for timeout in [10000,30000,60000]:
                page=context.new_page();page.set_default_timeout(timeout)
                try:
                    page.set_content(html,wait_until="domcontentloaded",timeout=timeout)
                    png=page.screenshot(type="png",full_page=False,animations="disabled",timeout=timeout)
                    out.write_bytes(png);ok=True;break
                except Exception as e:last=repr(e)
                finally:
                    try:page.close()
                    except Exception:pass
            if ok:rendered[r["domain"]+"|"+r["snapshot_id"]]=str(out)
            else:render_errors.append({"domain":r["domain"],"snapshot_id":r["snapshot_id"],"error":last})
            if i%25==0 or i==len(snaps):print(f"EXT_RENDER {i}/{len(snaps)} errors={len(render_errors)}",flush=True)
        context.close();browser.close()

    complete_pairs=[]
    for q in pairs:
        k1=q["domain"]+"|"+q["t1"];k2=q["domain"]+"|"+q["t2"]
        if k1 in rendered and k2 in rendered:complete_pairs.append(q)
    pair_cov=len(complete_pairs)/max(1,len(pairs))
    # Re-check applicable predeclared threshold on visually complete pairs.
    pos=sum(q["label"]=="defaced" for q in complete_pairs);ben=sum(q["label"]=="legitimate" for q in complete_pairs)
    ph=len({q["host_group"] for q in complete_pairs if q["label"]=="defaced"})
    bh=len({q["host_group"] for q in complete_pairs if q["label"]=="legitimate"})
    mode=json.loads((ROOT/"G1C_DMoS_TEMPORAL_COHORT.json").read_text(encoding="utf-8"))["verdict"]
    threshold_ok=(pos>=100 and ph>=8 and ben>=100 and bh>=8) if mode=="PASS_FULL_COHORT" else (pos>=50 and ph>=5)
    use_visual=bool(pair_cov>=0.95 and threshold_ok)

    visual_paths={}
    if use_visual:
        proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_COMMIT,use_fast=False)
        clip=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_COMMIT).eval().to(device)
        render_snaps=[r for r in snaps if r["domain"]+"|"+r["snapshot_id"] in rendered]
        for start in range(0,len(render_snaps),64):
            batch=render_snaps[start:start+64];imgs=[]
            for r in batch:
                with Image.open(rendered[r["domain"]+"|"+r["snapshot_id"]]) as im:imgs.append(im.convert("RGB").copy())
            pix=proc(images=imgs,return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode():
                z=clip.get_image_features(pixel_values=pix);z=z/z.norm(dim=-1,keepdim=True).clamp(min=1e-12);arr=z.detach().float().cpu().numpy()
            for j,r in enumerate(batch):
                dom,sid=safe_key(r);out=VIS/dom/f"{sid}.npy";out.parent.mkdir(parents=True,exist_ok=True);np.save(out,arr[j].astype(np.float32))
                visual_paths[r["domain"]+"|"+r["snapshot_id"]]=str(out)
            print(f"EXT_CLIP {min(start+64,len(render_snaps))}/{len(render_snaps)}",flush=True)
        del clip
        if torch.cuda.is_available():torch.cuda.empty_cache()

    # Write feature manifest and the actually authorized evaluation pair set.
    eval_pairs=complete_pairs if use_visual else pairs
    fm=ROOT/"external_feature_manifest.jsonl"
    with fm.open("w",encoding="utf-8",newline="\n") as f:
        for r in snaps:
            k=r["domain"]+"|"+r["snapshot_id"]
            x={**r,"text_path":text_paths[k],"graph_path":graph_paths[k],
               "screenshot_path":rendered.get(k),"visual_path":visual_paths.get(k)}
            f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
    ep=ROOT/"external_eval_pairs.jsonl"
    with ep.open("w",encoding="utf-8",newline="\n") as f:
        for q in eval_pairs:f.write(json.dumps(q,sort_keys=True,separators=(",",":"))+"\n")
    report={
      "gate":"G3_EXTERNAL_FEATURES","verdict":"PASS",
      "snapshots":len(snaps),"pairs_input":len(pairs),"rendered_snapshots":len(rendered),
      "render_errors":render_errors[:100],"render_pair_coverage":pair_cov,
      "visual_mode":"text_dom_visual" if use_visual else "text_dom",
      "eval_pairs":len(eval_pairs),
      "eval_positive_pairs":sum(q["label"]=="defaced" for q in eval_pairs),
      "eval_benign_pairs":sum(q["label"]=="legitimate" for q in eval_pairs),
      "eval_host_groups":len({q["host_group"] for q in eval_pairs}),
      "feature_manifest_sha256":sha_file(fm),"eval_pairs_sha256":sha_file(ep),
      "elapsed_seconds":time.time()-started,
    }
    (ROOT/"G3_EXTERNAL_FEATURES.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
