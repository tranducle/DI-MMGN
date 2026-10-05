import hashlib, json, os, platform, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import transformers

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
SNAPS=V4/"snapshot_manifest.jsonl"
G1=ROOT/"gates"/"G1_preprocessing_integrity.json"
TEXT=V4/"text"
SEQ=V4/"text_sequence_cache.npy"
MASK=V4/"text_sequence_mask.npy"
INDEX=V4/"text_sequence_index.json"
TMAN=V4/"text_manifest.jsonl"
PROV=V4/"text_provenance.json"

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
COMMIT="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LEN=512
CHAR_CAP=3072
BATCH=32
WORKERS=12

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def visible_text(path):
    raw=Path(path).read_bytes()
    try: s=BeautifulSoup(raw,"lxml")
    except Exception: s=BeautifulSoup(raw,"html.parser")
    for tag in list(s.find_all(["script","style","noscript","template"])):
        tag.decompose()
    for tag in list(s.find_all(True)):
        attrs=tag.attrs if isinstance(tag.attrs,dict) else {}
        style=str(attrs.get("style","")).replace(" ","").lower()
        if ("hidden" in attrs or str(attrs.get("aria-hidden","")).lower()=="true"
            or "display:none" in style or "visibility:hidden" in style
            or (tag.name=="input" and str(attrs.get("type","")).lower()=="hidden")):
            tag.decompose()
    return " ".join(s.get_text(separator=" ",strip=True).split())[:CHAR_CAP]

def main():
    g1=json.loads(G1.read_text(encoding="utf-8"))
    if g1.get("verdict")!="PASS": raise SystemExit("G1 is not PASS")
    rows=[json.loads(x) for x in SNAPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
    n=len(rows)
    TEXT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=COMMIT)
    model=AutoModel.from_pretrained(MODEL_ID,revision=COMMIT)
    model.eval().to(device)
    actual=getattr(model.config,"_commit_hash",None)
    if actual and actual!=COMMIT: raise RuntimeError(f"commit mismatch {actual} != {COMMIT}")

    seq=np.lib.format.open_memmap(SEQ,mode="w+",dtype=np.float16,shape=(n,MAX_LEN,384))
    mask=np.lib.format.open_memmap(MASK,mode="w+",dtype=np.uint8,shape=(n,MAX_LEN))
    idx={}
    trows=[]
    started=time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0,n,BATCH):
            batch=rows[start:start+BATCH]
            paths=[ROOT/r["html_path"] for r in batch]
            for p in paths:
                if not p.exists(): raise FileNotFoundError(p)
            texts=list(pool.map(visible_text,paths))
            enc=tok(texts,padding="max_length",truncation=True,max_length=MAX_LEN,return_tensors="pt")
            attn=enc["attention_mask"].cpu().numpy().astype(np.uint8)
            enc={k:v.to(device,non_blocking=True) for k,v in enc.items()}
            with torch.inference_mode():
                h=model(**enc).last_hidden_state.detach().float().cpu().numpy()
            if h.shape!=(len(batch),MAX_LEN,384): raise RuntimeError(h.shape)
            if not np.isfinite(h).all(): raise RuntimeError("non-finite text sequence")
            pooled=h[:,0,:]
            seq[start:start+len(batch)]=h.astype(np.float16)
            mask[start:start+len(batch)]=attn
            for j,r in enumerate(batch):
                out=TEXT/r["domain"]/f"{r['snapshot_id']}.npy"
                out.parent.mkdir(parents=True,exist_ok=True)
                np.save(out,pooled[j].astype(np.float32))
                key=r["domain"]+"|"+r["snapshot_id"]
                if key in idx: raise RuntimeError("duplicate text key "+key)
                idx[key]=start+j
                trows.append({
                    "domain":r["domain"],"snapshot_id":r["snapshot_id"],
                    "html_path":r["html_path"],"html_sha256":r["html_sha256"],
                    "text_path":str(out.relative_to(ROOT)).replace("\\","/"),
                    "text_sha256":sha_file(out),"dim":384,
                    "nonpadding_tokens":int(attn[j].sum()),
                })
            seq.flush(); mask.flush()
            print("text",min(start+BATCH,n),"/",n,flush=True)

    INDEX.write_text(json.dumps(idx,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    trows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
    with TMAN.open("w",encoding="utf-8",newline="\n") as f:
        for r in trows: f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    prov={
        "schema_version":4,"generated_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "model_id":MODEL_ID,"model_commit_hash":actual,"expected_commit_hash":COMMIT,
        "model_mode":"eval","gradient_tracking":False,"max_length":MAX_LEN,"char_cap":CHAR_CAP,
        "pre_html_rule":"BeautifulSoup parse; remove script/style/noscript/template and hidden/aria-hidden/display:none/visibility:hidden/input[type=hidden]; collapse whitespace; truncate to 3072 characters",
        "pooled_vector":"last_hidden_state[:,0] float32",
        "sequence_tensor":"last_hidden_state float16 storage; float32 training cast",
        "snapshot_count":n,"sequence_shape":[n,MAX_LEN,384],"mask_shape":[n,MAX_LEN],
        "text_manifest_sha256":sha_file(TMAN),"sequence_sha256":sha_file(SEQ),"mask_sha256":sha_file(MASK),"index_sha256":sha_file(INDEX),
        "torch_version":torch.__version__,"transformers_version":transformers.__version__,"python_version":platform.python_version(),
        "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds":time.time()-started,
    }
    PROV.write_text(json.dumps(prov,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(prov,indent=2),flush=True)

if __name__=="__main__":
    main()
