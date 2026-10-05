import argparse, hashlib, json, pathlib, sys, time
import torch
from transformers import AutoTokenizer

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import load_manifest
from dom import DOMWindowBuilder

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
DEFAULT_OUT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\defacementfusion_mlm_cache.pt")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default=MANIFEST)
    ap.add_argument("--out",default=str(DEFAULT_OUT))
    ap.add_argument("--max-length",type=int,default=512)
    ap.add_argument("--overlap-tokens",type=int,default=256)
    ap.add_argument("--max-windows",type=int,default=4)
    ap.add_argument("--limit",type=int,default=None)
    args=ap.parse_args()
    rows=load_manifest(args.manifest,"pretrain")
    # Pretraining corpus is current S_t HTML from pretrain families only.
    seen=set(); uniq=[]
    for r in rows:
        key=r["html_path"]
        if key not in seen:
            seen.add(key); uniq.append(r)
    if args.limit: uniq=uniq[:args.limit]
    tok=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
    builder=DOMWindowBuilder(tok,args.max_length,args.overlap_tokens,args.max_windows)
    ids=[]; masks=[]; tags=[]; depths=[]; nodes=[]; page_idx=[]
    pages_meta=[]; t0=time.time()
    L=args.max_length
    for i,r in enumerate(uniq):
        ws=builder.windows(r["html_path"])
        pages_meta.append({"page_index":i,"pair_id":r["pair_id"],"domain":r["domain"],"family_id":r["family_id"],
                           "html_path":r["html_path"],"window_count":len(ws)})
        for w in ws:
            n=len(w["input_ids"])
            if n>L: raise RuntimeError("window length exceeds max_length")
            pad=L-n
            ids.append(w["input_ids"]+[tok.pad_token_id or 0]*pad)
            masks.append(w["attention_mask"]+[0]*pad)
            tags.append(w["tag_ids"]+[0]*pad)
            depths.append(w["depth_ids"]+[0]*pad)
            nodes.append(w["node_ids"]+[0]*pad)
            page_idx.append(i)
        if (i+1)%50==0 or i+1==len(uniq):
            print(f"CACHE_PROGRESS {i+1}/{len(uniq)} windows={len(ids)} elapsed={time.time()-t0:.1f}s",flush=True)
    data={
        "input_ids":torch.tensor(ids,dtype=torch.int32),
        "attention_mask":torch.tensor(masks,dtype=torch.uint8),
        "tag_ids":torch.tensor(tags,dtype=torch.int16),
        "depth_ids":torch.tensor(depths,dtype=torch.int16),
        "node_ids":torch.tensor(nodes,dtype=torch.int16),
        "page_index":torch.tensor(page_idx,dtype=torch.int32),
        "metadata":{
            "source_manifest":args.manifest,
            "source_split":"pretrain",
            "unique_current_pages":len(uniq),
            "window_count":len(ids),
            "max_length":L,"overlap_tokens":args.overlap_tokens,"max_windows":args.max_windows,
            "tokenizer":"bert-base-uncased",
            "family_count":len({r["family_id"] for r in uniq}),
            "domain_count":len({r["domain"] for r in uniq}),
            "label_counts":{str(y):sum(int(r["label"])==y for r in uniq) for y in [0,1]},
            "elapsed_seconds":time.time()-t0,
            "pages":pages_meta
        }
    }
    out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,out)
    summary={k:v for k,v in data["metadata"].items() if k!="pages"}
    summary.update({"cache_path":str(out),"cache_bytes":out.stat().st_size,"cache_sha256":sha256(out)})
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
