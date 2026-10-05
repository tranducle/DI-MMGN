import argparse, hashlib, json, multiprocessing as mp, os, pathlib, sys, time
import numpy as np

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
sys.path.insert(0,str(REIMPL/"defacementfusion_2025"))
from static_dataset import WordTokenizer, load_manifest, visible_text_from_html
from dom import DOMWindowBuilder
from transformers import AutoTokenizer

G_WORD=None; G_BUILDER=None

def init_worker(vocab_path):
    global G_WORD,G_BUILDER
    G_WORD=WordTokenizer.load(vocab_path)
    bert=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
    G_BUILDER=DOMWindowBuilder(bert,max_length=512,overlap_tokens=256,max_windows=4)

def work(item):
    idx,row=item
    try:
        text=visible_text_from_html(row["html_path"])
        tids=np.asarray(G_WORD.encode(text,128),dtype=np.int32)
        ws=G_BUILDER.windows(row["html_path"])
        arrays={
            "input_ids":np.zeros((4,512),dtype=np.int32),
            "attention":np.zeros((4,512),dtype=np.uint8),
            "tag_ids":np.zeros((4,512),dtype=np.uint16),
            "depth_ids":np.zeros((4,512),dtype=np.uint8),
            "node_ids":np.zeros((4,512),dtype=np.uint16),
        }
        for wi,w in enumerate(ws[:4]):
            L=min(512,len(w["input_ids"]))
            arrays["input_ids"][wi,:L]=np.asarray(w["input_ids"][:L],dtype=np.int32)
            arrays["attention"][wi,:L]=np.asarray(w["attention_mask"][:L],dtype=np.uint8)
            arrays["tag_ids"][wi,:L]=np.asarray(w["tag_ids"][:L],dtype=np.uint16)
            arrays["depth_ids"][wi,:L]=np.asarray(w["depth_ids"][:L],dtype=np.uint8)
            arrays["node_ids"][wi,:L]=np.asarray(w["node_ids"][:L],dtype=np.uint16)
        return idx,tids,arrays,len(ws),None
    except Exception as e:
        return idx,None,None,0,repr(e)

def open_mm(path,dtype,shape,create):
    if create:
        return np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=shape)
    return np.lib.format.open_memmap(path,mode="r+")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl")
    ap.add_argument("--vocab",default=str(REIMPL/"bilstm_efficientnet_2021"/"vocab.json"))
    ap.add_argument("--out",default=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\cache_v1")
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(args.manifest,None); N=len(rows)
    if N!=4678: raise RuntimeError(f"expected 4678 rows, got {N}")
    manifest_sha=hashlib.sha256(pathlib.Path(args.manifest).read_bytes()).hexdigest()
    vocab_sha=hashlib.sha256(pathlib.Path(args.vocab).read_bytes()).hexdigest()
    pair_ids=[r["pair_id"] for r in rows]
    (out/"pair_ids.json").write_text(json.dumps(pair_ids)+"\n",encoding="utf-8")

    specs={
        "text_ids":(np.int32,(N,128)),
        "dom_input_ids":(np.int32,(N,4,512)),
        "dom_attention":(np.uint8,(N,4,512)),
        "dom_tag_ids":(np.uint16,(N,4,512)),
        "dom_depth_ids":(np.uint8,(N,4,512)),
        "dom_node_ids":(np.uint16,(N,4,512)),
        "dom_window_count":(np.uint8,(N,)),
        "done":(np.uint8,(N,))
    }
    create=not (out/"done.npy").exists()
    mm={k:open_mm(out/f"{k}.npy",dt,shape,create) for k,(dt,shape) in specs.items()}
    todo=[(i,r) for i,r in enumerate(rows) if int(mm["done"][i])==0]
    print(json.dumps({"stage":"cache_start","rows":N,"todo":len(todo),"workers":args.workers,"out":str(out)}),flush=True)
    errors=[]; completed=N-len(todo); t0=time.time()
    ctx=mp.get_context("spawn")
    with ctx.Pool(processes=args.workers,initializer=init_worker,initargs=(args.vocab,)) as pool:
        for idx,tids,arrs,nw,err in pool.imap_unordered(work,todo,chunksize=2):
            if err is not None:
                errors.append({"index":idx,"pair_id":rows[idx]["pair_id"],"error":err})
                print(json.dumps({"cache_error":errors[-1]}),flush=True)
                continue
            mm["text_ids"][idx]=tids
            mm["dom_input_ids"][idx]=arrs["input_ids"]
            mm["dom_attention"][idx]=arrs["attention"]
            mm["dom_tag_ids"][idx]=arrs["tag_ids"]
            mm["dom_depth_ids"][idx]=arrs["depth_ids"]
            mm["dom_node_ids"][idx]=arrs["node_ids"]
            mm["dom_window_count"][idx]=nw
            mm["done"][idx]=1
            completed+=1
            if completed%100==0 or completed==N:
                for x in mm.values(): x.flush()
                print(f"CACHE_PROGRESS {completed}/{N} errors={len(errors)} elapsed={time.time()-t0:.1f}s",flush=True)
    for x in mm.values(): x.flush()
    done_count=int(np.asarray(mm["done"]).sum())
    counts=np.bincount(np.asarray(mm["dom_window_count"],dtype=np.int64),minlength=5).tolist()
    summary={
        "status":"PASS" if done_count==N and not errors else "FAIL",
        "rows":N,"done_count":done_count,"errors":errors[:50],"error_count":len(errors),
        "manifest_sha256":manifest_sha,"vocab_sha256":vocab_sha,
        "bert_model":"bert-base-uncased","text_max_words":128,
        "dom":{"max_length":512,"overlap_tokens":256,"max_windows":4,"window_count_histogram_0_to_4":counts[:5]},
        "workers":args.workers,"elapsed_seconds":time.time()-t0,"out":str(out)
    }
    (out/"cache_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    raise SystemExit(0 if summary["status"]=="PASS" else 3)

if __name__=="__main__":
    mp.freeze_support()
    main()
