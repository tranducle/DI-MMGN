import argparse, json, pathlib, sys, torch
from torch.utils.data import DataLoader, Subset

HERE=pathlib.Path(__file__).resolve(); REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import WordTokenizer, TextIdCache, DOMWindowCache, StaticPageDataset
from model import DefacementFusion2025

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
CACHE_DIR=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\cache_v1"

def move_dom(dom,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in dom.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--no-amp",action="store_true"); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=(device.type=="cuda" and not args.no_amp)
    tok=WordTokenizer.load(REIMPL/"bilstm_efficientnet_2021"/"vocab.json")
    tc=TextIdCache(CACHE_DIR); dc=DOMWindowCache(CACHE_DIR)
    ds=StaticPageDataset(MANIFEST,"pretrain",tok,128,224,text_id_cache=tc)
    # Choose worst-case cached pages with four DOM windows.
    idx=[]
    for i,r in enumerate(ds.rows):
        ci=dc.index[r["pair_id"]]
        if int(dc.window_count[ci])==4: idx.append(i)
        if len(idx)>=args.batch_size: break
    if len(idx)<args.batch_size: raise RuntimeError("not enough 4-window pages for smoke")
    items=[ds[i] for i in idx]
    pair_ids=[x["pair_id"] for x in items]
    dom=dc.batch(pair_ids)
    text_ids=torch.stack([x["text_ids"] for x in items]).to(device)
    images=torch.stack([x["image"] for x in items]).to(device)
    labels=torch.stack([x["label"] for x in items]).to(device)
    model=DefacementFusion2025(vocab_size=len(tok.word_index)+1,d_model=128,transformer_layers=1,nhead=4,
                              ffn_dim=256,transformer_dropout=0.1,bert_name="bert-base-uncased",
                              pretrained_effnet=True,local_files_only=True).to(device)
    model.train(); target=torch.nn.functional.one_hot(labels,2).float()
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    # Optimizer is created only so GradScaler can unscale gradients before the
    # scientific finiteness check, matching the semantics of official training.
    opt=torch.optim.Adam(model.parameters(),lr=2e-5)
    with torch.amp.autocast("cuda",enabled=amp):
        logits,feats=model(text_ids,images,move_dom(dom,device))
        loss=torch.nn.BCEWithLogitsLoss()(logits,target)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    finite=True; grad_counts={}
    for name,module in [("html",model.html),("text",model.text),("image",model.image),("fusion",model.fusion),("head",model.head)]:
        n=0; ok=0
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                n+=1; ok+=int(torch.isfinite(p.grad).all())
        grad_counts[name]=[ok,n]; finite=finite and n>0 and ok==n
    if device.type=="cuda":
        peak=torch.cuda.max_memory_allocated()/1024**2
        reserved=torch.cuda.max_memory_reserved()/1024**2
    else: peak=reserved=0.0
    result={"gate":"S2B_cached_fullwindow_smoke","status":"PASS" if torch.isfinite(loss).item() and finite else "FAIL",
            "batch_size":args.batch_size,"pair_ids":pair_ids,"dom_windows":int(dom["input_ids"].shape[0]),
            "seq_len":int(dom["input_ids"].shape[1]),"loss":float(loss.detach().cpu()),
            "gradient_tensors":grad_counts,"peak_allocated_mib":peak,"peak_reserved_mib":reserved,
            "device":str(device),"amp":amp}
    out=HERE.parent/f"S2B_CACHED_SMOKE_B{args.batch_size}.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["status"]=="PASS" else 3)

if __name__=="__main__": main()
