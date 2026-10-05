import argparse, json, pathlib, random, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import load_manifest
from dom import DOMWindowBuilder, node_aware_mlm_mask
from model import HTMLDefaceMLM

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class WindowDataset(Dataset):
    def __init__(self,windows): self.windows=windows
    def __len__(self): return len(self.windows)
    def __getitem__(self,i): return self.windows[i]

def collate_windows(items,pad_token_id):
    L=max(len(x["input_ids"]) for x in items)
    def pad(name,padval=0):
        return torch.tensor([x[name]+[padval]*(L-len(x[name])) for x in items],dtype=torch.long)
    return {"input_ids":pad("input_ids",pad_token_id),"attention_mask":pad("attention_mask",0),
            "tag_ids":pad("tag_ids",0),"depth_ids":pad("depth_ids",0),"node_ids":pad("node_ids",0)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--physical-batch-size",type=int,default=4)
    ap.add_argument("--accum-steps",type=int,default=4)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--max-windows",type=int,default=4)
    ap.add_argument("--mini-pages",type=int,default=None)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--official",action="store_true")
    ap.add_argument("--out",default=None)
    args=ap.parse_args()
    if args.official:
        if args.epochs!=5 or args.lr!=2e-5 or args.physical_batch_size*args.accum_steps!=16:
            raise RuntimeError("official HTML pretraining must preserve paper-reported epochs=5, lr=2e-5, effective batch=16")
        if args.max_windows!=4:
            raise RuntimeError("official local protocol freezes max_windows=4 per SPEC.md")
    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=pathlib.Path(args.out or (HERE.parent/"html_pretrain"/("official" if args.official else "sanity")))
    out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(MANIFEST,"pretrain")
    if args.mini_pages is not None: rows=rows[:args.mini_pages]
    tok=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
    builder=DOMWindowBuilder(tok,max_length=512,overlap_tokens=256,max_windows=args.max_windows)

    print(json.dumps({"stage":"cache_windows","pages":len(rows),"max_windows":args.max_windows}),flush=True)
    windows=[]; tcache=time.time()
    for i,r in enumerate(rows,1):
        windows.extend(builder.windows(r["html_path"]))
        if i%100==0 or i==len(rows):
            print(f"HTML_WINDOW_CACHE {i}/{len(rows)} windows={len(windows)} elapsed={time.time()-tcache:.1f}s",flush=True)

    ds=WindowDataset(windows)
    gen=torch.Generator(); gen.manual_seed(args.seed)
    dl=DataLoader(ds,batch_size=args.physical_batch_size,shuffle=True,num_workers=args.workers,
                  collate_fn=lambda xs:collate_windows(xs,tok.pad_token_id or 0),generator=gen,pin_memory=True)
    model=HTMLDefaceMLM(local_files_only=True).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=random.Random(args.seed)
    hist=[]; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); total=0.; ntok=0; nb=0; opt_steps=0
        opt.zero_grad(set_to_none=True)
        for step,b in enumerate(dl,1):
            masked,labels=node_aware_mlm_mask(b,tok,0.15,rng)
            for k in ["input_ids","attention_mask","tag_ids","depth_ids","node_ids"]:
                masked[k]=masked[k].to(device)
            labels=labels.to(device)
            outp=model(masked,labels)
            loss=outp.loss/args.accum_steps
            if not torch.isfinite(loss): raise RuntimeError("nonfinite MLM loss")
            loss.backward()
            if step%args.accum_steps==0 or step==len(dl):
                opt.step(); opt.zero_grad(set_to_none=True); opt_steps+=1
            nmask=int((labels!=-100).sum().item())
            total+=float(loss.detach().cpu())*args.accum_steps*nmask; ntok+=nmask; nb+=1
            if step%100==0:
                print(json.dumps({"epoch":ep,"step":step,"batches":len(dl),"mean_masked_token_loss":total/max(1,ntok),"optimizer_steps":opt_steps}),flush=True)
        row={"epoch":ep,"mean_masked_token_loss":total/max(1,ntok),"masked_tokens":ntok,"optimizer_steps":opt_steps}
        hist.append(row); print(json.dumps(row),flush=True)
        torch.save({"html_encoder":model.export_html_encoder_state(),"epoch":ep,"history":hist,
                    "config":vars(args),"pages":len(rows),"windows":len(windows)},out/f"epoch_{ep}.pt")
    final=out/"html_encoder_final.pt"
    torch.save({"html_encoder":model.export_html_encoder_state(),"epoch":args.epochs,"history":hist,
                "config":vars(args),"pages":len(rows),"windows":len(windows)},final)
    summary={"status":"PASS","official":args.official,"pages":len(rows),"windows":len(windows),
             "epochs":args.epochs,"effective_batch_size":args.physical_batch_size*args.accum_steps,
             "history":hist,"checkpoint":str(final),"elapsed_seconds":time.time()-t0,"cache_seconds":time.time()-tcache-(time.time()-t0),
             "device":str(device)}
    (out/"result.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
