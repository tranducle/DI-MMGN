import argparse, json, math, pathlib, random, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import WordTokenizer, TextIdCache, DOMWindowCache, StaticPageDataset
from dom import DOMWindowBuilder
from model import DefacementFusion2025

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
CACHE_DIR=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\cache_v1"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def balanced_indices(ds,n,seed):
    if n is None or n>=len(ds): return list(range(len(ds)))
    rng=random.Random(seed); by={0:[],1:[]}
    for i,r in enumerate(ds.rows): by[int(r["label"])].append(i)
    for v in by.values(): rng.shuffle(v)
    n0=n//2; idx=by[0][:n0]+by[1][:n-n0]; rng.shuffle(idx); return idx

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "defaced_f1":float(f1_score(y,p,pos_label=1,average="binary",zero_division=0)),
        "accuracy":float(accuracy_score(y,p)),
        "precision":float(precision_score(y,p,zero_division=0)),
        "attack_recall":float(recall_score(y,p,zero_division=0)),
        "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def make_collate(builder):
    def collate(items):
        dom=builder.batch([x["pair_id"] for x in items])
        return {
            "text_ids":torch.stack([x["text_ids"] for x in items]),
            "image":torch.stack([x["image"] for x in items]),
            "label":torch.stack([x["label"] for x in items]),
            "pair_id":[x["pair_id"] for x in items],
            "domain":[x["domain"] for x in items],
            "family_id":[x["family_id"] for x in items],
            "attack_type":[x["attack_type"] for x in items],
            "source_type":[x["source_type"] for x in items],
            "dom":dom,
        }
    return collate

def move_dom(dom,device):
    out={}
    for k,v in dom.items(): out[k]=v.to(device) if torch.is_tensor(v) else v
    return out

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); rows=[]; ys=[]; ps=[]
    for b in loader:
        y=b["label"].to(device)
        with torch.amp.autocast("cuda",enabled=(device.type=="cuda")):
            lg,_=model(b["text_ids"].to(device),b["image"].to(device),move_dom(b["dom"],device))
        pr=torch.softmax(lg,dim=-1); pred=pr.argmax(-1)
        for i,pid in enumerate(b["pair_id"]):
            rows.append({"pair_id":pid,"label":int(y[i].cpu()),"pred":int(pred[i].cpu()),
                         "prob_benign":float(pr[i,0].cpu()),"prob_defaced":float(pr[i,1].cpu()),
                         "domain":b["domain"][i],"family_id":b["family_id"][i],
                         "attack_type":b["attack_type"][i] or None,"source_type":b["source_type"][i] or None})
        ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
    return metrics(ys,ps),rows

def load_html_pretrain(model,path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    sd=ck["html_encoder"] if "html_encoder" in ck else ck
    missing,unexpected=model.html.load_state_dict(sd,strict=False)
    allowed_missing={"proj.weight","proj.bias","bert.pooler.dense.weight","bert.pooler.dense.bias"}
    real_missing=[x for x in missing if x not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(f"HTML pretrain checkpoint mismatch missing={real_missing} unexpected={unexpected}")
    return {"path":str(path),"missing_allowed":list(missing)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--physical-batch-size",type=int,default=2)
    ap.add_argument("--accum-steps",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--mini-train",type=int,default=None)
    ap.add_argument("--mini-val",type=int,default=None)
    ap.add_argument("--max-windows",type=int,default=4)
    ap.add_argument("--html-pretrain",default=None)
    ap.add_argument("--allow-unpretrained-html",action="store_true")
    ap.add_argument("--no-amp",action="store_true",help="Disable CUDA autocast/GradScaler; used for official DefacementFusion after S2B AMP instability gate.")
    ap.add_argument("--official",action="store_true")
    ap.add_argument("--out",default=None)
    args=ap.parse_args()
    if args.official and (not args.html_pretrain or args.allow_unpretrained_html):
        raise RuntimeError("official run requires an HTML-pretraining checkpoint and forbids --allow-unpretrained-html")
    seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=pathlib.Path(args.out or (HERE.parent/"runs"/("official" if args.official else "sanity")/f"seed_{args.seed}"))
    out.mkdir(parents=True,exist_ok=True)

    word_tok=WordTokenizer.load(REIMPL/"bilstm_efficientnet_2021"/"vocab.json")
    text_cache=TextIdCache(CACHE_DIR)
    builder=DOMWindowCache(CACHE_DIR)
    ds_tr=StaticPageDataset(MANIFEST,"pretrain",word_tok,128,224,text_id_cache=text_cache)
    ds_va=StaticPageDataset(MANIFEST,"val",word_tok,128,224,text_id_cache=text_cache)
    ds_te=StaticPageDataset(MANIFEST,"test",word_tok,128,224,text_id_cache=text_cache)
    tri=balanced_indices(ds_tr,args.mini_train,args.seed); vai=balanced_indices(ds_va,args.mini_val,args.seed+1)
    collate=make_collate(builder)
    tr=DataLoader(Subset(ds_tr,tri),batch_size=args.physical_batch_size,shuffle=True,num_workers=args.workers,
                  pin_memory=True,collate_fn=collate)
    va=DataLoader(Subset(ds_va,vai),batch_size=args.physical_batch_size,shuffle=False,num_workers=args.workers,
                  pin_memory=True,collate_fn=collate)
    te=DataLoader(ds_te,batch_size=args.physical_batch_size,shuffle=False,num_workers=args.workers,
                  pin_memory=True,collate_fn=collate) if args.official else None

    model=DefacementFusion2025(vocab_size=len(word_tok.word_index)+1,d_model=128,transformer_layers=1,nhead=4,
                              ffn_dim=256,transformer_dropout=0.1,bert_name="bert-base-uncased",
                              pretrained_effnet=True,local_files_only=True).to(device)
    pretrain_info=None
    if args.html_pretrain: pretrain_info=load_html_pretrain(model,args.html_pretrain)
    elif not args.allow_unpretrained_html: raise RuntimeError("HTML-pretraining checkpoint required unless sanity run explicitly uses --allow-unpretrained-html")

    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    amp=(device.type=="cuda" and not args.no_amp); scaler=torch.amp.GradScaler("cuda",enabled=amp)
    best=-1.0; best_epoch=0; hist=[]; t0=time.time()
    opt.zero_grad(set_to_none=True)
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.; n=0; steps=0
        for step,b in enumerate(tr,1):
            y=b["label"].to(device); target=torch.nn.functional.one_hot(y,2).float()
            with torch.amp.autocast("cuda",enabled=amp):
                lg,_=model(b["text_ids"].to(device),b["image"].to(device),move_dom(b["dom"],device))
                loss=loss_fn(lg,target)/args.accum_steps
            if not torch.isfinite(loss): raise RuntimeError("nonfinite classification loss")
            scaler.scale(loss).backward()
            if step%args.accum_steps==0 or step==len(tr):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); steps+=1
            tot+=float(loss.detach().cpu())*args.accum_steps*len(y); n+=len(y)
        vm,vrows=evaluate(model,va,device)
        row={"epoch":ep,"train_loss":tot/max(1,n),"optimizer_steps":steps,
             **{f"val_{k}":v for k,v in vm.items() if isinstance(v,(int,float))}}
        hist.append(row); print(json.dumps(row),flush=True)
        if vm["macro_f1"]>best:
            best=vm["macro_f1"]; best_epoch=ep
            torch.save({"state_dict":model.state_dict(),"epoch":ep,"val_macro_f1":best},out/"best.pt")

    ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"])
    vm,vrows=evaluate(model,va,device)
    with (out/"val_predictions.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for r in vrows:f.write(json.dumps(r,sort_keys=True)+"\n")
    result={"model":"defacementfusion_2025","fidelity":"paper-faithful reimplementation; not official/exact source",
            "seed":args.seed,"official":args.official,"train_rows":len(tri),"val_rows":len(vai),
            "test_rows":len(ds_te) if args.official else 0,"config":vars(args),"effective_batch_size":args.physical_batch_size*args.accum_steps,
            "html_pretrain":pretrain_info,"best_epoch":best_epoch,"best_val_macro_f1":best,"history":hist,
            "val":vm,"elapsed_seconds":time.time()-t0,"device":str(device),"mixed_precision":amp}
    if args.official:
        tm,trows=evaluate(model,te,device)
        with (out/"test_predictions.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for r in trows:f.write(json.dumps(r,sort_keys=True)+"\n")
        result["test"]=tm
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","out":str(out),"val":vm,"test":result.get("test")},indent=2),flush=True)

if __name__=="__main__": main()
