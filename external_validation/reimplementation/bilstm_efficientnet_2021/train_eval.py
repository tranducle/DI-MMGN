import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import WordTokenizer, TextIdCache, StaticPageDataset, StaticTextDataset, StaticImageDataset
from model import TextBiLSTM2021, EfficientNetB02021, soft_vote_probs

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
CACHE_DIR=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\cache_v1"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def balanced_indices(ds,n,seed):
    if n is None or n>=len(ds): return list(range(len(ds)))
    rng=random.Random(seed)
    by={0:[],1:[]}
    for i,r in enumerate(ds.rows): by[int(r["label"])].append(i)
    for v in by.values(): rng.shuffle(v)
    n0=n//2; n1=n-n0
    idx=by[0][:n0]+by[1][:n1]
    rng.shuffle(idx)
    return idx

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

@torch.no_grad()
def eval_branch(model,loader,device,kind):
    model.eval(); ys=[]; ps=[]; probs=[]; ids=[]
    for b in loader:
        y=b["label"].to(device)
        x=b["text_ids"].to(device) if kind=="text" else b["image"].to(device)
        lg=model(x); pr=torch.softmax(lg,dim=-1)
        ys.extend(y.cpu().tolist()); ps.extend(pr.argmax(-1).cpu().tolist()); probs.extend(pr.cpu().tolist()); ids.extend(list(b["pair_id"]))
    return metrics(ys,ps),{"pair_id":ids,"label":ys,"probs":probs,"pred":ps}

def train_branch(model,train_loader,val_loader,device,kind,epochs,lr,out_path):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=torch.nn.CrossEntropyLoss()
    best=-1.0; best_epoch=0; hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.0; n=0
        for b in train_loader:
            y=b["label"].to(device)
            x=b["text_ids"].to(device) if kind=="text" else b["image"].to(device)
            opt.zero_grad(set_to_none=True)
            lg=model(x); loss=loss_fn(lg,y)
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite {kind} loss")
            loss.backward(); opt.step()
            total+=float(loss.detach().cpu())*len(y); n+=len(y)
        vm,_=eval_branch(model,val_loader,device,kind)
        row={"epoch":ep,"train_loss":total/max(1,n),**{f"val_{k}":v for k,v in vm.items() if isinstance(v,(int,float))}}
        hist.append(row); print(json.dumps({"branch":kind,**row}),flush=True)
        if vm["macro_f1"]>best:
            best=vm["macro_f1"]; best_epoch=ep
            torch.save({"state_dict":model.state_dict(),"epoch":ep,"val_macro_f1":best},out_path)
    ck=torch.load(out_path,map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"])
    return {"best_epoch":best_epoch,"best_val_macro_f1":best,"history":hist}

@torch.no_grad()
def eval_fusion(text_model,image_model,loader,device):
    text_model.eval(); image_model.eval()
    rows=[]; ys=[]; ps=[]
    for b in loader:
        y=b["label"].to(device)
        tl=text_model(b["text_ids"].to(device)); il=image_model(b["image"].to(device))
        pr=soft_vote_probs(tl,il); pred=pr.argmax(-1)
        for i,pid in enumerate(b["pair_id"]):
            rows.append({
                "pair_id":pid,"label":int(y[i].cpu()),"pred":int(pred[i].cpu()),
                "prob_benign":float(pr[i,0].cpu()),"prob_defaced":float(pr[i,1].cpu()),
                "domain":b["domain"][i],"family_id":b["family_id"][i],
                "attack_type":None if b["attack_type"][i] in ("",None) else b["attack_type"][i],
                "source_type":None if b["source_type"][i] in ("",None) else b["source_type"][i],
            })
        ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
    return metrics(ys,ps),rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--lr-text",type=float,default=1e-3)
    ap.add_argument("--lr-image",type=float,default=1e-4)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--mini-train",type=int,default=None)
    ap.add_argument("--mini-val",type=int,default=None)
    ap.add_argument("--official",action="store_true")
    ap.add_argument("--out",default=None)
    args=ap.parse_args()
    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=pathlib.Path(args.out or (HERE.parent/"runs"/("official" if args.official else "sanity")/f"seed_{args.seed}"))
    out.mkdir(parents=True,exist_ok=True)
    tok=WordTokenizer.load(HERE.parent/"vocab.json")
    text_cache=TextIdCache(CACHE_DIR)
    tr=StaticPageDataset(MANIFEST,"pretrain",tok,128,224,text_id_cache=text_cache)
    va=StaticPageDataset(MANIFEST,"val",tok,128,224,text_id_cache=text_cache)
    te=StaticPageDataset(MANIFEST,"test",tok,128,224,text_id_cache=text_cache)
    tr_text=StaticTextDataset(MANIFEST,"pretrain",tok,128,text_id_cache=text_cache)
    va_text=StaticTextDataset(MANIFEST,"val",tok,128,text_id_cache=text_cache)
    tr_img=StaticImageDataset(MANIFEST,"pretrain",224)
    va_img=StaticImageDataset(MANIFEST,"val",224)
    tri=balanced_indices(tr,args.mini_train,args.seed); vai=balanced_indices(va,args.mini_val,args.seed+1)
    trl_text=DataLoader(Subset(tr_text,tri),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_text=DataLoader(Subset(va_text,vai),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    trl_img=DataLoader(Subset(tr_img,tri),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_img=DataLoader(Subset(va_img,vai),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    val=DataLoader(Subset(va,vai),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    tel=DataLoader(te,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True) if args.official else None
    tm=TextBiLSTM2021(len(tok.word_index)+1).to(device)
    im=EfficientNetB02021(pretrained=True).to(device)
    t0=time.time()
    tres=train_branch(tm,trl_text,val_text,device,"text",args.epochs,args.lr_text,out/"text_best.pt")
    ires=train_branch(im,trl_img,val_img,device,"image",args.epochs,args.lr_image,out/"image_best.pt")
    vm,vrows=eval_fusion(tm,im,val,device)
    with (out/"val_predictions.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for r in vrows:f.write(json.dumps(r,sort_keys=True)+"\n")
    result={"model":"bilstm_efficientnet_2021","fidelity":"high-fidelity architectural reimplementation","seed":args.seed,"official":args.official,
            "train_rows":len(tri),"val_rows":len(vai),"test_rows":len(te) if args.official else 0,
            "config":vars(args),"text":tres,"image":ires,"val_fusion":vm,"elapsed_seconds":time.time()-t0,"device":str(device)}
    if args.official:
        tmets,trows=eval_fusion(tm,im,tel,device)
        with (out/"test_predictions.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for r in trows:f.write(json.dumps(r,sort_keys=True)+"\n")
        result["test_fusion"]=tmets
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","out":str(out),"val_fusion":vm,"test_fusion":result.get("test_fusion")},indent=2),flush=True)
if __name__=="__main__": main()
