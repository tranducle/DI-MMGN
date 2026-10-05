import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def move_snap(s,device):
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in s.items()}

def finite_grads(model):
    n=0
    for p in model.parameters():
        if p.grad is not None:
            n+=1
            if not torch.isfinite(p.grad).all(): return False,n
    return n>0,n

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.run_root).resolve(); out=Path(a.out).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    src=root/"src"; sys.path.insert(0,str(src))
    report={"gate_id":"FB4_smoke","run_root":str(root),"checks":{},"errors":[],"verdict":"FAIL_REPAIR"}
    pair=root/"dataset_pipeline"/"v4"/"unified_pairs.jsonl"
    rows=[json.loads(x) for x in pair.read_text(encoding="utf-8").splitlines() if x.strip()]
    split_counts={s:sum(1 for r in rows if r["split"]==s) for s in ("pretrain","val","test")}
    report["checks"]["nonempty_splits"]={"counts":split_counts,"pass":all(v>0 for v in split_counts.values())}
    fam={s:set(r["family_id"] for r in rows if r["split"]==s) for s in split_counts}
    dom={s:set(r["domain"] for r in rows if r["split"]==s) for s in split_counts}
    fo={};do={}
    order=["pretrain","val","test"]
    for i,x in enumerate(order):
        for y in order[i+1:]:
            fo[f"{x}__{y}"]=sorted(fam[x]&fam[y]); do[f"{x}__{y}"]=sorted(dom[x]&dom[y])
    report["checks"]["zero_family_overlap"]={"overlaps":fo,"pass":all(not v for v in fo.values())}
    report["checks"]["zero_domain_overlap"]={"overlaps":do,"pass":all(not v for v in do.values())}

    from data.dimmgn_dataset import DimmgnPairDatasetV4,DimmgnCollateV4
    from models.dimmgn_v3 import MODEL_REGISTRY_V3
    from models.losses_v3 import composite_loss_v3
    from models.baseline_gnn import BaselineGraphSAGE
    cfg=yaml.safe_load((src/"config"/"experiment_v3.yaml").read_text(encoding="utf-8-sig"))
    expected=cfg["protocol"]["official_pairs_sha256"]
    observed=sha(pair)
    report["checks"]["pair_sha256"]={"observed":observed,"expected":expected,"pass":observed==expected}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"]={"type":str(device),"name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

    loaders={}
    collate=DimmgnCollateV4()
    for s in order:
        ds=DimmgnPairDatasetV4(str(pair),data_root=str(root/"dataset_pipeline"),split=s)
        _=ds[0]; _=ds[len(ds)-1]
        loaders[s]=DataLoader(ds,batch_size=4,shuffle=False,collate_fn=collate,num_workers=0)
    b=next(iter(loaders["pretrain"]))
    report["checks"]["four_modality_mask"]={"shape":list(b["mask"].shape),"all_true":bool(torch.all(b["mask"]).item()),"pass":tuple(b["mask"].shape)==(4,4) and bool(torch.all(b["mask"]).item())}
    batch={"t1":move_snap(b["t1"],device),"t2":move_snap(b["t2"],device),"mask":b["mask"].to(device)}
    y=b["label"].to(device)
    m=MODEL_REGISTRY_V3["concat"](cfg).to(device); m.train()
    o=m(batch); loss=composite_loss_v3(o["benign_proj"],o["score"],y,o["ortho_residual"],cfg); m.zero_grad(set_to_none=True); loss.backward()
    fg,ng=finite_grads(m); report["checks"]["concat_forward_backward"]={"loss":float(loss.detach().cpu()),"gradient_tensors":ng,"pass":bool(torch.isfinite(loss).item()) and fg and bool(torch.isfinite(o["score"]).all().item())}
    del m,o,loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    m=BaselineGraphSAGE(cfg).to(device); m.train(); logits=m(batch["t2"]); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y.float()); m.zero_grad(set_to_none=True); loss.backward()
    fg,ng=finite_grads(m); report["checks"]["graphsage_forward_backward"]={"loss":float(loss.detach().cpu()),"gradient_tensors":ng,"pass":bool(torch.isfinite(loss).item()) and fg and bool(torch.isfinite(logits).all().item())}
    del m,logits,loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    tbp=src/"run_v3_text_baseline.py"; spec=importlib.util.spec_from_file_location("tbv4",tbp); tb=importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
    ds=tb.TextSeqDataset("pretrain"); dl=DataLoader(ds,batch_size=2,shuffle=False,collate_fn=tb.collate,num_workers=0); q=next(iter(dl))
    tm=tb.TextBiLSTM(hidden=128,dropout=float(cfg["model"]["dropout"])).to(device); x=q["seq"].to(device); mask=q["mask"].to(device); yy=q["label"].float().to(device); z=tm(x,mask); loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yy); tm.zero_grad(set_to_none=True); loss.backward()
    fg,ng=finite_grads(tm); report["checks"]["text_bilstm_forward_backward"]={"loss":float(loss.detach().cpu()),"gradient_tensors":ng,"pass":bool(torch.isfinite(loss).item()) and fg and bool(torch.isfinite(z).all().item())}
    mandatory=[x.get("pass",False) for x in report["checks"].values()]
    report["verdict"]="PASS" if all(mandatory) and not report["errors"] else "FAIL_REPAIR"
    report["allowed_next_step"]="launch_30_run_matrix" if report["verdict"]=="PASS" else "repair_smoke_failure"
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2))
    return 0 if report["verdict"]=="PASS" else 3
if __name__=="__main__": raise SystemExit(main())
