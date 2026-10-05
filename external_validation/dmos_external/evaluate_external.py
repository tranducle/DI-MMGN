import json, pathlib, sys, statistics
import numpy as np, torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

EXT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external")
RUN=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\run_official")
sys.path.insert(0,str(RUN/"src"))
from models.dimmgn_v3 import MODEL_REGISTRY_V3
from data.dimmgn_dataset_v4 import DimmgnPairDatasetV4,DimmgnCollateV4

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,int)
    if len(np.unique(y))==1:
        return {"n":len(y),"attack_recall":float(recall_score(y,p,pos_label=1,zero_division=0)),
                "defaced_f1":float(f1_score(y,p,pos_label=1,zero_division=0)),
                "precision":float(precision_score(y,p,pos_label=1,zero_division=0)),
                "accuracy":float(accuracy_score(y,p)),
                "macro_f1":None,"legitimate_specificity":None}
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"n":len(y),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "defaced_f1":float(f1_score(y,p,pos_label=1,zero_division=0)),
            "precision":float(precision_score(y,p,pos_label=1,zero_division=0)),
            "attack_recall":float(recall_score(y,p,pos_label=1,zero_division=0)),
            "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else None,
            "accuracy":float(accuracy_score(y,p)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def load_graph(path):
    g=torch.load(path,map_location="cpu",weights_only=False)
    d=Data(x=g["x"].float(),edge_index=g["edge_index"].long());d.num_nodes=int(g["num_nodes"]);return d

def snap_from_feat(r,visual_mode):
    text=torch.from_numpy(np.load(r["text_path"]).astype(np.float32))
    g=load_graph(r["graph_path"])
    vis=torch.from_numpy(np.load(r["visual_path"]).astype(np.float32)) if visual_mode=="text_dom_visual" else torch.zeros(512)
    http=torch.zeros(32)
    return text,g,vis,http

def ext_batches(pairs,features,batch_size,visual_mode):
    for st in range(0,len(pairs),batch_size):
        qs=pairs[st:st+batch_size]
        s1=[];s2=[]
        for q in qs:
            s1.append(snap_from_feat(features[q["domain"]+"|"+q["t1"]],visual_mode))
            s2.append(snap_from_feat(features[q["domain"]+"|"+q["t2"]],visual_mode))
        def pack(ss):
            gb=Batch.from_data_list([x[1] for x in ss])
            return {"text_embedding":torch.stack([x[0] for x in ss]),"dom_x":gb.x,"dom_edge_index":gb.edge_index,
                    "dom_batch":gb.batch,"visual":torch.stack([x[2] for x in ss]),"http":torch.stack([x[3] for x in ss])}
        mask=[1,1,1,0] if visual_mode=="text_dom_visual" else [1,1,0,0]
        yield qs,pack(s1),pack(s2),torch.tensor([mask]*len(qs),dtype=torch.bool)

def move(s,dev):return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in s.items()}

@torch.no_grad()
def eval_external(model,pairs,features,dev,visual_mode):
    ys=[];ps=[];probs=[];rows=[]
    model.eval()
    for qs,t1,t2,mask in ext_batches(pairs,features,16,visual_mode):
        out=model({"t1":move(t1,dev),"t2":move(t2,dev),"mask":mask.to(dev)})
        pr=torch.sigmoid(out["score"]);pd=(pr>0.5).long().cpu().tolist()
        pro=pr.cpu().tolist()
        for i,q in enumerate(qs):
            y=0 if q["label"]=="legitimate" else 1
            ys.append(y);ps.append(pd[i]);probs.append(pro[i])
            rows.append({**q,"y":y,"pred":pd[i],"prob":pro[i]})
    return metrics(ys,ps),rows

@torch.no_grad()
def eval_internal_masked(model,dev,mask_vec):
    pairs=str(RUN/"dataset_pipeline"/"v4"/"unified_pairs.jsonl")
    ds=DimmgnPairDatasetV4(pairs,data_root=str(RUN/"dataset_pipeline"),split="test")
    ld=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=DimmgnCollateV4(),num_workers=0)
    ys=[];ps=[]
    model.eval()
    for b in ld:
        mask=torch.tensor([mask_vec]*len(b["label"]),dtype=torch.bool,device=dev)
        out=model({"t1":move(b["t1"],dev),"t2":move(b["t2"],dev),"mask":mask})
        pr=torch.sigmoid(out["score"]);pd=(pr>0.5).long()
        ys.extend(b["label"].tolist());ps.extend(pd.cpu().tolist())
    return metrics(ys,ps)

def main():
    g3=json.loads((EXT/"G3_EXTERNAL_FEATURES.json").read_text(encoding="utf-8"))
    if g3.get("verdict")!="PASS":raise SystemExit("G3 not PASS")
    visual_mode=g3["visual_mode"]
    pairs=[json.loads(x) for x in (EXT/"external_eval_pairs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    feats={x["domain"]+"|"+x["snapshot_id"]:x for x in [json.loads(z) for z in (EXT/"external_feature_manifest.jsonl").read_text(encoding="utf-8").splitlines() if z.strip()]}
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask=[1,1,1,0] if visual_mode=="text_dom_visual" else [1,1,0,0]
    allres={};outdir=EXT/"external_predictions";outdir.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        ck=torch.load(RUN/"results"/"checkpoints"/f"concat_none_seed{seed}.pt",map_location=dev,weights_only=False)
        model=MODEL_REGISTRY_V3["concat"](ck["config"]).to(dev);model.load_state_dict(ck["state_dict"])
        em,rows=eval_external(model,pairs,feats,dev,visual_mode)
        im=eval_internal_masked(model,dev,mask)
        with (outdir/f"dmos_external_seed{seed}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
        allres[str(seed)]={"external":em,"internal_modality_matched":im}
        print(json.dumps({"seed":seed,"external":em,"internal_modality_matched":im}),flush=True)
    report={"gate":"G4_EXTERNAL_FROZEN_EVAL","verdict":"PASS","visual_mode":visual_mode,"mask":mask,
            "threshold":0.5,"retrained_on_dmos":False,"seeds":allres,
            "external_pairs":len(pairs),"external_positive":sum(q["label"]=="defaced" for q in pairs),
            "external_benign":sum(q["label"]=="legitimate" for q in pairs)}
    (EXT/"G4_EXTERNAL_FROZEN_EVAL.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
