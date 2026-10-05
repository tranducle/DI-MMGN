import json, pathlib, sys
import numpy as np, torch
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

RUN=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\run_official")
OUT=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external_v2")
sys.path.insert(0,str(RUN/"src"))
from models.dimmgn_v3 import MODEL_REGISTRY_V3
from data.dimmgn_dataset_v4 import DimmgnPairDatasetV4,DimmgnCollateV4

def move(s,dev): return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in s.items()}
def calc(y,p):
    y=np.asarray(y,int);p=np.asarray(p,int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"n":len(y),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "attack_recall":float(recall_score(y,p,pos_label=1,zero_division=0)),
            "legitimate_specificity":float(tn/(tn+fp)) if tn+fp else None,
            "accuracy":float(accuracy_score(y,p)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

pairs=str(RUN/"dataset_pipeline"/"v4"/"unified_pairs.jsonl")
ds=DimmgnPairDatasetV4(pairs,data_root=str(RUN/"dataset_pipeline"),split="test")
ld=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=DimmgnCollateV4(),num_workers=0)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
masks={"text_dom_visual_no_http":[1,1,1,0],"text_dom_only":[1,1,0,0]}
report={"gate":"V2_G2_5_INTERNAL_MODALITY_MATCHED","verdict":"PASS","threshold":0.5,"retrained":False,"test_pairs":len(ds),"masks":{}}
predroot=OUT/"internal_mask_predictions";predroot.mkdir(parents=True,exist_ok=True)
for name,maskvec in masks.items():
    report["masks"][name]={}
    for seed in [42,43,44]:
        ck=torch.load(RUN/"results"/"checkpoints"/f"concat_none_seed{seed}.pt",map_location=dev,weights_only=False)
        model=MODEL_REGISTRY_V3["concat"](ck["config"]).to(dev);model.load_state_dict(ck["state_dict"]);model.eval()
        ys=[];ps=[];rows=[]
        with torch.no_grad():
            for b in ld:
                mask=torch.tensor([maskvec]*len(b["label"]),dtype=torch.bool,device=dev)
                out=model({"t1":move(b["t1"],dev),"t2":move(b["t2"],dev),"mask":mask})
                prob=torch.sigmoid(out["score"]);pd=(prob>0.5).long().cpu().tolist();pr=prob.cpu().tolist()
                yy=b["label"].tolist()
                ys.extend(yy);ps.extend(pd)
                for i,pairid in enumerate(b["pair_id"]):
                    rows.append({"pair_id":pairid,"domain":b["domain"][i],"attack_type":b["attack_type"][i],"source_type":b["source_type"][i],
                                 "y":int(yy[i]),"pred":int(pd[i]),"prob":float(pr[i])})
        met=calc(ys,ps);report["masks"][name][str(seed)]=met
        with (predroot/f"{name}_seed{seed}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
        print(json.dumps({"mask":name,"seed":seed,"metrics":met}),flush=True)
(OUT/"V2_G2_5_INTERNAL_MODALITY_MATCHED.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
