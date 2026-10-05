import sys,yaml,torch
from pathlib import Path
from torch.utils.data import DataLoader
ROOT=Path(r"D:\RESEARCH\DI_MM_V4\run_official")
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))
from data.dimmgn_dataset import DimmgnPairDatasetV4,DimmgnCollateV4
from models.baseline_gnn import BaselineGraphSAGE,BaselineGAT,BaselineGIN
from models.encoders_v3 import MultiModalEncoderV3
import torch.nn as nn

class SnapshotMMV3(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=int(cfg["model"]["d"]); p=float(cfg["model"].get("dropout",0.0))
        self.encoder=MultiModalEncoderV3(cfg)
        self.head=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(p),nn.Linear(d,1))
    def forward(self,batch):
        e2=self.encoder(batch["t2"],batch["mask"])
        return self.head(e2).squeeze(-1)

def move_snap(snap,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in snap.items()}

cfg=yaml.safe_load((SRC/"config"/"experiment_v3.yaml").read_text(encoding="utf-8-sig"))
pairs=str((ROOT/"dataset_pipeline"/"v4"/"unified_pairs.jsonl").resolve())
data_root=str((ROOT/"dataset_pipeline").resolve())
ds=DimmgnPairDatasetV4(pairs,data_root=data_root,split="pretrain")
loader=DataLoader(ds,batch_size=4,shuffle=False,collate_fn=DimmgnCollateV4(),num_workers=0)
b=next(iter(loader))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
t2=move_snap(b["t2"],device); mask=b["mask"].to(device)
models=[("graphsage",BaselineGraphSAGE(cfg),False),("gat",BaselineGAT(cfg),False),("gin",BaselineGIN(cfg),False),("snapshot_mm",SnapshotMMV3(cfg),True)]
for name,m,is_mm in models:
    m=m.to(device).train()
    out=m({"t2":t2,"mask":mask}) if is_mm else m(t2)
    if out.shape!=(4,) or not torch.isfinite(out).all():
        raise RuntimeError(f"{name} bad output {out.shape}")
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out,b["label"].float().to(device))
    m.zero_grad(set_to_none=True); loss.backward()
    grads=sum(1 for p in m.parameters() if p.grad is not None)
    if grads==0: raise RuntimeError(f"{name} no gradients")
    print(f"{name}|PASS|loss={float(loss):.6f}|grads={grads}",flush=True)
print("STATIC_MODEL_PREFLIGHT=PASS",flush=True)
