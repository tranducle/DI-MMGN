import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT=Path(r"D:\RESEARCH\DI_MM_V4")
V4=ROOT/"dataset_pipeline"/"v4"
rows=[json.loads(x) for x in (V4/"snapshot_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows.sort(key=lambda r:(r["domain"],r["snapshot_id"]))
rows=rows[:4]
paths=[V4/"screenshots"/r["domain"]/f"{r['snapshot_id']}.png" for r in rows]
assert all(p.exists() for p in paths),paths
imgs=[]
for p in paths:
    with Image.open(p) as im:
        imgs.append(im.convert("RGB").copy())
MODEL_ID="openai/clip-vit-base-patch32"
COMMIT="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=COMMIT,use_fast=False)
model=CLIPModel.from_pretrained(MODEL_ID,revision=COMMIT).eval().to(device)
inputs=processor(images=imgs,return_tensors="pt")
pix=inputs["pixel_values"].to(device)
with torch.inference_mode():
    feats=model.get_image_features(pixel_values=pix)
    feats=feats/feats.norm(dim=-1,keepdim=True).clamp(min=1e-12)
arr=feats.detach().float().cpu().numpy()
print({"shape":list(arr.shape),"finite":bool(np.isfinite(arr).all()),"norms":np.linalg.norm(arr,axis=1).round(6).tolist(),"device":str(device)},flush=True)
raise SystemExit(0 if arr.shape==(4,512) and np.isfinite(arr).all() else 3)
