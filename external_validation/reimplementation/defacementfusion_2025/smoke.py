import json, pathlib, sys, torch
from transformers import AutoTokenizer

HERE=pathlib.Path(__file__).resolve()
REIMPL=HERE.parents[1]
sys.path.insert(0,str(REIMPL/"common"))
from static_dataset import WordTokenizer, StaticPageDataset
from dom import DOMWindowBuilder
from model import DefacementFusion2025

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_tok=WordTokenizer.load(REIMPL/"bilstm_efficientnet_2021"/"vocab.json")
bert_tok=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
ds=StaticPageDataset(MANIFEST,"pretrain",word_tok,max_len=128,image_size=224)
items=[ds[0],ds[1]]
text_ids=torch.stack([x["text_ids"] for x in items]).to(device)
images=torch.stack([x["image"] for x in items]).to(device)
labels=torch.stack([x["label"] for x in items]).to(device)
paths=[x["html_path"] for x in items]

# Smoke-only reduced window count; official config remains max_windows=4.
builder=DOMWindowBuilder(bert_tok,max_length=512,overlap_tokens=256,max_windows=2)
dom=builder.collate(paths)
for k,v in list(dom.items()):
    if torch.is_tensor(v):
        dom[k]=v.to(device)

model=DefacementFusion2025(
    vocab_size=len(word_tok.word_index)+1,
    d_model=128,
    transformer_layers=1,nhead=4,ffn_dim=256,transformer_dropout=0.1,
    bert_name="bert-base-uncased",pretrained_effnet=True,local_files_only=True
).to(device)
model.train()
logits,feats=model(text_ids,images,dom)
target=torch.nn.functional.one_hot(labels,num_classes=2).float()
loss=torch.nn.BCEWithLogitsLoss()(logits,target)
loss.backward()

def grad_stats(module):
    finite=0; total=0
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            total+=1
            if torch.isfinite(p.grad).all(): finite+=1
    return finite,total

parts={}
for name,module in [("html",model.html),("text",model.text),("image",model.image),("fusion",model.fusion),("head",model.head)]:
    parts[name]=grad_stats(module)

result={
    "gate":"S2_defacementfusion_smoke",
    "status":"PASS" if torch.isfinite(loss).item() and all(v[0]>0 and v[0]==v[1] for v in parts.values()) else "FAIL",
    "device":str(device),
    "pair_ids":[x["pair_id"] for x in items],
    "labels":labels.detach().cpu().tolist(),
    "dom_windows":int(dom["input_ids"].shape[0]),
    "dom_seq_len":int(dom["input_ids"].shape[1]),
    "logits_shape":list(logits.shape),
    "feature_shapes":{k:list(v.shape) for k,v in feats.items()},
    "loss":float(loss.detach().cpu()),
    "gradient_tensors_finite_over_total":{k:list(v) for k,v in parts.items()},
    "trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
    "smoke_note":"max_windows=2 for smoke only; official config specifies max_windows=4"
}
(HERE.parent/"S2_SMOKE.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["status"]=="PASS" else 3)
