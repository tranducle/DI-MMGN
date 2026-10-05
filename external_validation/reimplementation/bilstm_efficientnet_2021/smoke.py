import json, pathlib, sys, torch
from torch.utils.data import DataLoader

HERE=pathlib.Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]/"common"))
from static_dataset import WordTokenizer, StaticPageDataset
from model import TextBiLSTM2021, EfficientNetB02021, soft_vote_probs

MANIFEST=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok=WordTokenizer.load(HERE.parent/"vocab.json")
ds=StaticPageDataset(MANIFEST,"pretrain",tok,max_len=128,image_size=224)
# deterministic real records; batch size 2 keeps BatchNorm valid.
items=[ds[0],ds[1]]
text_ids=torch.stack([x["text_ids"] for x in items]).to(device)
images=torch.stack([x["image"] for x in items]).to(device)
labels=torch.stack([x["label"] for x in items]).to(device)

text_model=TextBiLSTM2021(len(tok.word_index)+1).to(device)
image_model=EfficientNetB02021(pretrained=True).to(device)
text_model.train(); image_model.train()
ce=torch.nn.CrossEntropyLoss()

tl=text_model(text_ids)
il=image_model(images)
loss=ce(tl,labels)+ce(il,labels)
loss.backward()
probs=soft_vote_probs(tl.detach(),il.detach())
pred=probs.argmax(dim=-1)

def grad_count(m):
    return sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in m.parameters() if p.requires_grad)

result={
    "gate":"S1_2021_comparator_smoke",
    "status":"PASS" if torch.isfinite(loss).item() and grad_count(text_model)>0 and grad_count(image_model)>0 else "FAIL",
    "device":str(device),
    "torch":torch.__version__,
    "pair_ids":[x["pair_id"] for x in items],
    "labels":labels.detach().cpu().tolist(),
    "text_logits_shape":list(tl.shape),
    "image_logits_shape":list(il.shape),
    "soft_vote_shape":list(probs.shape),
    "predictions":pred.cpu().tolist(),
    "loss":float(loss.detach().cpu()),
    "text_grad_tensors":grad_count(text_model),
    "image_grad_tensors":grad_count(image_model),
    "text_params":sum(p.numel() for p in text_model.parameters() if p.requires_grad),
    "image_params":sum(p.numel() for p in image_model.parameters() if p.requires_grad),
}
out=HERE.parent/"S1_SMOKE.json"
out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["status"]=="PASS" else 3)
