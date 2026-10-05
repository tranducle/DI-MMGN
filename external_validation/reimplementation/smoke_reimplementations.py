import json, pathlib, sys, torch
from transformers import AutoTokenizer

BASE=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\reimplementation")
sys.path.insert(0,str(BASE/"common"))
sys.path.insert(0,str(BASE/"defacementfusion_2025"))

from static_dataset import WordTokenizer, StaticPageDataset, load_manifest, visible_text_from_html
import importlib.util

spec21=importlib.util.spec_from_file_location("impl2021",BASE/"bilstm_efficientnet_2021"/"model.py")
impl2021=importlib.util.module_from_spec(spec21); spec21.loader.exec_module(impl2021)
TextBiLSTM2021=impl2021.TextBiLSTM2021
EfficientNetB02021=impl2021.EfficientNetB02021
soft_vote_probs=impl2021.soft_vote_probs

spec=importlib.util.spec_from_file_location("df_model",BASE/"defacementfusion_2025"/"model.py")
df_model=importlib.util.module_from_spec(spec); spec.loader.exec_module(df_model)
from dom import DOMWindowBuilder

MAN=r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl"
rows=load_manifest(MAN,"pretrain")[:2]
tok=WordTokenizer().fit(visible_text_from_html(r["html_path"]) for r in rows)
ds=StaticPageDataset(MAN,"pretrain",tok,max_len=128,image_size=224,cache_text=False)
batch=[ds[0],ds[1]]
ids=torch.stack([x["text_ids"] for x in batch])
images=torch.stack([x["image"] for x in batch])
labels=torch.stack([x["label"] for x in batch])

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ids=ids.to(device); images=images.to(device); labels=labels.to(device)

# 2021 branch smoke.
tm=TextBiLSTM2021(len(tok.word_index)+1).to(device)
im=EfficientNetB02021(pretrained=True).to(device)
tm.train(); im.train()
tl=tm(ids); il=im(images)
loss=torch.nn.functional.cross_entropy(tl,labels)+torch.nn.functional.cross_entropy(il,labels)
loss.backward()
probs=soft_vote_probs(tl.detach(),il.detach())
print(json.dumps({
    "bilstm_effnet_2021":{
        "text_logits":list(tl.shape),"image_logits":list(il.shape),
        "soft_vote":list(probs.shape),"loss":float(loss.detach().cpu()),
        "finite":bool(torch.isfinite(tl).all() and torch.isfinite(il).all())
    }
},indent=2),flush=True)
del tm,im,tl,il,loss,probs
if torch.cuda.is_available(): torch.cuda.empty_cache()

# DefacementFusion smoke with shortened DOM windows to validate plumbing.
bert_tok=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
dom_builder=DOMWindowBuilder(bert_tok,max_length=128,overlap_tokens=64,max_windows=1)
dom=dom_builder.collate([x["html_path"] for x in batch])
for k,v in list(dom.items()):
    if torch.is_tensor(v): dom[k]=v.to(device)
model=df_model.DefacementFusion2025(
    vocab_size=len(tok.word_index)+1,
    d_model=128,transformer_layers=1,nhead=4,ffn_dim=256,
    bert_name="bert-base-uncased",pretrained_effnet=True,local_files_only=True
).to(device).train()
logits,feats=model(ids,images,dom)
target=torch.nn.functional.one_hot(labels,num_classes=2).float()
loss2=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
loss2.backward()
print(json.dumps({
    "defacementfusion_2025":{
        "logits":list(logits.shape),
        "html":list(feats["html"].shape),
        "text":list(feats["text"].shape),
        "image":list(feats["image"].shape),
        "fused":list(feats["fused"].shape),
        "dom_windows":int(dom["input_ids"].shape[0]),
        "dom_length":int(dom["input_ids"].shape[1]),
        "loss":float(loss2.detach().cpu()),
        "finite":bool(torch.isfinite(logits).all())
    },
    "device":str(device),
    "status":"PASS"
},indent=2),flush=True)
