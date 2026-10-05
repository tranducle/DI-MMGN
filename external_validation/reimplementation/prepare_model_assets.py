from transformers import AutoTokenizer, AutoModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
print("ASSET_PREP_START",flush=True)
tok=AutoTokenizer.from_pretrained("bert-base-uncased")
print("TOKENIZER_OK",len(tok),flush=True)
m=AutoModel.from_pretrained("bert-base-uncased")
print("BERT_OK",sum(p.numel() for p in m.parameters()),flush=True)
del m
if torch.cuda.is_available(): torch.cuda.empty_cache()
e=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print("EFFNET_OK",sum(p.numel() for p in e.parameters()),flush=True)
print("ASSET_PREP_PASS",flush=True)
