import pathlib, importlib.util
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM

HERE=pathlib.Path(__file__).resolve()
_impl_path=HERE.parents[1]/"bilstm_efficientnet_2021"/"model.py"
_spec=importlib.util.spec_from_file_location("bilstm_effnet_2021_model",_impl_path)
_impl=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
TextBiLSTM2021=_impl.TextBiLSTM2021
EfficientNetB02021=_impl.EfficientNetB02021

class HTMLDefaceMLM(nn.Module):
    def __init__(self,model_name="bert-base-uncased",tag_buckets=128,depth_buckets=32,node_buckets=512,
                 local_files_only=False):
        super().__init__()
        self.mlm=AutoModelForMaskedLM.from_pretrained(model_name,local_files_only=local_files_only)
        h=self.mlm.config.hidden_size
        self.tag_emb=nn.Embedding(tag_buckets,h)
        self.depth_emb=nn.Embedding(depth_buckets,h)
        self.node_emb=nn.Embedding(node_buckets,h)
        nn.init.normal_(self.tag_emb.weight,std=0.02)
        nn.init.normal_(self.depth_emb.weight,std=0.02)
        nn.init.normal_(self.node_emb.weight,std=0.02)
        with torch.no_grad():
            self.tag_emb.weight[0].zero_(); self.depth_emb.weight[0].zero_(); self.node_emb.weight[0].zero_()

    def forward(self,batch,labels):
        word=self.mlm.bert.embeddings.word_embeddings(batch["input_ids"])
        x=word+self.tag_emb(batch["tag_ids"])+self.depth_emb(batch["depth_ids"])+self.node_emb(batch["node_ids"])
        return self.mlm(inputs_embeds=x,attention_mask=batch["attention_mask"],labels=labels)

    def export_html_encoder_state(self):
        out={}
        for k,v in self.mlm.bert.state_dict().items(): out["bert."+k]=v.detach().cpu()
        for prefix,module in [("tag_emb",self.tag_emb),("depth_emb",self.depth_emb),("node_emb",self.node_emb)]:
            for k,v in module.state_dict().items(): out[prefix+"."+k]=v.detach().cpu()
        return out

class HTMLDefaceEncoder(nn.Module):
    def __init__(self,model_name="bert-base-uncased",d_model=128,
                 tag_buckets=128,depth_buckets=32,node_buckets=512,
                 local_files_only=False):
        super().__init__()
        self.bert=AutoModel.from_pretrained(model_name,local_files_only=local_files_only,add_pooling_layer=False)
        h=self.bert.config.hidden_size
        self.tag_emb=nn.Embedding(tag_buckets,h)
        self.depth_emb=nn.Embedding(depth_buckets,h)
        self.node_emb=nn.Embedding(node_buckets,h)
        nn.init.normal_(self.tag_emb.weight,std=0.02)
        nn.init.normal_(self.depth_emb.weight,std=0.02)
        nn.init.normal_(self.node_emb.weight,std=0.02)
        with torch.no_grad():
            self.tag_emb.weight[0].zero_(); self.depth_emb.weight[0].zero_(); self.node_emb.weight[0].zero_()
        self.proj=nn.Linear(h,d_model)

    def forward(self,batch):
        ids=batch["input_ids"]
        word=self.bert.embeddings.word_embeddings(ids)
        x=word+self.tag_emb(batch["tag_ids"])+self.depth_emb(batch["depth_ids"])+self.node_emb(batch["node_ids"])
        out=self.bert(inputs_embeds=x,attention_mask=batch["attention_mask"]).last_hidden_state[:,0]
        owner=batch["owner"]
        B=int(batch["batch_size"])
        pooled=torch.zeros((B,out.size(-1)),device=out.device,dtype=out.dtype)
        pooled.index_add_(0,owner,out)
        cnt=torch.bincount(owner,minlength=B).to(out.device).clamp_min(1).unsqueeze(1)
        pooled=pooled/cnt
        return self.proj(pooled)

class DefacementFusion2025(nn.Module):
    def __init__(self,vocab_size,d_model=128,text_embed_dim=64,text_hidden=64,
                 spatial_dropout=0.2,transformer_layers=1,nhead=4,ffn_dim=256,
                 transformer_dropout=0.1,bert_name="bert-base-uncased",
                 pretrained_effnet=True,local_files_only=False):
        super().__init__()
        self.html=HTMLDefaceEncoder(bert_name,d_model,local_files_only=local_files_only)
        self.text=TextBiLSTM2021(vocab_size,text_embed_dim,text_hidden,spatial_dropout)
        self.image=EfficientNetB02021(pretrained_effnet)
        text_dim=text_hidden*2
        self.text_proj=nn.Identity() if text_dim==d_model else nn.Linear(text_dim,d_model)
        self.image_proj=nn.Identity() if d_model==128 else nn.Linear(128,d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                                         dim_feedforward=ffn_dim,dropout=transformer_dropout,
                                         batch_first=True,activation="relu")
        self.fusion=nn.TransformerEncoder(layer,num_layers=transformer_layers)
        self.head=nn.Sequential(nn.Linear(d_model,256),nn.ReLU(),nn.Linear(256,2))

    def forward(self,text_ids,image,dom_batch):
        h=self.html(dom_batch)
        t=self.text_proj(self.text.features(text_ids))
        i=self.image_proj(self.image.features(image))
        mods=torch.stack([h,t,i],dim=1)
        z=self.fusion(mods).mean(dim=1)
        return self.head(z),{"html":h,"text":t,"image":i,"fused":z}
