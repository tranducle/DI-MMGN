import argparse, hashlib, json, math, pathlib, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertForMaskedLM, AutoTokenizer

HERE=pathlib.Path(__file__).resolve()
CACHE_DIR=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\cache_v1")
MANIFEST=pathlib.Path(r"D:\RESEARCH\DI_MM_V4\external_candidates\sota_static\lwded_v4_static_current_manifest.jsonl")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

class CachedPretrainWindows(Dataset):
    def __init__(self,cache_dir=CACHE_DIR,manifest=MANIFEST):
        d=pathlib.Path(cache_dir)
        summary=json.loads((d/"cache_summary.json").read_text(encoding="utf-8"))
        if summary.get("status")!="PASS": raise RuntimeError("input cache is not PASS")
        rows=[json.loads(x) for x in pathlib.Path(manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
        pair_ids=json.loads((d/"pair_ids.json").read_text(encoding="utf-8"))
        if [r["pair_id"] for r in rows] != pair_ids:
            raise RuntimeError("cache pair-id order does not match frozen manifest")
        if summary.get("manifest_sha256") != sha256(manifest):
            raise RuntimeError("cache manifest SHA does not match frozen manifest")
        self.summary=summary
        self.rows=rows
        self.ids=np.load(d/"dom_input_ids.npy",mmap_mode="r")
        self.att=np.load(d/"dom_attention.npy",mmap_mode="r")
        self.tags=np.load(d/"dom_tag_ids.npy",mmap_mode="r")
        self.depth=np.load(d/"dom_depth_ids.npy",mmap_mode="r")
        self.nodes=np.load(d/"dom_node_ids.npy",mmap_mode="r")
        self.count=np.load(d/"dom_window_count.npy",mmap_mode="r")
        self.index=[]
        for ri,r in enumerate(rows):
            if r["split"]!="pretrain": continue
            for wi in range(int(self.count[ri])):
                self.index.append((ri,wi))
        self.pretrain_row_count=sum(r["split"]=="pretrain" for r in rows)
        self.pretrain_family_count=len({r["family_id"] for r in rows if r["split"]=="pretrain"})
        self.pretrain_domain_count=len({r["domain"] for r in rows if r["split"]=="pretrain"})
    def __len__(self): return len(self.index)
    def __getitem__(self,i):
        ri,wi=self.index[i]
        return (
            torch.from_numpy(np.asarray(self.ids[ri,wi],dtype=np.int64).copy()),
            torch.from_numpy(np.asarray(self.att[ri,wi],dtype=np.int64).copy()),
            torch.from_numpy(np.asarray(self.tags[ri,wi],dtype=np.int64).copy()),
            torch.from_numpy(np.asarray(self.depth[ri,wi],dtype=np.int64).copy()),
            torch.from_numpy(np.asarray(self.nodes[ri,wi],dtype=np.int64).copy()),
        )

class HTMLDefaceMLM(nn.Module):
    def __init__(self,model_name="bert-base-uncased",tag_buckets=128,depth_buckets=32,node_buckets=512):
        super().__init__()
        self.mlm=BertForMaskedLM.from_pretrained(model_name,local_files_only=True)
        h=self.mlm.config.hidden_size
        self.tag_emb=nn.Embedding(tag_buckets,h)
        self.depth_emb=nn.Embedding(depth_buckets,h)
        self.node_emb=nn.Embedding(node_buckets,h)
        nn.init.normal_(self.tag_emb.weight,std=0.02)
        nn.init.normal_(self.depth_emb.weight,std=0.02)
        nn.init.normal_(self.node_emb.weight,std=0.02)
        with torch.no_grad():
            self.tag_emb.weight[0].zero_(); self.depth_emb.weight[0].zero_(); self.node_emb.weight[0].zero_()

    def forward(self,input_ids,attention_mask,tag_ids,depth_ids,node_ids,labels):
        word=self.mlm.bert.embeddings.word_embeddings(input_ids)
        emb=word+self.tag_emb(tag_ids)+self.depth_emb(depth_ids)+self.node_emb(node_ids)
        return self.mlm(inputs_embeds=emb,attention_mask=attention_mask,labels=labels)

    def html_encoder_state(self):
        out={f"bert.{k}":v.detach().cpu() for k,v in self.mlm.bert.state_dict().items()}
        out["tag_emb.weight"]=self.tag_emb.weight.detach().cpu()
        out["depth_emb.weight"]=self.depth_emb.weight.detach().cpu()
        out["node_emb.weight"]=self.node_emb.weight.detach().cpu()
        return out

def node_aware_mask(input_ids,attention,node_ids,tokenizer,generator):
    """
    DefacementFusion paper masking:
    sample DOM nodes iteratively and include every token in each selected node
    until masked-token count reaches/exceeds 15%, then BERT 80/10/10 replacement.
    """
    x=input_ids.clone()
    labels=torch.full_like(input_ids,-100)
    total_masked=0
    for b in range(x.size(0)):
        valid=attention[b].bool() & (node_ids[b]>0)
        pos=torch.nonzero(valid,as_tuple=False).flatten()
        if pos.numel()==0: continue
        target=max(1,int(math.ceil(0.15*pos.numel())))
        uniq=torch.unique(node_ids[b,pos])
        order=torch.randperm(len(uniq),generator=generator)
        chosen=[]; count=0
        for j in order.tolist():
            nid=uniq[j]
            p=torch.nonzero(valid & (node_ids[b]==nid),as_tuple=False).flatten()
            chosen.append(p); count+=int(p.numel())
            if count>=target: break
        mpos=torch.cat(chosen) if chosen else pos[:1]
        labels[b,mpos]=input_ids[b,mpos]
        total_masked+=int(mpos.numel())
        r=torch.rand(mpos.numel(),generator=generator)
        p80=mpos[r<0.8]
        p10=mpos[(r>=0.8)&(r<0.9)]
        x[b,p80]=tokenizer.mask_token_id
        if p10.numel():
            x[b,p10]=torch.randint(0,tokenizer.vocab_size,(p10.numel(),),generator=generator,dtype=x.dtype)
    return x,labels,total_masked

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-dir",default=str(CACHE_DIR))
    ap.add_argument("--manifest",default=str(MANIFEST))
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--physical-batch-size",type=int,default=4)
    ap.add_argument("--accum-steps",type=int,default=4)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--weight-decay",type=float,default=0.01)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--limit-windows",type=int,default=None)
    ap.add_argument("--out",default=str(HERE.parent/"pretrain"/"html_deface_mlm_final.pt"))
    args=ap.parse_args()

    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=CachedPretrainWindows(args.cache_dir,args.manifest)
    idx=list(range(len(ds)))
    if args.limit_windows is not None:
        rng=random.Random(args.seed); rng.shuffle(idx); idx=idx[:min(args.limit_windows,len(idx))]
    subset=Subset(ds,idx)
    loader=DataLoader(subset,batch_size=args.physical_batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    tok=AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True)
    model=HTMLDefaceMLM().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    amp=device.type=="cuda"; scaler=torch.amp.GradScaler("cuda",enabled=amp)
    history=[]; t0=time.time(); opt.zero_grad(set_to_none=True)

    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0.; nseq=0; masked_total=0; opt_steps=0
        gen=torch.Generator(device="cpu"); gen.manual_seed(args.seed+epoch*100003)
        for step,batch in enumerate(loader,1):
            ids,attn,tags,depths,nodes=[x.long() for x in batch]
            masked,labels,nmasked=node_aware_mask(ids,attn,nodes,tok,gen)
            masked_total+=nmasked
            masked=masked.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
            attn=attn.to(device,non_blocking=True); tags=tags.to(device,non_blocking=True)
            depths=depths.to(device,non_blocking=True); nodes=nodes.to(device,non_blocking=True)
            with torch.amp.autocast("cuda",enabled=amp):
                out=model(masked,attn,tags,depths,nodes,labels)
                loss=out.loss/args.accum_steps
            if not torch.isfinite(loss): raise RuntimeError("nonfinite MLM loss")
            scaler.scale(loss).backward()
            if step%args.accum_steps==0 or step==len(loader):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); opt_steps+=1
            loss_sum+=float(loss.detach().cpu())*args.accum_steps*len(ids); nseq+=len(ids)
            if step%50==0 or step==len(loader):
                print(f"MLM_PROGRESS epoch={epoch}/{args.epochs} step={step}/{len(loader)} avg_loss={loss_sum/max(1,nseq):.6f} masked={masked_total} elapsed={time.time()-t0:.1f}s",flush=True)
        row={"epoch":epoch,"avg_mlm_loss":loss_sum/max(1,nseq),"sequences":nseq,
             "masked_tokens":masked_total,"optimizer_steps":opt_steps,"elapsed_seconds":time.time()-t0}
        history.append(row); print("MLM_EPOCH "+json.dumps(row),flush=True)

    outp=pathlib.Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True)
    cache_summary=json.loads((pathlib.Path(args.cache_dir)/"cache_summary.json").read_text(encoding="utf-8"))
    ck={"html_encoder":model.html_encoder_state(),"mlm_head":model.mlm.cls.state_dict(),
        "training":{
            "paper_specified":{"epochs":args.epochs,"effective_batch_size":args.physical_batch_size*args.accum_steps,
                               "lr":args.lr,"mask_probability":0.15,"mask_policy":"DOM-node-aware + BERT 80/10/10"},
            "implementation_assumptions":{"optimizer":"AdamW","weight_decay":args.weight_decay,"mixed_precision":amp},
            "seed":args.seed,"cache_dir":args.cache_dir,"cache_manifest_sha256":cache_summary["manifest_sha256"],
            "pretrain_rows":ds.pretrain_row_count,"pretrain_families":ds.pretrain_family_count,
            "pretrain_domains":ds.pretrain_domain_count,"window_count":len(subset),
            "history":history,"device":str(device),"torch":torch.__version__
        }}
    torch.save(ck,outp)
    result={"gate":"S3_HTML_MLM_PRETRAIN","status":"PASS","checkpoint":str(outp),
            "checkpoint_sha256":sha256(outp),"history":history,"window_count":len(subset),
            "pretrain_rows":ds.pretrain_row_count,"pretrain_families":ds.pretrain_family_count,
            "effective_batch_size":args.physical_batch_size*args.accum_steps,
            "elapsed_seconds":time.time()-t0}
    (outp.parent/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
