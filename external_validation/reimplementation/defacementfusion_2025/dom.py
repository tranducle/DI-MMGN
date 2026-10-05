import pathlib, zlib
from dataclasses import dataclass
from typing import List, Dict
import torch
from bs4 import BeautifulSoup, Comment

SKIP={"script","style","noscript","template"}

def _depth(tag):
    d=0; p=tag.parent
    while p is not None and getattr(p,"name",None) is not None:
        d+=1; p=p.parent
    return d

def _direct_text(tag):
    parts=[]
    for c in tag.children:
        if isinstance(c,str):
            s=" ".join(c.split())
            if s: parts.append(s)
    return " ".join(parts)

class DOMWindowBuilder:
    """
    Paper-faithful DOM preorder tokenizer.
    Assumptions not specified in DefacementFusion are documented in SPEC.md.
    """
    def __init__(self,tokenizer,max_length=512,overlap_tokens=256,max_windows=4,
                 tag_buckets=128,depth_buckets=32,node_buckets=512):
        self.tok=tokenizer
        self.max_length=int(max_length)
        self.body_cap=self.max_length-2
        self.overlap_tokens=int(overlap_tokens)
        self.max_windows=max_windows
        self.tag_buckets=tag_buckets
        self.depth_buckets=depth_buckets
        self.node_buckets=node_buckets

    def _tag_id(self,name):
        if not name: return 0
        return 1+(zlib.crc32(name.encode("utf-8"))%(self.tag_buckets-1))

    def nodes(self,path):
        raw=pathlib.Path(path).read_bytes()
        soup=BeautifulSoup(raw,"html.parser")
        for t in soup.find_all(list(SKIP)):
            t.decompose()
        for c in soup.find_all(string=lambda x:isinstance(x,Comment)):
            c.extract()
        out=[]
        idx=0
        for tag in soup.find_all(True):
            idx+=1
            name=(tag.name or "unk").lower()
            attrs=[]
            if tag.get("id"):
                attrs.append("id="+str(tag.get("id")))
            cls=tag.get("class")
            if cls:
                attrs.append("class="+" ".join(map(str,cls)))
            txt=_direct_text(tag)
            # Preserve tag markup + descriptive class/id + direct node text.
            content=" ".join([f"<{name}>"]+attrs+([txt] if txt else []))
            ids=self.tok.encode(content,add_special_tokens=False,truncation=True,max_length=self.body_cap)
            if not ids:
                ids=self.tok.encode(f"<{name}>",add_special_tokens=False)
            out.append({
                "token_ids":ids,
                "tag_id":self._tag_id(name),
                "depth":min(_depth(tag),self.depth_buckets-1),
                "node_id":min(idx,self.node_buckets-1),
            })
        return out

    def windows(self,path):
        nodes=self.nodes(path)
        windows=[]
        start=0
        while start<len(nodes) and (self.max_windows is None or len(windows)<self.max_windows):
            body=[]; tags=[]; depths=[]; nodeids=[]
            end=start
            while end<len(nodes):
                n=nodes[end]
                remain=self.body_cap-len(body)
                if remain<=0: break
                take=n["token_ids"][:remain]
                body.extend(take)
                tags.extend([n["tag_id"]]*len(take))
                depths.extend([n["depth"]]*len(take))
                nodeids.extend([n["node_id"]]*len(take))
                end+=1
                if len(body)>=self.body_cap: break
            if not body:
                break
            ids=[self.tok.cls_token_id]+body+[self.tok.sep_token_id]
            out={
                "input_ids":ids,
                "tag_ids":[0]+tags+[0],
                "depth_ids":[0]+depths+[0],
                "node_ids":[0]+nodeids+[0],
                "attention_mask":[1]*len(ids),
            }
            windows.append(out)
            if end>=len(nodes): break
            # Node-aware overlap: walk backward over whole nodes until ~overlap_tokens retained.
            kept=0; new_start=end
            while new_start>start:
                n=nodes[new_start-1]
                if kept+len(n["token_ids"])>self.overlap_tokens and kept>0:
                    break
                kept+=len(n["token_ids"])
                new_start-=1
            start=max(start+1,new_start)
        if not windows:
            ids=[self.tok.cls_token_id,self.tok.sep_token_id]
            windows=[{"input_ids":ids,"tag_ids":[0,0],"depth_ids":[0,0],"node_ids":[0,0],"attention_mask":[1,1]}]
        return windows

    def collate(self,paths):
        allw=[]; owners=[]
        for owner,p in enumerate(paths):
            for w in self.windows(p):
                allw.append(w); owners.append(owner)
        L=max(len(w["input_ids"]) for w in allw)
        def pad(name,padval=0):
            return torch.tensor([w[name]+[padval]*(L-len(w[name])) for w in allw],dtype=torch.long)
        return {
            "input_ids":pad("input_ids",self.tok.pad_token_id or 0),
            "attention_mask":pad("attention_mask",0),
            "tag_ids":pad("tag_ids",0),
            "depth_ids":pad("depth_ids",0),
            "node_ids":pad("node_ids",0),
            "owner":torch.tensor(owners,dtype=torch.long),
            "batch_size":len(paths),
        }

def node_aware_mlm_mask(batch,tokenizer,probability=0.15,rng=None):
    """Apply the paper's node-aware 15% MLM selection and BERT 80/10/10 replacement rule."""
    import random
    rng=rng or random.Random()
    ids=batch["input_ids"].clone()
    labels=torch.full_like(ids,-100)
    mask_id=tokenizer.mask_token_id
    vocab_size=int(tokenizer.vocab_size)
    for i in range(ids.size(0)):
        att=batch["attention_mask"][i].tolist()
        nodes=batch["node_ids"][i].tolist()
        groups={}
        for j,(a,nid) in enumerate(zip(att,nodes)):
            if not a or nid<=0: continue
            groups.setdefault(int(nid),[]).append(j)
        all_pos=[j for ps in groups.values() for j in ps]
        if not all_pos: continue
        target=max(1,int(round(len(all_pos)*probability)))
        keys=list(groups); rng.shuffle(keys)
        selected=[]
        for nid in keys:
            selected.extend(groups[nid])
            if len(selected)>=target: break
        for j in selected:
            original=int(ids[i,j])
            labels[i,j]=original
            r=rng.random()
            if r<0.8:
                ids[i,j]=mask_id
            elif r<0.9:
                ids[i,j]=rng.randrange(vocab_size)
            else:
                pass
    masked=dict(batch)
    masked["input_ids"]=ids
    return masked,labels
