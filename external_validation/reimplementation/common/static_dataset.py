import json, re, html as html_lib, pathlib, collections
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from bs4 import BeautifulSoup, Comment
from torchvision.transforms import functional as TF

KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

def visible_text_from_html(path: str) -> str:
    raw=pathlib.Path(path).read_bytes()
    soup=BeautifulSoup(raw,"html.parser")
    for tag in soup(["script","style","noscript","template"]):
        tag.decompose()
    for c in soup.find_all(string=lambda x:isinstance(x,Comment)):
        c.extract()
    parts=[]
    for node in soup.stripped_strings:
        s=str(node)
        if s:
            parts.append(s)
    return " ".join(parts)

class WordTokenizer:
    """Small Keras-Tokenizer-like word index used to avoid a TensorFlow dependency."""
    def __init__(self, lower=True, filters=KERAS_FILTERS, oov_token="<OOV>", min_freq=1, max_words=None):
        self.lower=lower
        self.filters=filters
        self.oov_token=oov_token
        self.min_freq=min_freq
        self.max_words=max_words
        self.word_index={oov_token:1}

    def _tokens(self,text:str):
        if self.lower:
            text=text.lower()
        trans=str.maketrans({c:" " for c in self.filters})
        text=text.translate(trans)
        return [t for t in text.split() if t]

    def fit(self,texts:Iterable[str]):
        cnt=collections.Counter()
        for text in texts:
            cnt.update(self._tokens(text))
        words=[w for w,n in cnt.most_common() if n>=self.min_freq and w!=self.oov_token]
        if self.max_words is not None:
            words=words[:max(0,self.max_words-2)]
        self.word_index={self.oov_token:1}
        for w in words:
            self.word_index[w]=len(self.word_index)+1
        return self

    def encode(self,text:str,max_len=128):
        ids=[self.word_index.get(t,1) for t in self._tokens(text)[:max_len]]
        if len(ids)<max_len:
            ids += [0]*(max_len-len(ids))
        return ids

    def save(self,path):
        pathlib.Path(path).write_text(json.dumps({
            "lower":self.lower,"filters":self.filters,"oov_token":self.oov_token,
            "min_freq":self.min_freq,"max_words":self.max_words,"word_index":self.word_index
        },indent=2)+"\n",encoding="utf-8")

    @classmethod
    def load(cls,path):
        x=json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        obj=cls(x["lower"],x["filters"],x["oov_token"],x["min_freq"],x["max_words"])
        obj.word_index={k:int(v) for k,v in x["word_index"].items()}
        return obj

def load_manifest(path,split=None):
    rows=[json.loads(x) for x in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if split is None or r["split"]==split]

class TextIdCache:
    def __init__(self,cache_dir):
        d=pathlib.Path(cache_dir)
        summary=json.loads((d/"cache_summary.json").read_text(encoding="utf-8"))
        if summary.get("status")!="PASS": raise RuntimeError(f"cache not PASS: {summary}")
        self.pair_ids=json.loads((d/"pair_ids.json").read_text(encoding="utf-8"))
        self.index={p:i for i,p in enumerate(self.pair_ids)}
        self.ids=np.load(d/"text_ids.npy",mmap_mode="r")
    def get(self,pair_id):
        return torch.from_numpy(np.asarray(self.ids[self.index[pair_id]],dtype=np.int64).copy())

class DOMWindowCache:
    def __init__(self,cache_dir):
        d=pathlib.Path(cache_dir)
        summary=json.loads((d/"cache_summary.json").read_text(encoding="utf-8"))
        if summary.get("status")!="PASS": raise RuntimeError(f"cache not PASS: {summary}")
        self.summary=summary
        self.pair_ids=json.loads((d/"pair_ids.json").read_text(encoding="utf-8"))
        self.index={p:i for i,p in enumerate(self.pair_ids)}
        self.input_ids=np.load(d/"dom_input_ids.npy",mmap_mode="r")
        self.attention=np.load(d/"dom_attention.npy",mmap_mode="r")
        self.tag_ids=np.load(d/"dom_tag_ids.npy",mmap_mode="r")
        self.depth_ids=np.load(d/"dom_depth_ids.npy",mmap_mode="r")
        self.node_ids=np.load(d/"dom_node_ids.npy",mmap_mode="r")
        self.window_count=np.load(d/"dom_window_count.npy",mmap_mode="r")

    def batch(self,pair_ids):
        ids=[]; att=[]; tags=[]; depths=[]; nodes=[]; owners=[]
        for owner,pid in enumerate(pair_ids):
            idx=self.index[pid]; n=int(self.window_count[idx])
            if n<1: raise RuntimeError(f"no DOM windows for {pid}")
            for wi in range(n):
                ids.append(np.asarray(self.input_ids[idx,wi],dtype=np.int64))
                att.append(np.asarray(self.attention[idx,wi],dtype=np.int64))
                tags.append(np.asarray(self.tag_ids[idx,wi],dtype=np.int64))
                depths.append(np.asarray(self.depth_ids[idx,wi],dtype=np.int64))
                nodes.append(np.asarray(self.node_ids[idx,wi],dtype=np.int64))
                owners.append(owner)
        return {
            "input_ids":torch.from_numpy(np.stack(ids).copy()),
            "attention_mask":torch.from_numpy(np.stack(att).copy()),
            "tag_ids":torch.from_numpy(np.stack(tags).copy()),
            "depth_ids":torch.from_numpy(np.stack(depths).copy()),
            "node_ids":torch.from_numpy(np.stack(nodes).copy()),
            "owner":torch.tensor(owners,dtype=torch.long),
            "batch_size":len(pair_ids),
        }

class StaticTextDataset(Dataset):
    def __init__(self,manifest_path,split,tokenizer:WordTokenizer,max_len=128,cache_text=False,text_id_cache=None):
        self.rows=load_manifest(manifest_path,split); self.tokenizer=tokenizer; self.max_len=max_len
        self._text_cache={} if cache_text else None
        self.text_id_cache=text_id_cache
    def __len__(self): return len(self.rows)
    def _text(self,row):
        key=row["html_path"]
        if self._text_cache is not None and key in self._text_cache: return self._text_cache[key]
        t=visible_text_from_html(key)
        if self._text_cache is not None: self._text_cache[key]=t
        return t
    def __getitem__(self,i):
        r=self.rows[i]
        ids=self.text_id_cache.get(r["pair_id"]) if self.text_id_cache is not None else torch.tensor(self.tokenizer.encode(self._text(r),self.max_len),dtype=torch.long)
        return {"text_ids":ids,
                "label":torch.tensor(r["label"],dtype=torch.long),"pair_id":r["pair_id"],
                "domain":r["domain"],"family_id":r["family_id"],
                "attack_type":r.get("attack_type") or "","source_type":r.get("source_type") or "",
                "html_path":r["html_path"]}

class StaticImageDataset(Dataset):
    def __init__(self,manifest_path,split,image_size=224):
        self.rows=load_manifest(manifest_path,split); self.image_size=image_size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r["screenshot_path"]) as im:
            im=im.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
            img=TF.pil_to_tensor(im).float()/255.0
        return {"image":img,"label":torch.tensor(r["label"],dtype=torch.long),"pair_id":r["pair_id"],
                "domain":r["domain"],"family_id":r["family_id"],
                "attack_type":r.get("attack_type") or "","source_type":r.get("source_type") or "",
                "screenshot_path":r["screenshot_path"]}

class StaticPageDataset(Dataset):
    def __init__(self, manifest_path, split, tokenizer:WordTokenizer, max_len=128, image_size=224,
                 cache_text=False,text_id_cache=None):
        self.rows=load_manifest(manifest_path,split)
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.image_size=image_size
        self.cache_text=cache_text
        self._text_cache={} if cache_text else None
        self.text_id_cache=text_id_cache

    def __len__(self):
        return len(self.rows)

    def _text(self,row):
        key=row["html_path"]
        if self._text_cache is not None and key in self._text_cache:
            return self._text_cache[key]
        t=visible_text_from_html(key)
        if self._text_cache is not None:
            self._text_cache[key]=t
        return t

    def __getitem__(self,i):
        r=self.rows[i]
        ids=self.text_id_cache.get(r["pair_id"]) if self.text_id_cache is not None else torch.tensor(self.tokenizer.encode(self._text(r),self.max_len),dtype=torch.long)
        with Image.open(r["screenshot_path"]) as im:
            im=im.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
            img=TF.pil_to_tensor(im).float()/255.0
        return {
            "text_ids":ids,
            "image":img,
            "label":torch.tensor(r["label"],dtype=torch.long),
            "pair_id":r["pair_id"],
            "domain":r["domain"],
            "family_id":r["family_id"],
            "attack_type":r.get("attack_type") or "",
            "source_type":r.get("source_type") or "",
            "html_path":r["html_path"],
            "screenshot_path":r["screenshot_path"],
        }
