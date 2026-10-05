"""LWDED-v4 strict full-modality pair loader."""
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

_GRAPH_CACHE={}
_VEC_CACHE={}

class DimmgnPairDatasetV4(Dataset):
    def __init__(self, unified_pairs_path, data_root=".", split=None, pairs=None):
        self.root=data_root
        if pairs is None:
            rows=[json.loads(x) for x in open(unified_pairs_path,encoding="utf-8") if x.strip()]
        else:
            rows=list(pairs)
        if split is not None: rows=[r for r in rows if r.get("split")==split]
        self.pairs=rows
    def __len__(self): return len(self.pairs)
    def _full(self,rel):
        if not rel: raise ValueError("LWDED-v4 pair contains missing modality path")
        p=os.path.join(self.root,rel.replace("/",os.sep))
        if not os.path.exists(p): raise FileNotFoundError(p)
        return p
    def _load_graph(self,rel):
        p=self._full(rel); key=os.path.normpath(p)
        if key not in _GRAPH_CACHE:
            g=torch.load(p,map_location="cpu",weights_only=False)
            x=g["x"].float(); ei=g["edge_index"].long()
            if x.ndim!=2 or x.shape[1]!=70: raise ValueError(f"invalid graph x {tuple(x.shape)}: {p}")
            if ei.ndim!=2 or ei.shape[0]!=2: raise ValueError(f"invalid edge_index {tuple(ei.shape)}: {p}")
            d=Data(x=x,edge_index=ei); d.num_nodes=int(g["num_nodes"]); _GRAPH_CACHE[key]=d
        return _GRAPH_CACHE[key]
    def _load_vec(self,rel,dim):
        p=self._full(rel); key=(os.path.normpath(p),dim)
        if key not in _VEC_CACHE:
            a=np.load(p)
            if a.shape!=(dim,): raise ValueError(f"expected {(dim,)}, got {a.shape}: {p}")
            if not np.isfinite(a).all(): raise ValueError(f"nonfinite vector: {p}")
            _VEC_CACHE[key]=torch.from_numpy(a.astype(np.float32,copy=False))
        return _VEC_CACHE[key]
    def __getitem__(self,i):
        r=self.pairs[i]
        te=r["text_embedding"]; dg=r["delta_dom_graph"]; dv=r["delta_visual"]; dh=r["delta_http"]
        return {
            "text1":self._load_vec(te["t1"],384),"text2":self._load_vec(te["t2"],384),
            "g1":self._load_graph(dg["t1"]),"g2":self._load_graph(dg["t2"]),
            "v1":self._load_vec(dv["t1"],512),"v2":self._load_vec(dv["t2"],512),
            "h1":self._load_vec(dh["t1"],32),"h2":self._load_vec(dh["t2"],32),
            "label":0 if r["label"]=="legitimate" else 1,
            "domain":r["domain"],"attack_type":r.get("attack_type"),"pair_id":r["pair_id"],"source_type":r["source_type"],
        }

class DimmgnCollateV4:
    def _batch_graphs(self,graphs):
        b=Batch.from_data_list(graphs); return b.x,b.edge_index,b.batch
    def __call__(self,samples):
        t1_gx,t1_ei,t1_b=self._batch_graphs([s["g1"] for s in samples])
        t2_gx,t2_ei,t2_b=self._batch_graphs([s["g2"] for s in samples])
        t1={"text_embedding":torch.stack([s["text1"] for s in samples]),"dom_x":t1_gx,"dom_edge_index":t1_ei,"dom_batch":t1_b,"visual":torch.stack([s["v1"] for s in samples]),"http":torch.stack([s["h1"] for s in samples])}
        t2={"text_embedding":torch.stack([s["text2"] for s in samples]),"dom_x":t2_gx,"dom_edge_index":t2_ei,"dom_batch":t2_b,"visual":torch.stack([s["v2"] for s in samples]),"http":torch.stack([s["h2"] for s in samples])}
        return {"t1":t1,"t2":t2,"mask":torch.ones((len(samples),4),dtype=torch.bool),"label":torch.tensor([s["label"] for s in samples],dtype=torch.long),"domain":[s["domain"] for s in samples],"attack_type":[s["attack_type"] for s in samples],"pair_id":[s["pair_id"] for s in samples],"source_type":[s["source_type"] for s in samples]}
