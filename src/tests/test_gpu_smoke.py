"""GPU smoke (Phase 2 §7 step 2.9): confirm DI-MMGN fits RTX 3090 24 GB at the
planned batch=16 / accum=2. Loads a real batch on CUDA, forward+backward,
reports peak VRAM. Labelled SMOKE per Rules.md §5."""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from src.data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from src.models.dimmgn import DI_MMGN
from src.models.losses import composite_loss

UNIFIED = os.path.join(ROOT, "dataset_pipeline", "v2", "unified_pairs.jsonl")
DATA_ROOT = os.path.join(ROOT, "dataset_pipeline")

def pick_full(n):
    out=[]
    for l in open(UNIFIED, encoding="utf-8"):
        r=json.loads(l)
        d=r.get("delta_dom_graph") or {}; v=r.get("delta_visual") or {}; h=r.get("delta_http") or {}
        if d.get("t1") and d.get("t2") and v.get("t1") and v.get("t2") and h.get("t1") and h.get("t2"):
            out.append(r)
            if len(out)>=n: break
    return out

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    dev="cuda"
    print(f"[gpu-smoke] device={dev} ({torch.cuda.get_device_name(0)})")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    pairs=pick_full(16)
    ds=DimmgnPairDataset(pairs=pairs, data_root=DATA_ROOT, tokenizer=tok)
    collate=DimmgnCollate(tok)
    from torch.utils.data import DataLoader
    dl=DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    batch=next(iter(dl))
    cfg={"model":{"d":128,"text_backbone":"sentence-transformers/all-MiniLM-L6-v2",
                  "text_frozen":True,"dom_layers":2,"dom_hidden":64,"visual_in":512,"http_in":32,
                  "orthogonality":"soft"},
         "loss":{"w_legit":1.0,"w_deface":1.0,"w_ortho":1.0,"temperature":0.1}}
    model=DI_MMGN(cfg).to(dev)
    # move batch tensors to cuda
    def mv(snap):
        for k in ("dom_x","dom_edge_index","dom_batch","visual","http","text_input_ids","text_attention_mask"):
            snap[k]=snap[k].to(dev)
        return snap
    batch["t1"]=mv(batch["t1"]); batch["t2"]=mv(batch["t2"])
    batch["mask"]=batch["mask"].to(dev); batch["label"]=batch["label"].to(dev)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    out=model(batch)
    L=composite_loss(out["legit_proj"],out["score"],batch["label"],model.ortho.orthogonality_residual(),cfg)
    L.backward()
    model.after_step()
    peak=torch.cuda.max_memory_allocated()/1e9
    print(f"[gpu-smoke] batch=16 forward+backward OK; loss={float(L):.4f}; peak VRAM={peak:.2f} GB")
    fits = peak < 24.0
    print(f"[gpu-smoke] 24GB RTX 3090 fit at batch=16: {'YES' if fits else 'NO (need accum/grad-ckpt)'}")
    print("=== GPU-SMOKE: PASS ===" if fits else "=== GPU-SMOKE: WARN — needs gradient accumulation/ckpt ===")

if __name__=="__main__":
    main()
