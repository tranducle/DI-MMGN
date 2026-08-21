"""
Train DI-MMGN for Phase 3.
Implements Phase 3.A (Contrastive pretraining of d_legit) and Phase 3.B (Anchoring d_deface).
"""
import os
import sys
import json
import copy
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN
from models.losses import composite_loss, ContrastiveLegitLoss, DefaceAnchorLoss

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """Full deterministic seeding for reproducible official runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Best-effort determinism; PyG pure-python scatter path remains functional.
    os.environ["PYTHONHASHSEED"] = str(seed)

def evaluate(model, loader, device, cfg):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            
            fwd_batch = {"t1": t1, "t2": t2, "mask": mask}
            out = model(fwd_batch)
            
            # During evaluation, prediction is based on the score (projection onto d_deface)
            preds = (torch.sigmoid(out["score"]) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Compute composite loss just for logging
            loss = composite_loss(out["legit_proj"], out["score"], labels, out.get("ortho_residual", torch.tensor(0.0, device=device)), cfg)
            total_loss += loss.item()
            
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), f1, accuracy_score(all_labels, all_preds)

def train_dimmgn(seed=42, config_path="config/default.yaml", ortho_override=None, ckpt_path="dimmgn_best.pt"):
    set_seed(seed)
    cfg = load_config(config_path)
    if ortho_override is not None:
        cfg["model"]["orthogonality"] = ortho_override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DI-MMGN] Using device: {device} | seed={seed} | ortho={cfg['model']['orthogonality']} | ckpt={ckpt_path}", flush=True)
    
    with open("data/splits.json") as f:
        splits = json.load(f)
        
    print("Loading datasets...")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full_dataset = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    
    # Phase 3.A Pretraining: Legitimate pairs from pretrain split
    pretrain_legit = [p for p in full_dataset.pairs if p["domain"] in splits["pretrain"] and p.get("label") == "legitimate"]
    # Phase 3.B Finetuning (Anchoring): All pairs from pretrain split
    train_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["pretrain"]]
    val_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["val"]]
    
    pretrain_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=pretrain_legit)
    train_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=train_pairs)
    val_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=val_pairs)
    
    collate_fn = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    pretrain_loader = DataLoader(pretrain_ds, batch_size=cfg["train"]["batch"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch"], shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Pretrain pairs (legit only): {len(pretrain_ds)}")
    print(f"Finetune pairs (all): {len(train_ds)}")
    
    model = DI_MMGN(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    accum = cfg["train"]["accum"]
    
    # Phase 3.A: Pretraining d_legit
    epochs_pretrain = cfg["train"]["epochs_pretrain"]
    print(f"\n--- Phase 3.A: Pretraining d_legit over {epochs_pretrain} epochs ---", flush=True)
    legit_loss_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    
    for ep in range(epochs_pretrain):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(pretrain_loader):
            labels = batch["label"].to(device) # Should be all 0s here
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            
            fwd_batch = {"t1": t1, "t2": t2, "mask": mask}
            out = model(fwd_batch)
            
            loss = legit_loss_fn(out["legit_proj"], labels)
            loss = loss / accum
            loss.backward()
            
            if (i + 1) % accum == 0 or (i + 1) == len(pretrain_loader):
                optimizer.step()
                model.after_step() # Renormalize orthogonal vectors
                optimizer.zero_grad()
                
            total_loss += loss.item() * accum
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(pretrain_loader)} - Loss: {loss.item()*accum:.4f}", flush=True)
            
        print(f"Pretrain Epoch {ep+1:02d}/{epochs_pretrain} | Loss: {total_loss/len(pretrain_loader):.4f}", flush=True)
        
    # Phase 3.B: Finetuning (Anchoring d_deface + Contrastive d_legit)
    epochs_finetune = cfg["train"]["epochs_finetune"]
    print(f"\n--- Phase 3.B: Finetuning (Anchoring) over {epochs_finetune} epochs ---", flush=True)
    best_val_f1 = -1.0  # -1 guarantees a checkpoint is always saved after epoch 1
    
    for ep in range(epochs_finetune):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            labels = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            
            fwd_batch = {"t1": t1, "t2": t2, "mask": mask}
            out = model(fwd_batch)
            
            residual = out.get("ortho_residual", torch.tensor(0.0, device=device))
            loss = composite_loss(out["legit_proj"], out["score"], labels, residual, cfg)
            
            loss = loss / accum
            loss.backward()
            
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                model.after_step() # Renormalize orthogonal vectors
                optimizer.zero_grad()
                
            total_loss += loss.item() * accum
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item()*accum:.4f}", flush=True)
            
        val_loss, val_f1, val_acc = evaluate(model, val_loader, device, cfg)
        print(f"Finetune Epoch {ep+1:02d}/{epochs_finetune} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}", flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
    print(f"[DI-MMGN] Training complete | best Val F1: {best_val_f1:.4f} | saved: {ckpt_path}", flush=True)

if __name__ == "__main__":
    train_dimmgn()
