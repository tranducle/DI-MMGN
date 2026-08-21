"""
Train Baseline GNN (GraphSAGE) for Phase 3.
Reads splits.json, trains BaselineGraphSAGE on the pretrain split, evaluates on val/test.
"""
import os
import sys
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer

from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.baseline_gnn import BaselineGraphSAGE

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """Full deterministic seeding for reproducible official runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].float().to(device)
            # Send t2 to device
            t2 = batch["t2"]
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t2.items()}
            
            logits = model(t2)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), f1, accuracy_score(all_labels, all_preds)

def train_baseline(seed=42, config_path="config/default.yaml", ckpt_path="baseline_best.pt"):
    set_seed(seed)
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] Using device: {device} | seed={seed} | ckpt={ckpt_path}", flush=True)
    
    # Load splits
    with open("data/splits.json") as f:
        splits = json.load(f)
        
    print("Loading datasets...")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full_dataset = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    
    # Filter by splits
    train_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["pretrain"]]
    val_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["val"]]
    test_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["test"]]
    
    train_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=train_pairs)
    val_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=val_pairs)
    test_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=test_pairs)
    
    collate_fn = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch"], shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch"], shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}, Test pairs: {len(test_ds)}")
    
    model = BaselineGraphSAGE(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = cfg["train"]["epochs_finetune"] # use finetune epochs for baseline
    accum = cfg["train"]["accum"]
    
    print(f"Training Baseline GNN over {epochs} epochs...", flush=True)
    best_val_f1 = -1.0  # -1 guarantees a checkpoint is always saved after epoch 1
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            labels = batch["label"].float().to(device)
            t2 = batch["t2"]
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t2.items()}
            
            logits = model(t2)
            loss = criterion(logits, labels)
            loss = loss / accum
            loss.backward()
            
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accum
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item()*accum:.4f}", flush=True)
            
        val_loss, val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}", flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            
    print("Testing Baseline GNN...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_loss, test_f1, test_acc = evaluate(model, test_loader, device)
    print(f"Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save sanity report
    with open("../8_Project_Management/baseline_sanity.md", "w") as f:
        f.write("# Baseline Sanity Report\n\n")
        f.write("**Gate**: BASELINE-SANITY\n")
        f.write("**Model**: BaselineGraphSAGE\n")
        f.write(f"**Test F1**: {test_f1:.4f}\n")
        f.write(f"**Test Accuracy**: {test_acc:.4f}\n\n")
        f.write("**Status**: " + ("✅ PASS" if test_f1 > 0.5 else "❌ FAIL") + "\n")
        
    return test_f1

if __name__ == "__main__":
    train_baseline()
