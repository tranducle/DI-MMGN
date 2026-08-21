"""
Phase-4c: Extra strong GNN baselines (GAT, GIN) for reviewer breadth (Hard Rule #4).
Same data/splits/optimizer/budget as train_baseline.py (GraphSAGE), only the encoder
changes. 3 seeds each. Crash-safe incremental -> phase4c_baselines_raw.json.
"""
import os, sys, json, random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.baseline_gnn import BASELINE_REGISTRY

RESULTS = "../8_Project_Management/phase4c_baselines_raw.json"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    with torch.no_grad():
        for b in loader:
            y = b["label"].float().to(device)
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            logit = model(t2)
            total += crit(logit, y).item()
            preds.extend((torch.sigmoid(logit) > 0.5).long().cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return total / len(loader), f1_score(labels, preds, zero_division=0), accuracy_score(labels, preds)


def train_one(name, cls, seed, cfg, loaders, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]
    epochs = cfg["train"]["epochs_finetune"]
    model = cls(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    crit = nn.BCEWithLogitsLoss()
    ckpt = f"baseline_{name}_{seed}.pt"
    print(f"\n--- [Phase4c] {name.upper()} seed={seed} over {epochs} epochs ---", flush=True)
    best = -1.0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); tl = 0.0
        for i, b in enumerate(train_loader):
            y = b["label"].float().to(device)
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            loss = crit(model(t2), y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); opt.zero_grad()
            tl += loss.item() * accum
            if (i + 1) % 50 == 0:
                print(f"  {name}/{seed} Batch {i+1}/{len(train_loader)} - Loss {loss.item()*accum:.4f}", flush=True)
        vl, vf, _ = evaluate(model, val_loader, device)
        print(f"{name}/{seed} Epoch {ep+1:02d}/{epochs} | Train {tl/len(train_loader):.4f} | ValLoss {vl:.4f} | ValF1 {vf:.4f}", flush=True)
        if vf > best:
            best = vf; torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    _, tf1, tacc = evaluate(model, test_loader, device)
    try:
        os.remove(ckpt)
    except OSError:
        pass
    print(f"[Phase4c] {name}/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)
    return float(tf1)


def main():
    seeds = [42, 43, 44]
    names = ["gat", "gin"]
    raw = {n: {} for n in names}
    raw["done"] = False
    if os.path.exists(RESULTS):
        try:
            raw = json.load(open(RESULTS))
            for n in names:
                raw.setdefault(n, {})
        except Exception:
            pass
    _save(raw)

    cfg = yaml.safe_load(open("config/default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase4c] device={device} | baselines={names} | seeds={seeds} | epochs={cfg['train']['epochs_finetune']}", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def pairs_of(split):
        return [p for p in full.pairs if p["domain"] in splits[split]]
    collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def loader(split, shuffle):
        ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok, max_text_len=cfg["data"]["max_text_len"], pairs=pairs_of(split))
        return DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=shuffle, collate_fn=collate, num_workers=0)
    loaders = (loader("pretrain", True), loader("val", False), loader("test", False))

    for name in names:
        cls = BASELINE_REGISTRY[name]
        for seed in seeds:
            if str(seed) in raw.get(name, {}):
                print(f"[Phase4c] skip {name}/{seed} (done)", flush=True)
                continue
            f1 = train_one(name, cls, seed, cfg, loaders, device)
            raw.setdefault(name, {})[str(seed)] = f1
            _save(raw)

    def mci(vals):
        a = np.array(vals, float)
        from scipy import stats
        m, se = a.mean(), stats.sem(a)
        return float(m), float(se * stats.t.ppf(0.975, len(a) - 1)) if len(a) > 1 else 0.0
    raw["summary"] = {n: {"mean": mci([raw[n][str(s)] for s in seeds])[0],
                          "ci95": mci([raw[n][str(s)] for s in seeds])[1]} for n in names}
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase4c] DONE -> {RESULTS}", flush=True)
    for n in names:
        s = raw["summary"][n]
        print(f"  {n.upper()} Test F1 = {s['mean']*100:.2f} ± {s['ci95']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
