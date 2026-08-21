"""
Phase-5c A2: Single-Snapshot Multi-Modal baseline.
Classifies E(S_t) directly (all 4 modalities, NO change-vector Δ_t).

Isolates: CONCAT(95.5%) − A2(?) = change-vector contribution.
Also: A2 − TextSnapshot(84.3%) = DOM/visual/HTTP contribution over text-only.

3 seeds, 30 epochs, dropout=0.15, crash-safe incremental.
"""
import os, sys, json, random, yaml, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.encoders import MultiModalEncoder

RESULTS = "../8_Project_Management/phase5c_snapshot_mm_results_raw.json"
CONFIG = "config/default_stochastic.yaml"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


class SnapshotMultiModal(nn.Module):
    """Classify E(S_t) directly — no Δ_t, no ortho decomposition."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"]["d"]
        p = float(cfg["model"].get("dropout", 0.0))
        self.encoder = MultiModalEncoder(cfg)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, batch):
        e2 = self.encoder(batch["t2"], batch["mask"])  # [B,d] — snapshot only
        return self.head(e2).squeeze(-1)  # [B]


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            y = b["label"].to(device)
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            logit = model({"t2": t2, "mask": mask})
            preds.extend((torch.sigmoid(logit) > 0.5).long().cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return f1_score(labels, preds, zero_division=0), accuracy_score(labels, preds)


def train_one(seed, cfg, loaders, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    epochs = cfg["train"]["epochs_finetune"]
    accum = cfg["train"]["accum"]
    model = SnapshotMultiModal(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    crit = nn.BCEWithLogitsLoss()
    ckpt = f"snapmm_{seed}_{uuid.uuid4().hex[:6]}.pt"
    print(f"\n--- [Phase5c] Snapshot-MM seed={seed} dropout={cfg['model']['dropout']} ---", flush=True)
    best = -1.0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); tl = 0.0
        for i, b in enumerate(train_loader):
            y = b["label"].float().to(device)
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            loss = crit(model({"t2": t2, "mask": mask}), y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); opt.zero_grad()
            tl += loss.item() * accum
            if (i + 1) % 100 == 0:
                print(f"  snapmm/{seed} batch {i+1}/{len(train_loader)} loss {loss.item()*accum:.4f}", flush=True)
        vf, _ = evaluate(model, val_loader, device)
        print(f"snapmm/{seed} ep {ep+1:02d}/{epochs} | train {tl/len(train_loader):.4f} | valF1 {vf:.4f}", flush=True)
        if vf > best:
            best = vf; torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    tf1, tacc = evaluate(model, test_loader, device)
    try: os.remove(ckpt)
    except OSError: pass
    print(f"[Phase5c] Snapshot-MM/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)
    return float(tf1)


def main():
    seeds = [42, 43, 44]
    raw = {"snapshot_mm": {}, "done": False}
    if os.path.exists(RESULTS):
        try: raw = json.load(open(RESULTS))
        except Exception: pass
    _save(raw)

    cfg = yaml.safe_load(open(CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase5c] device={device} | dropout={cfg['model']['dropout']} | snapshot multi-modal (no Δ_t)", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def pairs_of(split):
        return [p for p in full.pairs if p["domain"] in splits[split]]
    collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def loader(split, shuffle):
        ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                               max_text_len=cfg["data"]["max_text_len"], pairs=pairs_of(split))
        return DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=shuffle, collate_fn=collate, num_workers=0)
    loaders = (loader("pretrain", True), loader("val", False), loader("test", False))

    for seed in seeds:
        if str(seed) in raw.get("snapshot_mm", {}):
            print(f"[Phase5c] skip seed={seed} (done)", flush=True)
            continue
        f1 = train_one(seed, cfg, loaders, device)
        raw.setdefault("snapshot_mm", {})[str(seed)] = f1
        _save(raw)

    from scipy import stats
    vals = [raw["snapshot_mm"][str(s)] for s in seeds]
    m = float(np.mean(vals))
    ci = float(stats.sem(vals) * stats.t.ppf(0.975, len(vals)-1)) if len(vals) > 1 else 0.0
    raw["summary"] = {"mean": m, "ci95": ci}
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase5c] DONE  Snapshot-MM = {m*100:.2f} ± {ci*100:.2f}  -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
