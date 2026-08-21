"""
Phase-5 A1: Non-GNN text baselines for TIFS submission.
  A1a: Text-Snapshot classifier (frozen MiniLM → BiLSTM → max-pool → FC)
       Classifies single snapshot S_t text — standard text-based defacement detector.
  A1b: Text-Change-Vector classifier (frozen MiniLM → proj → Δ_text → FC)
       Classifies the temporal text difference — isolates change-vector from multi-modal.

Reuses DimmgnPairDataset + DimmgnCollate (same data pipeline as all other experiments).
Crash-safe incremental results -> phase5_text_baselines_raw.json.
3 seeds {42,43,44}, 30 epochs each.
"""
import os, sys, json, random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate

RESULTS = "../8_Project_Management/phase5_text_baselines_raw.json"
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


# ── A1a: Text-Snapshot (frozen MiniLM → BiLSTM → max-pool → FC) ─────────
class TextSnapshotClassifier(nn.Module):
    """Standard text-based detector: classify a single page's text."""
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BACKBONE)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bilstm = nn.LSTM(384, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, snap):
        with torch.no_grad():
            out = self.bert(input_ids=snap["text_input_ids"],
                            attention_mask=snap["text_attention_mask"])
            seq = out.last_hidden_state  # [B, L, 384]
        lstm_out, _ = self.bilstm(seq)   # [B, L, 256]
        pooled = lstm_out.max(dim=1)[0]  # [B, 256]
        return self.fc(pooled).squeeze(-1)  # [B]


# ── A1b: Text-Change-Vector (frozen MiniLM → proj → Δ_text → FC) ────────
class TextChangeVectorClassifier(nn.Module):
    """Text-only change vector: Δ_text = E_text(S_t) − E_text(S_{t-1})."""
    def __init__(self, d=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BACKBONE)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(384, d)
        self.fc = nn.Linear(d, 1)

    def forward(self, batch):
        with torch.no_grad():
            o1 = self.bert(input_ids=batch["t1"]["text_input_ids"],
                           attention_mask=batch["t1"]["text_attention_mask"]).last_hidden_state[:, 0]
            o2 = self.bert(input_ids=batch["t2"]["text_input_ids"],
                           attention_mask=batch["t2"]["text_attention_mask"]).last_hidden_state[:, 0]
        e1 = self.proj(o1)   # [B, d]
        e2 = self.proj(o2)   # [B, d]
        delta = e2 - e1      # [B, d]
        return self.fc(delta).squeeze(-1)  # [B]


def evaluate(model, loader, device, mode):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            y = b["label"].to(device)
            if mode == "snapshot":
                t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
                logit = model(t2)
            else:
                t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
                t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
                logit = model({"t1": t1, "t2": t2})
            preds.extend((torch.sigmoid(logit) > 0.5).long().cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return f1_score(labels, preds, zero_division=0), accuracy_score(labels, preds)


def train_one(mode, seed, cfg, loaders, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    epochs = cfg["train"]["epochs_finetune"]
    accum = cfg["train"]["accum"]
    model = (TextSnapshotClassifier() if mode == "snapshot" else TextChangeVectorClassifier()).to(device)
    opt = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"]
    )
    crit = nn.BCEWithLogitsLoss()
    ckpt = f"text_{mode}_{seed}.pt"
    print(f"\n--- [Phase5] Text-{mode} seed={seed} over {epochs} epochs ---", flush=True)
    best = -1.0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); tl = 0.0
        for i, b in enumerate(train_loader):
            y = b["label"].float().to(device)
            if mode == "snapshot":
                t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
                loss = crit(model(t2), y) / accum
            else:
                t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
                t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
                loss = crit(model({"t1": t1, "t2": t2}), y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); opt.zero_grad()
            tl += loss.item() * accum
            if (i + 1) % 100 == 0:
                print(f"  text-{mode}/{seed} batch {i+1}/{len(train_loader)} loss {loss.item()*accum:.4f}", flush=True)
        vf, _ = evaluate(model, val_loader, device, mode)
        print(f"text-{mode}/{seed} epoch {ep+1:02d}/{epochs} | train {tl/len(train_loader):.4f} | valF1 {vf:.4f}", flush=True)
        if vf > best:
            best = vf; torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    tf1, tacc = evaluate(model, test_loader, device, mode)
    try: os.remove(ckpt)
    except OSError: pass
    print(f"[Phase5] Text-{mode}/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)
    return float(tf1)


def main():
    modes = ["snapshot", "changevec"]
    seeds = [42, 43, 44]
    raw = {"snapshot": {}, "changevec": {}, "done": False}
    if os.path.exists(RESULTS):
        try: raw = json.load(open(RESULTS))
        except Exception: pass
    _save(raw)

    cfg = yaml.safe_load(open("config/default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase5] device={device} | modes={modes} | seeds={seeds} | epochs={cfg['train']['epochs_finetune']}", flush=True)

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

    for mode in modes:
        for seed in seeds:
            if str(seed) in raw.get(mode, {}):
                print(f"[Phase5] skip text-{mode}/{seed} (done)", flush=True)
                continue
            f1 = train_one(mode, seed, cfg, loaders, device)
            raw.setdefault(mode, {})[str(seed)] = f1
            _save(raw)

    # summary
    from scipy import stats
    def mci(vals):
        a = np.array(vals, float)
        m, se = a.mean(), stats.sem(a)
        return float(m), float(se * stats.t.ppf(0.975, len(a)-1)) if len(a) > 1 else 0.0
    raw["summary"] = {}
    for mode in modes:
        vals = [raw[mode][str(s)] for s in seeds if str(s) in raw.get(mode, {})]
        if vals:
            m, c = mci(vals)
            raw["summary"][mode] = {"mean": m, "ci95": c}
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase5] DONE -> {RESULTS}", flush=True)
    for mode in modes:
        if mode in raw.get("summary", {}):
            s = raw["summary"][mode]
            print(f"  Text-{mode} Test F1 = {s['mean']*100:.2f} ± {s['ci95']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
