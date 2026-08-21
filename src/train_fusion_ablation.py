"""
Phase-5e: Fusion-Strategy Ablation (Gap 3 fix — architecture depth).
Trains CrossAttn and Gated fusion variants alongside the existing Concat.
Systematic comparison: concat vs cross-attention vs gating.

Usage: python train_fusion_ablation.py [--modes crossattn gated]
Default: trains both crossattn and gated, 3 seeds each.
Crash-safe incremental → phase5e_fusion_results_raw.json.
"""
import os, sys, json, random, uuid, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import FUSION_REGISTRY
from models.losses import composite_loss, ContrastiveLegitLoss

STOCH_CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/phase5e_fusion_results_raw.json"
PREDS_DIR = "../8_Project_Management/phase5e_predictions"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


def evaluate(model, loader, device):
    model.eval()
    preds, labels, atks = [], [], []
    with torch.no_grad():
        for b in loader:
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            p = (torch.sigmoid(out["score"]) > 0.5).long()
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
            atks.extend(b.get("attack_type", ["unknown"] * len(y)))
    return f1_score(labels, preds, zero_division=0), accuracy_score(labels, preds), preds, labels, atks


def train_variant(mode, seed, cfg, loaders, device):
    set_seed(seed)
    ModelClass = FUSION_REGISTRY[mode]
    train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]
    model = ModelClass(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    crit = nn.BCEWithLogitsLoss()
    cl_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    ckpt = f"fusion_{mode}_{seed}_{uuid.uuid4().hex[:6]}.pt"

    print(f"\n--- [Phase5e] {mode} seed={seed} ---", flush=True)

    # Phase A
    for ep in range(cfg["train"]["epochs_pretrain"]):
        model.train(); opt.zero_grad()
        for i, b in enumerate(train_loader):
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            loss = cl_fn(out["legit_proj"], y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); model.after_step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            print(f"  pretrain ep {ep+1}/{cfg['train']['epochs_pretrain']}", flush=True)

    # Phase B
    best = -1.0
    for ep in range(cfg["train"]["epochs_finetune"]):
        model.train(); opt.zero_grad()
        for i, b in enumerate(train_loader):
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            residual = out.get("ortho_residual", torch.tensor(0.0, device=device))
            loss = composite_loss(out["legit_proj"], out["score"], y, residual, cfg) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); model.after_step(); opt.zero_grad()
            if (i + 1) % 100 == 0:
                print(f"  batch {i+1}/{len(train_loader)} loss {loss.item()*accum:.4f}", flush=True)
        vf, _, _, _, _ = evaluate(model, val_loader, device)
        print(f"  ep {ep+1}/{cfg['train']['epochs_finetune']} valF1={vf:.4f}", flush=True)
        if vf > best:
            best = vf; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    tf1, tacc, preds, labels, atks = evaluate(model, test_loader, device)
    try: os.remove(ckpt)
    except OSError: pass
    print(f"[Phase5e] {mode}/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)

    # Save predictions
    os.makedirs(PREDS_DIR, exist_ok=True)
    with open(os.path.join(PREDS_DIR, f"{mode}_{seed}.json"), "w") as f:
        json.dump({"f1": tf1, "preds": preds, "labels": labels, "attack_types": atks}, f)

    return float(tf1)


def main():
    modes = ["crossattn", "gated"]
    seeds = [42, 43, 44]
    raw = {m: {} for m in modes}
    raw["done"] = False
    if os.path.exists(RESULTS):
        try: raw = json.load(open(RESULTS))
        except Exception: pass
    _save(raw)

    cfg = yaml.safe_load(open(STOCH_CONFIG))
    cfg["model"]["orthogonality"] = "none"  # best variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase5e] device={device} | modes={modes} | seeds={seeds}", flush=True)

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
                print(f"[Phase5e] skip {mode}/{seed} (done)", flush=True)
                continue
            f1 = train_variant(mode, seed, cfg, loaders, device)
            raw.setdefault(mode, {})[str(seed)] = f1
            _save(raw)

    # Summary
    def mci(vals):
        a = np.array(vals, float)
        m, se = a.mean(), stats.sem(a)
        return float(m), float(se * stats.t.ppf(0.975, len(a)-1)) if len(a) > 1 else 0.0
    raw["summary"] = {}
    for m in modes:
        vals = [raw[m][str(s)] for s in seeds if str(s) in raw.get(m, {})]
        if vals:
            mean, ci = mci(vals)
            raw["summary"][m] = {"mean": mean, "ci95": ci}
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase5e] DONE -> {RESULTS}", flush=True)
    for m in modes:
        if m in raw.get("summary", {}):
            s = raw["summary"][m]
            print(f"  {m} Test F1 = {s['mean']*100:.2f} ± {s['ci95']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
