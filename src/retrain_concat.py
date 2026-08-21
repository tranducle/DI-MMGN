"""
Phase-5b: Retrain DI-MMGN with CONCAT architecture [E(S_t); Δ_t].
Root-cause fix for catastrophic FP on legitimate CMS mutations.

Uses dropout=0.15 (honest variance), 3 seeds, soft+none ortho.
Saves per-sample predictions WITH attack_type for immediate per-attack breakdown.
Crash-safe incremental → phase5b_concat_results_raw.json.
"""
import os, sys, json, random, copy, uuid, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN_Concat
from models.losses import composite_loss, ContrastiveLegitLoss

STOCH_CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/phase5b_concat_results_raw.json"
PREDS_DIR = "../8_Project_Management/phase5b_predictions"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


def evaluate(model, loader, device, cfg):
    model.eval()
    preds, labels, attack_types = [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            p = (torch.sigmoid(out["score"]) > 0.5).long()
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
            attack_types.extend(batch.get("attack_type", ["unknown"] * len(y)))
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    return float(f1), float(acc), preds, labels, attack_types


def train_concat(seed, config_path, ortho_mode, loaders, device):
    set_seed(seed)
    cfg = yaml.safe_load(open(config_path))
    cfg["model"]["orthogonality"] = ortho_mode
    d = cfg["model"]["d"]
    train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]

    model = DI_MMGN_Concat(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    ckpt = f"concat_{ortho_mode}_{seed}_{uuid.uuid4().hex[:6]}.pt"

    print(f"\n--- [Phase5b] CONCAT [E;Δ] dropout={cfg['model']['dropout']} ortho={ortho_mode} seed={seed} ---", flush=True)

    # Phase A: contrastive pretrain on legitimate pairs only
    print(f"--- Phase A: Pretraining d_legit ({cfg['train']['epochs_pretrain']} epochs) ---", flush=True)
    cl_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    for ep in range(cfg["train"]["epochs_pretrain"]):
        model.train(); opt.zero_grad(); tl = 0.0
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
            tl += loss.item() * accum
        if (ep + 1) % 5 == 0:
            print(f"  pretrain ep {ep+1}/{cfg['train']['epochs_pretrain']} loss {tl/len(train_loader):.4f}", flush=True)

    # Phase B: finetune with composite loss
    print(f"--- Phase B: Finetuning ({cfg['train']['epochs_finetune']} epochs) ---", flush=True)
    best_f1 = -1.0
    for ep in range(cfg["train"]["epochs_finetune"]):
        model.train(); opt.zero_grad(); tl = 0.0
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
            tl += loss.item() * accum
            if (i + 1) % 100 == 0:
                print(f"  batch {i+1}/{len(train_loader)} loss {loss.item()*accum:.4f}", flush=True)
        vf, _, _, _, _ = evaluate(model, val_loader, device, cfg)
        print(f"  finetune ep {ep+1}/{cfg['train']['epochs_finetune']} | train {tl/len(train_loader):.4f} | valF1 {vf:.4f}", flush=True)
        if vf > best_f1:
            best_f1 = vf; torch.save(model.state_dict(), ckpt)

    # Test eval
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    tf1, tacc, preds, labels, atks = evaluate(model, test_loader, device, cfg)
    try: os.remove(ckpt)
    except OSError: pass
    print(f"[Phase5b] CONCAT {ortho_mode}/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)

    # Save per-attack predictions
    os.makedirs(PREDS_DIR, exist_ok=True)
    with open(os.path.join(PREDS_DIR, f"{ortho_mode}_{seed}.json"), "w") as f:
        json.dump({"f1": tf1, "preds": preds, "labels": labels, "attack_types": atks}, f)

    return tf1


def main():
    seeds = [42, 43, 44]
    raw = {"soft": {}, "none": {}, "config": "concat [E;Δ], dropout=0.15", "done": False}
    if os.path.exists(RESULTS):
        try: raw = json.load(open(RESULTS))
        except Exception: pass
    _save(raw)

    cfg = yaml.safe_load(open(STOCH_CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase5b] device={device} | dropout={cfg['model']['dropout']} | concat [E(S_t); Δ_t]", flush=True)

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
        for ortho in ("soft", "none"):
            if str(seed) in raw.get(ortho, {}):
                print(f"[Phase5b] skip {ortho}/{seed} (done)", flush=True)
                continue
            f1 = train_concat(seed, STOCH_CONFIG, ortho, loaders, device)
            raw.setdefault(ortho, {})[str(seed)] = f1
            _save(raw)

    # Summary
    from scipy import stats
    def mci(vals):
        a = np.array(vals, float)
        m, se = a.mean(), stats.sem(a)
        return float(m), float(se * stats.t.ppf(0.975, len(a)-1)) if len(a) > 1 else 0.0
    raw["summary"] = {}
    for ortho in ("soft", "none"):
        vals = [raw[ortho][str(s)] for s in seeds if str(s) in raw.get(ortho, {})]
        if vals:
            m, c = mci(vals)
            raw["summary"][ortho] = {"mean": m, "ci95": c}
    if "soft" in raw.get("summary", {}) and "none" in raw.get("summary", {}):
        raw["summary"]["ortho_delta"] = raw["summary"]["soft"]["mean"] - raw["summary"]["none"]["mean"]
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase5b] DONE -> {RESULTS}", flush=True)
    for ortho in ("soft", "none"):
        if ortho in raw.get("summary", {}):
            s = raw["summary"][ortho]
            print(f"  CONCAT {ortho} Test F1 = {s['mean']*100:.2f} ± {s['ci95']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
