"""
Phase-4b: Honest-variance DI-MMGN re-run with dropout=0.15.
See 8_Project_Management/seeding_audit.md for rationale.

Trains DI-MMGN (soft + none) for seeds {42,43,44} using config/default_stochastic.yaml.
- Baseline is SKIPPED (it already shows real seed variance; CI ±3.03 from Phase 4).
- Saves per-sample test predictions -> enables a complementary bootstrap CI.
- Crash-safe incremental results -> phase4b_results_raw.json (resumes completed cells).
"""
import os
import sys
import copy
import json
import uuid
import yaml
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_dimmgn import train_dimmgn, load_config
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN

STOCH_CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/phase4b_results_raw.json"
PREDS_DIR = "../8_Project_Management/phase4b_predictions"


def _save_raw(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


def eval_with_preds(model, loader, device):
    """Return (f1, preds[], labels[]) over the test loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels_t = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            p = (torch.sigmoid(out["score"]) > 0.5).long()
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(labels_t.cpu().numpy().tolist())
    f1 = f1_score(labels, preds, zero_division=0)
    return float(f1), preds, labels


def run_variant(seed, ortho_mode, test_loader, device):
    cfg = load_config(STOCH_CONFIG)
    cfg["model"]["orthogonality"] = ortho_mode
    tag = f"{seed}_{ortho_mode}_{uuid.uuid4().hex[:6]}"
    cfg_path = f"config/tempb_{tag}.yaml"
    ckpt_path = f"dimmgnb_{tag}.pt"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"\n--- [Phase4b] DI-MMGN dropout=0.15 [Ortho={ortho_mode}, Seed={seed}] ---", flush=True)
    try:
        train_dimmgn(seed=seed, config_path=cfg_path, ortho_override=ortho_mode, ckpt_path=ckpt_path)
        model = DI_MMGN(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        f1, preds, labels = eval_with_preds(model, test_loader, device)
        # persist per-sample predictions for bootstrap
        os.makedirs(PREDS_DIR, exist_ok=True)
        with open(os.path.join(PREDS_DIR, f"{ortho_mode}_{seed}.json"), "w") as f:
            json.dump({"f1": f1, "preds": preds, "labels": labels}, f)
        return f1
    finally:
        for p in (cfg_path, ckpt_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[warn] remove {p}: {e}", flush=True)


def main():
    seeds = [42, 43, 44]
    raw = {"soft": {}, "none": {}, "config": "dropout=0.15", "done": False}
    if os.path.exists(RESULTS):
        try:
            raw = json.load(open(RESULTS))
        except Exception:
            pass
    _save_raw(raw)

    cfg = load_config(STOCH_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase4b] device={device} | dropout={cfg['model']['dropout']} | "
          f"epochs_pretrain={cfg['train']['epochs_pretrain']} epochs_finetune={cfg['train']['epochs_finetune']}", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_pairs = [p for p in full.pairs if p["domain"] in splits["test"]]
    test_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                                max_text_len=cfg["data"]["max_text_len"], pairs=test_pairs)
    collate_fn = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch"], shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    for seed in seeds:
        for ortho in ("soft", "none"):
            if str(seed) in raw.get(ortho, {}):
                print(f"[Phase4b] skip {ortho}/{seed} (already done)", flush=True)
                continue
            f1 = run_variant(seed, ortho, test_loader, device)
            raw.setdefault(ortho, {})[str(seed)] = f1
            _save_raw(raw)
            print(f"[Phase4b] {ortho}/{seed} Test F1 = {f1:.4f}", flush=True)

    # summarize
    def m(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)) * 1.96) if len(a) > 1 else 0.0
    sm, sc = m([raw["soft"][str(s)] for s in seeds])
    nm, nc = m([raw["none"][str(s)] for s in seeds])
    raw["summary"] = {"soft_mean": sm, "soft_ci95": sc, "none_mean": nm, "none_ci95": nc,
                      "ortho_delta": sm - nm, "dropout": 0.15}
    raw["done"] = True
    _save_raw(raw)
    print(f"\n[Phase4b] DONE  soft={sm*100:.2f}±{sc*100:.2f}  none={nm*100:.2f}±{nc*100:.2f}  "
          f"orthoΔ={(sm-nm)*100:.2f}pt  -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
