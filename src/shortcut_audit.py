"""Experiment #1 — Shortcut audit (reviewer7 / ieee-transactions-playbook ch03-ch04 red flag).

Question: the pure-delta model reports 100% attack recall. Is that a genuine
separation, or a label-correlated shortcut in one modality? We ablate each
modality at inference (force its mask bit off) and re-measure attack recall and
overall F1. If 100% recall survives dropping every single modality, the signal
is distributed; if it collapses when one modality is removed, that modality is
carrying a shortcut.

Runs on the trained pure-delta (DI_MMGN) and CONCAT (DI_MMGN_Concat) checkpoints.
Inference only — no training.
"""
import os, sys, json, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN, DI_MMGN_Concat

CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/exp1_shortcut_audit.json"
MOD_NAMES = ["text", "dom", "visual", "http"]   # mask order


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def eval_with_ablation(model, loader, device, ablate_idx=None):
    """ablate_idx: int in 0..3 to force that modality OFF, or None for all-on."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].clone().to(device)
            if ablate_idx is not None:
                mask[:, ablate_idx] = False       # force-ablate this modality
                # avoid empty mask (would divide by clamp(min=1)); keep at least text if all off
                if mask.sum(dim=1).min() == 0:
                    mask[:, 0] = True
            out = model({"t1": t1, "t2": t2, "mask": mask})
            p = (torch.sigmoid(out["score"]) > 0.5).long()
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    preds = np.array(preds); labels = np.array(labels)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    # attack recall = recall on defaced (label 1)
    defaced = labels == 1
    attack_recall = float((preds[defaced] == 1).sum() / max(defaced.sum(), 1))
    # legit precision = precision on the legitimate class (label 0) i.e. TN rate
    legit = labels == 0
    legit_specificity = float((preds[legit] == 0).sum() / max(legit.sum(), 1))
    return {"f1": float(f1), "acc": float(acc), "attack_recall": attack_recall,
            "legit_specificity": legit_specificity,
            "n_defaced": int(defaced.sum()), "n_legit": int(legit.sum())}


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CONFIG))
    print(f"[Exp1] device={device}", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_pairs = [p for p in full.pairs if p["domain"] in splits["test"]]
    collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                                max_text_len=cfg["data"]["max_text_len"], pairs=test_pairs)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch"], shuffle=False, collate_fn=collate, num_workers=0)
    print(f"[Exp1] test pairs = {len(test_ds)}", flush=True)

    out = {"models": {}}

    # ---- Pure-delta model ----
    pure_ckpt = "dimmgn_best.pt"
    if os.path.exists(pure_ckpt):
        print(f"\n[Exp1] Pure-delta (DI_MMGN) <- {pure_ckpt}", flush=True)
        pure = DI_MMGN(cfg).to(device)
        pure.load_state_dict(torch.load(pure_ckpt, map_location=device, weights_only=True))
        rows = {"all_on": eval_with_ablation(pure, test_loader, device, None)}
        print(f"  all_on: F1={rows['all_on']['f1']*100:.2f} attack_recall={rows['all_on']['attack_recall']*100:.2f} legit_spec={rows['all_on']['legit_specificity']*100:.2f}", flush=True)
        for k, name in enumerate(MOD_NAMES):
            r = eval_with_ablation(pure, test_loader, device, k)
            rows[f"drop_{name}"] = r
            print(f"  drop_{name:6s}: F1={r['f1']*100:.2f} attack_recall={r['attack_recall']*100:.2f} legit_spec={r['legit_specificity']*100:.2f}", flush=True)
        out["models"]["pure_delta"] = rows
    else:
        print(f"[Exp1] {pure_ckpt} not found — skipping pure-delta.", flush=True)

    # ---- CONCAT model ----
    concat_ckpt = "concat_retention_model.pt"
    if os.path.exists(concat_ckpt):
        print(f"\n[Exp1] CONCAT (DI_MMGN_Concat) <- {concat_ckpt}", flush=True)
        conc = DI_MMGN_Concat(cfg).to(device)
        conc.load_state_dict(torch.load(concat_ckpt, map_location=device, weights_only=True))
        rows = {"all_on": eval_with_ablation(conc, test_loader, device, None)}
        print(f"  all_on: F1={rows['all_on']['f1']*100:.2f} attack_recall={rows['all_on']['attack_recall']*100:.2f} legit_spec={rows['all_on']['legit_specificity']*100:.2f}", flush=True)
        for k, name in enumerate(MOD_NAMES):
            r = eval_with_ablation(conc, test_loader, device, k)
            rows[f"drop_{name}"] = r
            print(f"  drop_{name:6s}: F1={r['f1']*100:.2f} attack_recall={r['attack_recall']*100:.2f} legit_spec={r['legit_specificity']*100:.2f}", flush=True)
        out["models"]["concat"] = rows
    else:
        print(f"[Exp1] {concat_ckpt} not found — skipping concat.", flush=True)

    out["done"] = True
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Exp1] DONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
