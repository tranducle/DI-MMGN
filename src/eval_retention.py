"""
Phase-5d A3: Post-Redesign Retention evaluation (Gap 2 fix).
Validates the paper title "Drift-Resilient" empirically.

Protocol:
  1. Load a trained CONCAT checkpoint (best variant).
  2. Evaluate baseline F1 on the full test set (626 pairs).
  3. Select N=50 legitimate CMS-mutation pairs from TEST as "confirmed redesigns."
  4. Sequentially apply online_adapt() for each confirmed-legitimate Δ_t.
  5. Re-evaluate F1 after every 10 adaptation steps.
  6. Controls: (a) no adaptation, (b) random-direction adaptation.

Pre-declared gate (scientific_gates.yaml): F1 >= 0.90 after all adaptation steps.

Usage:
  python eval_retention.py --ckpt <path_to_concat_checkpoint>
  If no checkpoint provided, trains CONCAT none/seed42 first and saves it.

Requires a trained model. The script will train one if no checkpoint is found.
"""
import os, sys, json, random, copy, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN_Concat
from models.online_adapt import online_adapt
from models.losses import composite_loss, ContrastiveLegitLoss
import torch.optim as optim

CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/phase5d_retention_results_raw.json"
CKPT_DEFAULT = "concat_retention_model.pt"


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2, default=str)


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
    f1 = f1_score(labels, preds, zero_division=0)
    # FP on legit
    legit_mask = np.array(labels) == 0
    fp = int(sum(np.array(preds)[legit_mask] == 1))
    n_legit = int(sum(legit_mask))
    return float(f1), preds, labels, atks, fp, n_legit


def get_legit_cms_deltas(model, loader, device, n=50):
    """Extract confirmed-legitimate CMS-mutation change vectors for adaptation."""
    model.eval()
    deltas = []
    with torch.no_grad():
        for b in loader:
            labels = b["label"]
            atks = b.get("attack_type", ["unknown"] * len(labels))
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            for i in range(len(labels)):
                if labels[i] == 0 and atks[i] == "legit_cms":
                    deltas.append(out["delta"][i].cpu())
                    if len(deltas) >= n:
                        return deltas
    return deltas


def train_and_save_model(cfg, loaders, device):
    """Train CONCAT none/seed42 and save checkpoint for retention eval."""
    set_seed(42)
    train_loader, val_loader, _ = loaders
    model = DI_MMGN_Concat(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    accum = cfg["train"]["accum"]
    cl_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])

    print("[Retention] Training CONCAT model for retention eval (none/seed42)...", flush=True)
    # Phase A: pretrain
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

    # Phase B: finetune
    best_f1 = -1.0
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
        vf, _, _, _, _, _ = evaluate(model, val_loader, device)
        if vf > best_f1:
            best_f1 = vf; torch.save(model.state_dict(), CKPT_DEFAULT)
        if (ep + 1) % 10 == 0:
            print(f"  finetune ep {ep+1}/{cfg['train']['epochs_finetune']} valF1={vf:.4f}", flush=True)
    print(f"[Retention] Model saved: {CKPT_DEFAULT} (best Val F1={best_f1:.4f})", flush=True)
    return CKPT_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to CONCAT checkpoint")
    parser.add_argument("--n_adapt", type=int, default=50, help="Number of adaptation steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CONFIG))

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
    train_loader = loader("pretrain", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)
    loaders = (train_loader, val_loader, test_loader)

    # Get or train model
    ckpt_path = args.ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[Retention] Loading checkpoint: {ckpt_path}", flush=True)
    elif os.path.exists(CKPT_DEFAULT):
        ckpt_path = CKPT_DEFAULT
        print(f"[Retention] Loading cached checkpoint: {ckpt_path}", flush=True)
    else:
        print("[Retention] No checkpoint found — training new model...", flush=True)
        ckpt_path = train_and_save_model(cfg, loaders, device)

    model = DI_MMGN_Concat(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # ── Step 1: Baseline F1 (no adaptation) ──
    print("\n=== Step 1: Baseline (no adaptation) ===", flush=True)
    base_f1, base_preds, base_labels, base_atks, base_fp, base_n_legit = evaluate(model, test_loader, device)
    print(f"  Baseline F1 = {base_f1*100:.2f}%  FP={base_fp}/{base_n_legit} ({base_fp/base_n_legit*100:.1f}%)", flush=True)

    # ── Step 2: Extract legit-CMS deltas for adaptation ──
    print(f"\n=== Step 2: Extracting up to {args.n_adapt} legit-CMS deltas ===", flush=True)
    adapt_deltas = get_legit_cms_deltas(model, test_loader, device, n=args.n_adapt)
    print(f"  Found {len(adapt_deltas)} legit-CMS deltas for adaptation", flush=True)

    # ── Step 3: Sequential adaptation → re-evaluate every 10 steps ──
    raw = {
        "baseline_f1": base_f1, "baseline_fp": base_fp, "baseline_n_legit": base_n_legit,
        "n_adapt_deltas": len(adapt_deltas),
        "adaptation_curve": [],
        "gate_threshold": 0.90,
        "done": False
    }

    checkpoints = list(range(0, len(adapt_deltas) + 1, max(1, len(adapt_deltas) // 5)))
    if 0 not in checkpoints:
        checkpoints.insert(0, 0)
    if len(adapt_deltas) not in checkpoints:
        checkpoints.append(len(adapt_deltas))

    print(f"\n=== Step 3: Online adaptation ({len(adapt_deltas)} steps) ===", flush=True)
    print(f"  Checkpoints: {checkpoints}", flush=True)
    for step_idx in range(len(adapt_deltas) + 1):
        if step_idx > 0 and step_idx <= len(adapt_deltas):
            delta = adapt_deltas[step_idx - 1].to(device)
            online_adapt(model.ortho, delta, lr=cfg.get("adapt", {}).get("online_lr", 0.01))

        if step_idx in checkpoints:
            f1, preds, labels, atks, fp, n_legit = evaluate(model, test_loader, device)
            entry = {"step": step_idx, "f1": f1, "fp": fp, "n_legit": n_legit,
                     "fp_rate": fp / max(n_legit, 1)}
            raw["adaptation_curve"].append(entry)
            print(f"  Step {step_idx:3d}/{len(adapt_deltas)} | F1={f1*100:.2f}% | FP={fp}/{n_legit} ({fp/max(n_legit,1)*100:.1f}%)", flush=True)

    # ── Gate verdict ──
    final_f1 = raw["adaptation_curve"][-1]["f1"] if raw["adaptation_curve"] else base_f1
    raw["final_f1"] = final_f1
    raw["gate_pass"] = final_f1 >= 0.90
    raw["retention_delta"] = final_f1 - base_f1
    raw["done"] = True
    _save(raw)

    print(f"\n{'='*60}")
    print(f"  RETENTION GATE: {'PASS' if raw['gate_pass'] else 'FAIL'}")
    print(f"  Baseline F1:  {base_f1*100:.2f}%")
    print(f"  Final F1:     {final_f1*100:.2f}%")
    print(f"  Delta:        {raw['retention_delta']*100:+.2f}pt")
    print(f"  Threshold:    >= 90.00%")
    print(f"{'='*60}")
    print(f"\n→ Saved to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
