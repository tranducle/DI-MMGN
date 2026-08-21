"""Experiment #2 (FULL) — Parameter-matched confound control, 30/30 epochs x 3 seeds.

Upgrades the pilot (train_control_matched.py: 5/10 epochs, seed 42 only) to the
full headline protocol: EPOCHS_PRETRAIN=30, EPOCHS_FINETUNE=30, SEEDS={42,43,44}.
Trains CONCAT and ZERO-Δ under an identical protocol per seed and reports the
per-seed F1 gap plus the 3-seed mean ± 95% CI (t-distribution).

The quantity of interest is the GAP (CONCAT - ZERO-Δ), not the absolute F1; both
models share the exact trainable parameter count, so a positive gap is the Δ_t
signal, not head capacity.
"""
import os, sys, json, yaml, copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN_Concat, DI_MMGN_ConcatZeroDelta
from models.losses import composite_loss, ContrastiveLegitLoss

CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/exp2_control_matched_FULL.json"
EPOCHS_PRETRAIN = 30
EPOCHS_FINETUNE = 30
BATCH = 32
SEEDS = [42, 43, 44]


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
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
    preds = np.array(preds); labels = np.array(labels)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    atk = labels == 1
    attack_recall = float((preds[atk] == 1).sum() / max(atk.sum(), 1))
    legit = labels == 0
    legit_spec = float((preds[legit] == 0).sum() / max(legit.sum(), 1))
    return float(f1), float(acc), attack_recall, legit_spec


def train_one(ModelCls, cfg, loaders, device, tag, seed):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]
    model = ModelCls(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    cl_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Exp2-FULL] === {tag} seed={seed} (trainable params={nparams}) ===", flush=True)

    for ep in range(EPOCHS_PRETRAIN):
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
        print(f"  [{tag} s{seed}] pretrain ep {ep+1}/{EPOCHS_PRETRAIN} loss {tl/len(train_loader):.4f}", flush=True)

    best = (-1.0, None)
    for ep in range(EPOCHS_FINETUNE):
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
        vf, _, _, _ = evaluate(model, val_loader, device)
        print(f"  [{tag} s{seed}] finetune ep {ep+1}/{EPOCHS_FINETUNE} train {tl/len(train_loader):.4f} valF1 {vf:.4f}", flush=True)
        if vf > best[0]:
            best = (vf, copy.deepcopy(model.state_dict()))

    model.load_state_dict(best[1])
    tf1, tacc, ar, ls = evaluate(model, test_loader, device)
    print(f"[Exp2-FULL] {tag} s{seed} TEST F1={tf1*100:.2f} acc={tacc*100:.2f} attack_recall={ar*100:.2f} legit_spec={ls*100:.2f}", flush=True)
    return {"f1": tf1, "acc": tacc, "attack_recall": ar, "legit_spec": ls, "nparams": nparams, "best_val_f1": best[0]}


def ci95(vals):
    vals = np.array(vals)
    n = len(vals)
    mean = vals.mean()
    std = vals.std(ddof=1) if n > 1 else 0.0
    # t-distribution with n-1 dof, 95% two-sided
    import scipy.stats as st
    t = st.t.ppf(0.975, n - 1)
    return mean, t * std / np.sqrt(n)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CONFIG))
    cfg["train"]["batch"] = BATCH
    print(f"[Exp2-FULL] device={device} | protocol: pretrain={EPOCHS_PRETRAIN} finetune={EPOCHS_FINETUNE} batch={BATCH} seeds={SEEDS}", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def loader(split, shuffle):
        ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                               max_text_len=cfg["data"]["max_text_len"],
                               pairs=[p for p in full.pairs if p["domain"] in splits[split]])
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=collate, num_workers=0)
    loaders = (loader("pretrain", True), loader("val", False), loader("test", False))

    per_seed = {}
    for seed in SEEDS:
        concat = train_one(DI_MMGN_Concat, cfg, loaders, device, "CONCAT", seed)
        zerodelta = train_one(DI_MMGN_ConcatZeroDelta, cfg, loaders, device, "ZERO-DELTA", seed)
        per_seed[str(seed)] = {
            "concat": concat, "zero_delta": zerodelta,
            "f1_gap_concat_minus_zerodelta": concat["f1"] - zerodelta["f1"],
            "param_delta": zerodelta["nparams"] - concat["nparams"],
        }
        gap = (concat["f1"] - zerodelta["f1"]) * 100
        print(f"[Exp2-FULL] seed={seed} F1 gap (CONCAT - ZERO-Δ) = {gap:+.2f} pt | param_delta={zerodelta['nparams'] - concat['nparams']} (0 => matched)", flush=True)

    gaps = [per_seed[str(s)]["f1_gap_concat_minus_zerodelta"] for s in SEEDS]
    mean_gap, ci_gap = ci95(gaps)
    out = {
        "protocol": {"pretrain": EPOCHS_PRETRAIN, "finetune": EPOCHS_FINETUNE, "batch": BATCH, "seeds": SEEDS},
        "per_seed": per_seed,
        "aggregate": {
            "mean_gap_concat_minus_zerodelta": float(mean_gap),
            "ci95_gap": float(ci_gap),
            "mean_gap_pt": float(mean_gap * 100),
            "ci95_gap_pt": float(ci_gap * 100),
            "param_delta_all_zero": all(per_seed[str(s)]["param_delta"] == 0 for s in SEEDS),
        },
        "done": True,
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Exp2-FULL] mean gap = {mean_gap*100:+.2f} ± {ci_gap*100:.2f} pt (3 seeds) | param_delta_all_zero={out['aggregate']['param_delta_all_zero']}", flush=True)
    print(f"[Exp2-FULL] DONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()