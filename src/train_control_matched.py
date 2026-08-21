"""Experiment #2 — Parameter-matched confound control (ch04 'Ablation cùng tham số').

Does the +change-vector gain come from the Δ_t signal or from the extra 2d->d MLP
capacity that concatenation adds? Train TWO models under an IDENTICAL reduced
protocol (same encoder, same classifier parameter count, same optimizer/split/
seed/epochs) that differ in exactly one line:

  * CONCAT  : combined = [E(S_t);     Δ_t]
  * ZERO-Δ  : combined = [E(S_t); zeros_like(Δ_t)]   (same classifier params)

If CONCAT >> ZERO-Δ, the gain is the Δ signal, not head capacity.
If CONCAT ≈ ZERO-Δ, the gain is confounded by capacity.

NOTE: reduced epoch budget (pretrain=5, finetune=10) for tractability; BOTH
models use the identical budget so the comparison is fair. The absolute F1 will
be lower than the 30/30 headline (95.36); the GAP is the quantity of interest
and is what we report. Full 30/30 x 3-seed run is recommended for the final
version; this is a matched-protocol pilot with real training.
"""
import os, sys, json, yaml, copy, uuid
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
RESULTS = "../8_Project_Management/exp2_control_matched.json"
EPOCHS_PRETRAIN = 5
EPOCHS_FINETUNE = 10
BATCH = 32
SEED = 42


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


def train_one(ModelCls, cfg, loaders, device, tag):
    set_seed(SEED)
    train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]
    model = ModelCls(cfg).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    cl_fn = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Exp2] === {tag} (trainable params={nparams}) ===", flush=True)

    # Phase A: contrastive pretrain (legit-only) — skip if model has no ortho use; keep for parity
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
        print(f"  [{tag}] pretrain ep {ep+1}/{EPOCHS_PRETRAIN} loss {tl/len(train_loader):.4f}", flush=True)

    # Phase B: finetune with composite loss
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
        print(f"  [{tag}] finetune ep {ep+1}/{EPOCHS_FINETUNE} train {tl/len(train_loader):.4f} valF1 {vf:.4f}", flush=True)
        if vf > best[0]:
            best = (vf, copy.deepcopy(model.state_dict()))

    model.load_state_dict(best[1])
    tf1, tacc, ar, ls = evaluate(model, test_loader, device)
    print(f"[Exp2] {tag} TEST F1={tf1*100:.2f} acc={tacc*100:.2f} attack_recall={ar*100:.2f} legit_spec={ls*100:.2f}", flush=True)
    return {"f1": tf1, "acc": tacc, "attack_recall": ar, "legit_spec": ls, "nparams": nparams, "best_val_f1": best[0]}


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CONFIG))
    cfg["train"]["batch"] = BATCH
    print(f"[Exp2] device={device} | protocol: pretrain={EPOCHS_PRETRAIN} finetune={EPOCHS_FINETUNE} batch={BATCH} seed={SEED}", flush=True)

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

    concat = train_one(DI_MMGN_Concat, cfg, loaders, device, "CONCAT")
    zerodelta = train_one(DI_MMGN_ConcatZeroDelta, cfg, loaders, device, "ZERO-DELTA")

    out = {"protocol": {"pretrain": EPOCHS_PRETRAIN, "finetune": EPOCHS_FINETUNE, "batch": BATCH, "seed": SEED},
           "concat": concat, "zero_delta": zerodelta,
           "f1_gap_concat_minus_zerodelta": concat["f1"] - zerodelta["f1"],
           "param_delta": zerodelta["nparams"] - concat["nparams"],
           "done": True}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Exp2] F1 gap (CONCAT - ZERO-Δ) = {(concat['f1']-zerodelta['f1'])*100:+.2f} pt | param_delta={out['param_delta']} (0 => matched)", flush=True)
    print(f"[Exp2] DONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
