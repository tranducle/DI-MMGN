"""Experiment #3 — Constructive positive for the inert-adaptation reframing (ch05).

The reviewer/playbook noted that updating d_legit is inert in CONCAT (d_legit is
not in the inference path). This script supplies the missing *positive*: a small
learnable bias b added to E(S_t) BEFORE concatenation (DI_MMGN_ConcatAdapter),
which IS in the inference path, so updating b can change the score.

Protocol (fixes the test-reuse flaw):
  * Warm-start the adapter from the already-trained CONCAT checkpoint
    (concat_retention_model.pt), with b = 0 -> identical to CONCAT at step 0.
  * Adapt b online using confirmed-legitimate CMS pairs from the VAL split
    (domain-disjoint from test), pushing their score toward "legitimate".
  * Evaluate F1 / FP on the TEST split (disjoint domains) after every 10 steps.

If test F1/FP moves as b is adapted on val, the constructive positive holds
(inference-path adaptation works on a held-out split). If it does not move,
that is an honest negative. Either outcome resolves RQ3 with evidence.
"""
import os, sys, json, yaml, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN_ConcatAdapter

CONFIG = "config/default_stochastic.yaml"
RESULTS = "../8_Project_Management/exp3_adapter_retention.json"
CKPT = "concat_retention_model.pt"


def set_seed(seed=42):
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
    legit = labels == 0
    fp = int((preds[legit] == 1).sum()); n_legit = int(legit.sum())
    atk = labels == 1
    attack_recall = float((preds[atk] == 1).sum() / max(atk.sum(), 1))
    return {"f1": float(f1), "fp": fp, "n_legit": n_legit,
            "fp_rate": fp / max(n_legit, 1), "attack_recall": attack_recall}


def get_val_legit_cms_pairs(pairs):
    out = []
    for p in pairs:
        if p.get("label") == "legitimate" and p.get("attack_type") == "legit_cms":
            out.append(p)
    return out


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CONFIG))
    print(f"[Exp3] device={device}", flush=True)

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
        return DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=shuffle, collate_fn=collate, num_workers=0)
    test_loader = loader("test", False)

    # val legit-CMS pairs for adaptation (domain-disjoint from test)
    val_legit_cms = get_val_legit_cms_pairs([p for p in full.pairs if p["domain"] in splits["val"]])
    n_adapt = min(50, len(val_legit_cms))
    print(f"[Exp3] val legit-CMS pairs available={len(val_legit_cms)}; using {n_adapt} for adaptation", flush=True)
    adapt_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                                 max_text_len=cfg["data"]["max_text_len"], pairs=val_legit_cms[:n_adapt])
    adapt_loader = DataLoader(adapt_ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)

    # Load CONCAT checkpoint into the adapter (b=0 -> identical to CONCAT)
    model = DI_MMGN_ConcatAdapter(cfg).to(device)
    missing, unexpected = model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True), strict=False)
    print(f"[Exp3] loaded {CKPT} | missing keys (expected: adapt_bias)={missing}", flush=True)
    # Freeze everything except adapt_bias
    for n, p in model.named_parameters():
        p.requires_grad = (n == "adapt_bias")
    opt = torch.optim.SGD([model.adapt_bias], lr=cfg.get("adapt", {}).get("online_lr", 0.01) * 50)  # larger step for a bias

    raw = {"ckpt": CKPT, "n_adapt": n_adapt, "split": "adapt_on=val legit-CMS, eval_on=test (domain-disjoint)",
           "curve": [], "done": False}

    # Baseline (b=0)
    base = evaluate(model, test_loader, device)
    raw["baseline"] = base
    print(f"[Exp3] baseline (b=0): F1={base['f1']*100:.2f} FP={base['fp']}/{base['n_legit']} ({base['fp_rate']*100:.1f}%) attack_recall={base['attack_recall']*100:.2f}", flush=True)
    raw["curve"].append({"step": 0, **base})

    checkpoints = set(range(0, n_adapt + 1, max(1, n_adapt // 5)))
    checkpoints.add(n_adapt)
    adapt_iter = iter(adapt_loader)
    for step in range(1, n_adapt + 1):
        try:
            b = next(adapt_iter)
        except StopIteration:
            adapt_iter = iter(adapt_loader); b = next(adapt_iter)
        model.train()
        t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
        t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
        mask = b["mask"].to(device)
        out = model({"t1": t1, "t2": t2, "mask": mask})
        # confirmed-legitimate -> push score toward 0 (label 0)
        loss = F.binary_cross_entropy_with_logits(out["score"], torch.zeros_like(out["score"]))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in checkpoints:
            r = evaluate(model, test_loader, device)
            raw["curve"].append({"step": step, **r})
            print(f"[Exp3] step {step:3d}/{n_adapt}: F1={r['f1']*100:.2f} FP={r['fp']}/{r['n_legit']} ({r['fp_rate']*100:.1f}%) attack_recall={r['attack_recall']*100:.2f}", flush=True)

    final = raw["curve"][-1]
    raw["final"] = final
    raw["f1_delta"] = final["f1"] - base["f1"]
    raw["fp_delta"] = final["fp"] - base["fp"]
    raw["verdict"] = ("adapter_moves_score" if abs(raw["f1_delta"]) > 1e-6 or raw["fp_delta"] != 0 else "no_effect")
    raw["done"] = True
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n[Exp3] verdict={raw['verdict']} | F1 delta={raw['f1_delta']*100:+.2f}pt | FP delta={raw['fp_delta']:+d}", flush=True)
    print(f"[Exp3] DONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
