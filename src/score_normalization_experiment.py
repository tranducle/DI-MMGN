"""
Score normalization experiment: test if normalizing the detection score
by |Δ_t| fixes the catastrophic FP on CMS mutations.

Loads the saved DI-MMGN checkpoint (dimmgn_best.pt), forward-passes
the test set, and compares absolute vs cosine (normalized) scoring.

No training — forward-only, ~2 min on GPU.
"""
import os, sys, json, torch, numpy as np
import yaml
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN

CKPT = "dimmgn_best.pt"  # Phase 4 main checkpoint (no dropout)
UNIFIED = "../dataset_pipeline/v2/unified_pairs.jsonl"
DATA_ROOT = "../dataset_pipeline"
SPLITS = "data/splits.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    cfg = yaml.safe_load(open("config/default.yaml"))
    with open(SPLITS) as f:
        splits = json.load(f)

    # Load model
    print(f"Loading checkpoint: {CKPT} ...")
    model = DI_MMGN(cfg).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load test set
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=UNIFIED, data_root=DATA_ROOT,
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_pairs = [p for p in full.pairs if p["domain"] in splits["test"]]
    test_ds = DimmgnPairDataset(data_root=DATA_ROOT, tokenizer=tok,
                                max_text_len=cfg["data"]["max_text_len"], pairs=test_pairs)
    collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)

    # Metadata
    meta = [(0 if p.get("label") == "legitimate" else 1, p.get("attack_type", "unknown"))
            for p in test_pairs]

    # Forward pass → collect deltas, scores
    all_abs_scores = []   # |Δ · d_deface|
    all_cos_scores = []   # |cos(Δ, d_deface)|
    all_legit_proj = []   # Δ · d_legit
    all_labels = []
    all_deltas_norm = []

    print("Forward-passing test set...")
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})

            delta = out["delta"]  # [B, d]
            d_deface = out["d_deface"]  # [d]
            d_legit = out["d_legit"]    # [d]

            abs_score = (delta @ d_deface).abs()       # [B]
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cos_score = abs_score / delta_norm.squeeze(-1)  # [B]
            legit_p = delta @ d_legit                    # [B]

            all_abs_scores.extend(abs_score.cpu().numpy().tolist())
            all_cos_scores.extend(cos_score.cpu().numpy().tolist())
            all_legit_proj.extend(legit_p.cpu().numpy().tolist())
            all_deltas_norm.extend(delta_norm.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_abs_scores = np.array(all_abs_scores)
    all_cos_scores = np.array(all_cos_scores)
    all_legit_proj = np.array(all_legit_proj)
    all_deltas_norm = np.array(all_deltas_norm)
    all_labels = np.array(all_labels)

    # ── Compare scoring modes at various thresholds ──
    print("\n" + "=" * 70)
    print("SCORING MODE COMPARISON")
    print("=" * 70)

    def eval_at_threshold(scores, labels, threshold):
        preds = (scores > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return f1, acc, fp, fn, tp, tn

    def find_best_threshold(scores, labels):
        """Find threshold maximizing F1."""
        best_f1, best_t, best_metrics = 0, 0, None
        for t in np.percentile(scores, np.arange(1, 100, 0.5)):
            f1, acc, fp, fn, tp, tn = eval_at_threshold(scores, labels, t)
            if f1 >= best_f1:
                best_f1 = f1
                best_t = t
                best_metrics = (f1, acc, fp, fn, tp, tn)
        return best_t, best_metrics

    # Also test sigmoid > 0.5 (current production threshold)
    print("\n--- Mode 1: ABSOLUTE SCORE (current) |Δ · d_deface| ---")
    # Current production: sigmoid(score) > 0.5 → score > 0 (since logit(0.5) = 0)
    f1_prod, acc_prod, fp_prod, fn_prod, tp_prod, tn_prod = eval_at_threshold(all_abs_scores, all_labels, 0.0)
    print(f"  Production threshold (>0):  F1={f1_prod*100:.2f}%  Acc={acc_prod*100:.2f}%  FP={fp_prod}  FN={fn_prod}")
    best_t_abs, (f1_abs, acc_abs, fp_abs, fn_abs, tp_abs, tn_abs) = find_best_threshold(all_abs_scores, all_labels)
    print(f"  Optimal threshold ({best_t_abs:.4f}):  F1={f1_abs*100:.2f}%  Acc={acc_abs*100:.2f}%  FP={fp_abs}  FN={fn_abs}")

    print("\n--- Mode 2: COSINE SCORE (normalized) |cos(Δ, d_deface)| ---")
    best_t_cos, (f1_cos, acc_cos, fp_cos, fn_cos, tp_cos, tn_cos) = find_best_threshold(all_cos_scores, all_labels)
    print(f"  Optimal threshold ({best_t_cos:.4f}):  F1={f1_cos*100:.2f}%  Acc={acc_cos*100:.2f}%  FP={fp_cos}  FN={fn_cos}")

    print("\n--- Mode 3: RATIO SCORE |Δ·d_deface| / (|Δ·d_legit| + ε) ---")
    all_ratio_scores = all_abs_scores / (np.abs(all_legit_proj) + 1e-8)
    best_t_ratio, (f1_ratio, acc_ratio, fp_ratio, fn_ratio, tp_ratio, tn_ratio) = find_best_threshold(all_ratio_scores, all_labels)
    print(f"  Optimal threshold ({best_t_ratio:.4f}):  F1={f1_ratio*100:.2f}%  Acc={acc_ratio*100:.2f}%  FP={fp_ratio}  FN={fn_ratio}")

    # ── Per-attack-type with best thresholds ──
    print("\n" + "=" * 70)
    print("PER-ATTACK-TYPE BREAKDOWN (optimal thresholds)")
    print("=" * 70)

    for mode_name, scores, threshold in [
        ("ABSOLUTE", all_abs_scores, best_t_abs),
        ("COSINE", all_cos_scores, best_t_cos),
    ]:
        preds = (scores > threshold).astype(int)
        print(f"\n--- {mode_name} (threshold={threshold:.4f}) ---")
        print(f"{'Attack Type':<25} {'N':>5} {'Pred+':>6} {'FP':>5} {'FN':>5} {'F1%':>8}")
        print("─" * 55)
        groups = defaultdict(lambda: {"labels": [], "preds": []})
        for i, (label, atk) in enumerate(meta):
            atk = atk if atk is not None else "unknown"
            groups[atk]["labels"].append(all_labels[i])
            groups[atk]["preds"].append(preds[i])
        for atk in sorted(groups.keys()):
            g = groups[atk]
            n = len(g["labels"])
            p = sum(g["preds"])
            fp = sum(1 for pr, l in zip(g["preds"], g["labels"]) if pr == 1 and l == 0)
            fn = sum(1 for pr, l in zip(g["preds"], g["labels"]) if pr == 0 and l == 1)
            f1 = f1_score(g["labels"], g["preds"], zero_division=0) * 100
            print(f"{atk:<25} {n:>5} {p:>6} {fp:>5} {fn:>5} {f1:>8.2f}")

    # ── Δ norm analysis ──
    print("\n" + "=" * 70)
    print("DELTA NORM ANALYSIS (why absolute score fails on CMS)")
    print("=" * 70)
    for atk in sorted(set(a if a is not None else "unknown" for _, a in meta)):
        mask = np.array([(a if a is not None else "unknown") == atk for _, a in meta])
        if mask.sum() > 0:
            norms = all_deltas_norm[mask]
            print(f"  {atk:<25}  |Δ| mean={norms.mean():.4f}  std={norms.std():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")

    print("\n→ If cosine score has MUCH lower FP on legit_cms, the fix is to")
    print("  normalize the detection score in the model's forward() method.")
    print("  Then retrain with the normalized score.")


if __name__ == "__main__":
    main()
