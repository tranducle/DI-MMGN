"""
Per-attack-vector breakdown + false-positive analysis.
Reads saved predictions from phase4b and computes F1 per attack type,
FP rate on legitimate mutations, and comparison tables.

No GPU needed — pure CPU analysis on saved predictions + dataset metadata.
"""
import json, os, sys
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO, "..", "dataset_pipeline")
UNIFIED = os.path.join(DATA_ROOT, "v2", "unified_pairs.jsonl")
SPLITS = os.path.join(REPO, "data", "splits.json")
PREDS_DIR = os.path.join(REPO, "..", "8_Project_Management", "phase4b_predictions")
OUT = os.path.join(REPO, "..", "8_Project_Management", "per_attack_analysis.json")


def load_test_metadata():
    """Return list of (label, attack_type, domain) for test pairs in order."""
    with open(SPLITS) as f:
        splits = json.load(f)
    test_doms = set(splits["test"])
    meta = []
    with open(UNIFIED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["domain"] in test_doms:
                label = 0 if r.get("label") == "legitimate" else 1
                atk = r.get("attack_type", "unknown")
                # Normalize attack types
                if atk is None:
                    atk = "legitimate_organic"
                meta.append((label, atk, r["domain"]))
    return meta


def load_preds(name):
    """Load saved predictions JSON."""
    path = os.path.join(PREDS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d


def compute_per_group(meta, preds, labels):
    """Compute F1/acc per attack_type and per legitimate/defaced."""
    groups = defaultdict(lambda: {"preds": [], "labels": []})
    for i, (label, atk, dom) in enumerate(meta):
        groups[atk]["preds"].append(preds[i])
        groups[atk]["labels"].append(label)

    # Also group by legit vs defaced
    legit_groups = {"legitimate_all": {"preds": [], "labels": []},
                    "defaced_all": {"preds": [], "labels": []}}
    for i, (label, atk, dom) in enumerate(meta):
        key = "legitimate_all" if label == 0 else "defaced_all"
        legit_groups[key]["preds"].append(preds[i])
        legit_groups[key]["labels"].append(label)

    results = {}
    print("\n═══════════════════════════════════════════════════════════")
    print("  PER-ATTACK-TYPE BREAKDOWN")
    print("═══════════════════════════════════════════════════════════")
    print(f"{'Attack Type':<25} {'N':>5} {'F1(%)':>8} {'Acc(%)':>8} {'FP':>5} {'FN':>5}")
    print("─" * 60)

    for atk in sorted(groups.keys()):
        g = groups[atk]
        n = len(g["labels"])
        f1 = f1_score(g["labels"], g["preds"], zero_division=0) * 100
        acc = accuracy_score(g["labels"], g["preds"]) * 100
        # FP = predicted defaced (1) but actually legitimate (0)
        fp = sum(1 for p, l in zip(g["preds"], g["labels"]) if p == 1 and l == 0)
        # FN = predicted legitimate (0) but actually defaced (1)
        fn = sum(1 for p, l in zip(g["preds"], g["labels"]) if p == 0 and l == 1)
        print(f"{atk:<25} {n:>5} {f1:>8.2f} {acc:>8.2f} {fp:>5} {fn:>5}")
        results[atk] = {"n": n, "f1": f1, "acc": acc, "fp": fp, "fn": fn}

    print("─" * 60)
    # Overall
    overall_f1 = f1_score(labels, preds, zero_division=0) * 100
    overall_acc = accuracy_score(labels, preds) * 100
    print(f"{'OVERALL':<25} {len(labels):>5} {overall_f1:>8.2f} {overall_acc:>8.2f}")

    # FP analysis on legitimate
    print("\n═══════════════════════════════════════════════════════════")
    print("  FALSE-POSITIVE ANALYSIS (legitimate pairs)")
    print("═══════════════════════════════════════════════════════════")
    legit_labels = [l for l in labels if l == 0]
    legit_preds = [p for p, l in zip(preds, labels) if l == 0]
    n_legit = len(legit_labels)
    fp = sum(1 for p in legit_preds if p == 1)
    fp_rate = fp / n_legit * 100 if n_legit > 0 else 0
    tn = n_legit - fp
    print(f"  Legitimate pairs:    {n_legit}")
    print(f"  Correctly rejected:  {tn} (TN)")
    print(f"  False positives:     {fp} (FP)")
    print(f"  FP rate:             {fp_rate:.2f}%")
    results["_fp_analysis"] = {"n_legit": n_legit, "fp": fp, "tn": tn, "fp_rate": fp_rate}

    # FP breakdown by legitimate subtype
    legit_subtypes = defaultdict(lambda: {"total": 0, "fp": 0})
    for i, (label, atk, dom) in enumerate(meta):
        if label == 0:
            legit_subtypes[atk]["total"] += 1
            if preds[i] == 1:
                legit_subtypes[atk]["fp"] += 1
    print(f"\n  FP breakdown by legitimate subtype:")
    print(f"  {'Subtype':<25} {'N':>5} {'FP':>5} {'FP%':>8}")
    print("  " + "─" * 45)
    for st in sorted(legit_subtypes.keys()):
        d = legit_subtypes[st]
        pct = d["fp"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {st:<25} {d['total']:>5} {d['fp']:>5} {pct:>8.1f}")
        results[f"_fp_{st}"] = {"n": d["total"], "fp": d["fp"], "fp_rate": pct}

    # FN analysis on defaced by attack type
    print(f"\n═══════════════════════════════════════════════════════════")
    print("  FALSE-NEGATIVE ANALYSIS (defaced pairs — missed attacks)")
    print("═══════════════════════════════════════════════════════════")
    defaced_subtypes = defaultdict(lambda: {"total": 0, "fn": 0})
    for i, (label, atk, dom) in enumerate(meta):
        if label == 1:
            defaced_subtypes[atk]["total"] += 1
            if preds[i] == 0:
                defaced_subtypes[atk]["fn"] += 1
    print(f"  {'Attack Type':<25} {'N':>5} {'FN':>5} {'Miss%':>8} {'Recall%':>8}")
    print("  " + "─" * 55)
    for st in sorted(defaced_subtypes.keys()):
        d = defaced_subtypes[st]
        miss_pct = d["fn"] / d["total"] * 100 if d["total"] > 0 else 0
        recall = 100 - miss_pct
        print(f"  {st:<25} {d['total']:>5} {d['fn']:>5} {miss_pct:>8.1f} {recall:>8.1f}")
        results[f"_fn_{st}"] = {"n": d["total"], "fn": d["fn"], "miss_rate": miss_pct, "recall": recall}

    return results


def main():
    print("Loading test set metadata...")
    meta = load_test_metadata()
    print(f"Test pairs: {len(meta)}")

    # Attack type distribution
    atk_dist = defaultdict(int)
    legit_count = 0
    for label, atk, dom in meta:
        atk_dist[atk] += 1
        if label == 0:
            legit_count += 1
    print(f"\nAttack type distribution ({len(meta)} test pairs):")
    for atk in sorted(atk_dist.keys()):
        print(f"  {atk:<25} {atk_dist[atk]:>5}")
    print(f"  {'LEGITIMATE TOTAL':<25} {legit_count:>5}")
    print(f"  {'DEFACED TOTAL':<25} {len(meta) - legit_count:>5}")

    all_results = {}
    for pred_name in ["soft_42", "none_42"]:
        data = load_preds(pred_name)
        if data is None:
            print(f"\n⚠ Predictions not found: {pred_name}")
            continue
        print(f"\n{'='*60}")
        print(f"  DI-MMGN ({pred_name}, dropout=0.15, seed=42)")
        print(f"  Overall Test F1 = {data['f1']*100:.2f}%")
        print(f"{'='*60}")
        results = compute_per_group(meta, data["preds"], data["labels"])
        all_results[pred_name] = results

    # Save
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ Saved to {OUT}")


if __name__ == "__main__":
    main()
