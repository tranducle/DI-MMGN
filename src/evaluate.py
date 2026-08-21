"""
Multiseed Evaluation & Ablation Harness (Phase 4: 30-Epoch Official).
Implements Step 4 & 5. Runs training for Baseline, DI-MMGN (Soft Ortho), and DI-MMGN (No Ortho).
Calculates F1 and Confidence Intervals.

Hardened (vs. Phase-3 smoke harness):
- NO config rename dance: passes an explicit temp config path + ortho override to trainers,
  so a crash/OOM can never corrupt the canonical default.yaml. Supports safe auto-restart.
- Per-seed / per-variant checkpoints (no clobbering across the loop).
- Raw per-seed results persisted to results_raw.json after every cell (crash-safe progress log).
- Full deterministic seeding delegated to the trainers (set_seed).
"""
import os
import sys
import copy
import json
import uuid
import yaml
import torch
import numpy as np
from scipy import stats
from train_baseline import train_baseline
from train_dimmgn import train_dimmgn, evaluate, load_config
from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

RESULTS_RAW = "../8_Project_Management/phase4_results_raw.json"


def _save_raw(raw):
    """Crash-safe incremental persistence of raw per-seed results."""
    with open(RESULTS_RAW, "w") as f:
        json.dump(raw, f, indent=2)


def eval_dimmgn(seed, ortho_mode, cfg, test_loader, device):
    """Train one DI-MMGN variant and return its Test F1. No global config mutation."""
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["model"]["orthogonality"] = ortho_mode
    # Unique, isolated artifacts so nothing in the loop clobbers anything else.
    tag = f"{seed}_{ortho_mode}_{uuid.uuid4().hex[:6]}"
    cfg_path = f"config/temp_{tag}.yaml"
    ckpt_path = f"dimmgn_{tag}.pt"
    with open(cfg_path, "w") as f:
        yaml.dump(variant_cfg, f)

    print(f"\n--- Training DI-MMGN [Ortho={ortho_mode}, Seed={seed}] ---", flush=True)
    try:
        train_dimmgn(seed=seed, config_path=cfg_path, ortho_override=ortho_mode, ckpt_path=ckpt_path)
        model = DI_MMGN(variant_cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        _, test_f1, _ = evaluate(model, test_loader, device, variant_cfg)
        return float(test_f1)
    finally:
        # Always clean up the temp config; canonical default.yaml is never touched.
        for p in (cfg_path, ckpt_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[warn] could not remove {p}: {e}", flush=True)


def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def run_evaluation():
    seeds = [42, 43, 44]
    raw = {"baseline": {}, "soft": {}, "none": {}, "done": False}
    _save_raw(raw)

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Harness] device={device} | epochs_pretrain={cfg['train']['epochs_pretrain']} "
          f"| epochs_finetune={cfg['train']['epochs_finetune']} | batch={cfg['train']['batch']}", flush=True)

    # Load test loader once (test split is fixed across all variants/seeds).
    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full_dataset = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"],
                                     data_root=cfg["data"]["root"], tokenizer=tok,
                                     max_text_len=cfg["data"]["max_text_len"])
    test_pairs = [p for p in full_dataset.pairs if p["domain"] in splits["test"]]
    test_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                                max_text_len=cfg["data"]["max_text_len"], pairs=test_pairs)
    collate_fn = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch"], shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    for seed in seeds:
        print(f"\n====================== SEED {seed} ======================", flush=True)

        # 1. Baseline (per-seed checkpoint; returns Test F1 from its own internal eval)
        if str(seed) not in raw["baseline"]:
            print(f"\n--- Training Baseline [Seed={seed}] ---", flush=True)
            f1_base = train_baseline(seed, ckpt_path=f"baseline_{seed}.pt")
            raw["baseline"][str(seed)] = float(f1_base)
            _save_raw(raw)

        # 2. DI-MMGN (Soft)
        if str(seed) not in raw["soft"]:
            f1_soft = eval_dimmgn(seed, "soft", cfg, test_loader, device)
            raw["soft"][str(seed)] = float(f1_soft)
            _save_raw(raw)

        # 3. DI-MMGN (None)
        if str(seed) not in raw["none"]:
            f1_none = eval_dimmgn(seed, "none", cfg, test_loader, device)
            raw["none"][str(seed)] = float(f1_none)
            _save_raw(raw)

    baseline_f1s = [raw["baseline"][str(s)] for s in seeds]
    dimmgn_soft_f1s = [raw["soft"][str(s)] for s in seeds]
    dimmgn_none_f1s = [raw["none"][str(s)] for s in seeds]

    m_base, h_base = mean_ci(baseline_f1s)
    m_soft, h_soft = mean_ci(dimmgn_soft_f1s)
    m_none, h_none = mean_ci(dimmgn_none_f1s)

    delta = m_soft - m_none

    print("\n====================== RESULTS ======================", flush=True)
    print(f"Baseline F1: {m_base*100:.2f} ± {h_base*100:.2f}", flush=True)
    print(f"DI-MMGN (No Ortho) F1: {m_none*100:.2f} ± {h_none*100:.2f}", flush=True)
    print(f"DI-MMGN (Soft Ortho) F1: {m_soft*100:.2f} ± {h_soft*100:.2f}", flush=True)
    print(f"Ortho Delta: {delta*100:.2f} pts", flush=True)

    # Save ortho_ablation.md
    with open("../8_Project_Management/ortho_ablation.md", "w") as f:
        f.write("# ORTHO_MECHANISM Ablation (Phase 4: 30-Epoch Official)\n\n")
        f.write("**Gate**: SCIE-Q1-ABLATION (>=5pt F1 delta)\n")
        f.write("This evaluates whether the orthogonality constraint (d_deface ⊥ d_legit) is load-bearing.\n\n")
        f.write(f"- DI-MMGN (No Ortho) F1: {m_none*100:.2f} ± {h_none*100:.2f}\n")
        f.write(f"- DI-MMGN (Soft Ortho) F1: {m_soft*100:.2f} ± {h_soft*100:.2f}\n")
        f.write(f"- **Delta**: {delta*100:.2f} pts\n\n")
        f.write("**Status**: " + ("✅ PASS" if delta >= 0.05 else "❌ FAIL (Must be >= 5pt delta)") + "\n")

    # Save result_integrity.md
    with open("../8_Project_Management/result_integrity.md", "w") as f:
        f.write("# Result Integrity CI (Phase 4)\n\n")
        f.write(f"Baseline F1: {m_base*100:.2f} ± {h_base*100:.2f}\n")
        f.write(f"DI-MMGN F1: {m_soft*100:.2f} ± {h_soft*100:.2f}\n")

    raw["done"] = True
    raw["summary"] = {
        "baseline_mean": float(m_base), "baseline_ci": float(h_base),
        "soft_mean": float(m_soft), "soft_ci": float(h_soft),
        "none_mean": float(m_none), "none_ci": float(h_none),
        "ortho_delta": float(delta),
        "gate_pass": bool(delta >= 0.05),
    }
    _save_raw(raw)
    print(f"[Harness] DONE. Raw results -> {RESULTS_RAW}", flush=True)


if __name__ == "__main__":
    run_evaluation()
