"""End-to-end smoke test (Phase 2 acceptance §5.8 + §4.7).

Loads a tiny batch of REAL full-modality pairs from unified_pairs.jsonl, runs the
full DI_MMGN forward (4-modality encoder -> Δ_t -> orthogonal score -> loss), and
asserts: shapes correct, loss finite, orthogonality invariant holds.
Labelled SMOKE per Rules.md §5 (may precede scie-q1-evidence-coverage-gate).
"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # Papers/TRANG-PAPER-1
sys.path.insert(0, ROOT)

from src.data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from src.models.dimmgn import DI_MMGN
from src.models.losses import composite_loss

UNIFIED = os.path.join(ROOT, "dataset_pipeline", "v2", "unified_pairs.jsonl")
DATA_ROOT = os.path.join(ROOT, "dataset_pipeline")

def pick_full_modality(n=6):
    out = []
    with open(UNIFIED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("delta_dom_graph") or {}; v = r.get("delta_visual") or {}; h = r.get("delta_http") or {}
            if d.get("t1") and d.get("t2") and v.get("t1") and v.get("t2") and h.get("t1") and h.get("t2"):
                out.append(r)
                if len(out) >= n: break
    return out

def main():
    print("[smoke] SMOKE TEST — labelled smoke per Rules.md §5")
    pairs = pick_full_modality(6)
    assert len(pairs) >= 4, f"need >=4 full-modality pairs, got {len(pairs)}"
    print(f"[smoke] selected {len(pairs)} full-modality pairs: "
          + str([(p['attack_type'], p['label']) for p in pairs]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    ds = DimmgnPairDataset(pairs=pairs, data_root=DATA_ROOT, tokenizer=tok)
    collate = DimmgnCollate(tok)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    batch = next(iter(dl))
    B = batch["mask"].shape[0]
    print(f"[smoke] batch loaded: B={B} mask={batch['mask'].tolist()}")

    cfg = {"model": {"d": 128, "text_backbone": "sentence-transformers/all-MiniLM-L6-v2",
                     "text_frozen": True, "dom_layers": 2, "dom_hidden": 64,
                     "visual_in": 512, "http_in": 32, "orthogonality": "soft"},
           "loss": {"w_legit": 1.0, "w_deface": 1.0, "w_ortho": 1.0, "temperature": 0.1}}
    model = DI_MMGN(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    L = composite_loss(out["legit_proj"], out["score"], batch["label"],
                       model.ortho.orthogonality_residual(), cfg)
    print(f"[smoke] forward OK: delta{tuple(out['delta'].shape)} score{tuple(out['score'].shape)} "
          f"loss={float(L):.4f}")
    print(f"[smoke] d_legit norm={model.ortho.d_legit.norm().item():.3f} "
          f"d_deface norm={model.ortho.d_deface.norm().item():.3f} "
          f"ortho_residual={model.ortho.orthogonality_residual().item():.4e}")

    # assertions
    checks = {
        "delta shape [B,d]": out["delta"].shape == (B, cfg["model"]["d"]),
        "score shape [B]": out["score"].shape == (B,),
        "loss finite": torch.isfinite(L).item(),
        "d_legit unit": abs(model.ortho.d_legit.norm().item() - 1.0) < 1e-2,
        "d_deface unit": abs(model.ortho.d_deface.norm().item() - 1.0) < 1e-2,
    }
    print("[smoke] CHECKS:", checks)
    ok = all(checks.values())
    # one optimizer step to confirm backward works
    model.train()
    out = model(batch)
    L = composite_loss(out["legit_proj"], out["score"], batch["label"],
                       model.ortho.orthogonality_residual(), cfg)
    L.backward()
    model.after_step()
    res_after = model.ortho.orthogonality_residual().item()
    print(f"[smoke] backward OK; ortho_residual after step={res_after:.4e}")
    checks["backward + renorm"] = torch.isfinite(L).item() and res_after < 1.0
    ok = ok and checks["backward + renorm"]
    print(f"\n=== SMOKE: {'PASS' if ok else 'FAIL'} ===")
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
