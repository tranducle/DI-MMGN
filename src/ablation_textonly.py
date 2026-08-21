"""
Phase-4d: Single-modal text-only change-vector ablation.
Addresses Limitation (iii) in the manuscript: decomposes the +7.4 pt gain into
  (a) temporal change-vector on text alone, vs
  (b) full multi-modal fusion.
If text-only Δ_t already reaches ~full DI-MMGN, the gain is mostly the temporal
reframing + text; if it is much lower, multi-modal fusion is load-bearing.

Implementation: a TextOnlyCollate wrapper forces mask=[True,False,False,False]
on every batch, so the SAME DI_MMGN model + losses train/eval on text alone.
Reuses DimmgnPairDataset, DI_MMGN, composite_loss, ContrastiveLegitLoss.
3 seeds, crash-safe incremental -> phase4d_textonly_raw.json.
"""
import os, sys, json, random, uuid
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

from data.dimmgn_dataset import DimmgnPairDataset, DimmgnCollate
from models.dimmgn import DI_MMGN
from models.losses import composite_loss, ContrastiveLegitLoss

RESULTS = "../8_Project_Management/phase4d_textonly_raw.json"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); os.environ["PYTHONHASHSEED"] = str(s)


def _save(raw):
    with open(RESULTS, "w") as f:
        json.dump(raw, f, indent=2)


class TextOnlyCollate:
    """Wrap DimmgnCollate; force mask to text-only ([T,F,F,F]) for every pair."""
    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        b = self.base(batch)
        bm = b["mask"]
        B = bm.shape[0]
        m = torch.zeros((B, 4), dtype=bm.dtype)
        m[:, 0] = True          # text only
        b["mask"] = m
        return b


def evaluate(model, loader, device, cfg):
    model.eval()
    preds, labels = [], []
    tot = 0.0
    with torch.no_grad():
        for batch in loader:
            labels_t = batch["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["t2"].items()}
            mask = batch["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            preds.extend((torch.sigmoid(out["score"]) > 0.5).long().cpu().numpy().tolist())
            labels.extend(labels_t.cpu().numpy().tolist())
            res = out.get("ortho_residual", torch.tensor(0.0, device=device))
            tot += composite_loss(out["legit_proj"], out["score"], labels_t, res, cfg).item()
    return tot / len(loader), f1_score(labels, preds, zero_division=0), accuracy_score(labels, preds)


def train_textonly(seed, cfg, loaders, device):
    set_seed(seed)
    pretrain_loader, train_loader, val_loader, test_loader = loaders
    accum = cfg["train"]["accum"]
    model = DI_MMGN(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    legit_loss = ContrastiveLegitLoss(temperature=cfg["loss"]["temperature"])
    ckpt = f"textonly_{seed}_{uuid.uuid4().hex[:6]}.pt"
    print(f"\n--- [Phase4d] TEXT-ONLY Δ_t seed={seed} | pretrain {cfg['train']['epochs_pretrain']} + finetune {cfg['train']['epochs_finetune']} ---", flush=True)

    # Phase A: contrastive pretrain (text-only mask forced by collate)
    for ep in range(cfg["train"]["epochs_pretrain"]):
        model.train(); opt.zero_grad(); tl = 0.0
        for i, b in enumerate(pretrain_loader):
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            loss = legit_loss(out["legit_proj"], y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(pretrain_loader):
                opt.step(); model.after_step(); opt.zero_grad()
            tl += loss.item() * accum
        print(f"  text/{seed} pretrain ep {ep+1}/{cfg['train']['epochs_pretrain']} loss {tl/len(pretrain_loader):.4f}", flush=True)

    # Phase B: finetune (composite, text-only)
    best = -1.0
    for ep in range(cfg["train"]["epochs_finetune"]):
        model.train(); opt.zero_grad(); tl = 0.0
        for i, b in enumerate(train_loader):
            y = b["label"].to(device)
            t1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t1"].items()}
            t2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b["t2"].items()}
            mask = b["mask"].to(device)
            out = model({"t1": t1, "t2": t2, "mask": mask})
            res = out.get("ortho_residual", torch.tensor(0.0, device=device))
            loss = composite_loss(out["legit_proj"], out["score"], y, res, cfg) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                opt.step(); model.after_step(); opt.zero_grad()
            tl += loss.item() * accum
            if (i + 1) % 100 == 0:
                print(f"    text/{seed} batch {i+1}/{len(train_loader)} loss {loss.item()*accum:.4f}", flush=True)
        vl, vf, _ = evaluate(model, val_loader, device, cfg)
        print(f"  text/{seed} finetune ep {ep+1}/{cfg['train']['epochs_finetune']} | train {tl/len(train_loader):.4f} | valF1 {vf:.4f}", flush=True)
        if vf > best:
            best = vf; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    _, tf1, tacc = evaluate(model, test_loader, device, cfg)
    try:
        os.remove(ckpt)
    except OSError:
        pass
    print(f"[Phase4d] TEXT-ONLY/{seed} Test F1 = {tf1:.4f} | Acc {tacc:.4f}", flush=True)
    return float(tf1)


def main():
    seeds = [42, 43, 44]
    raw = {"textonly": {}, "done": False}
    if os.path.exists(RESULTS):
        try:
            raw = json.load(open(RESULTS))
        except Exception:
            pass
    raw.setdefault("textonly", {})
    _save(raw)

    cfg = yaml.safe_load(open("config/default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase4d] device={device} | text-only Δ_t ablation | seeds={seeds}", flush=True)

    with open("data/splits.json") as f:
        splits = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    full = DimmgnPairDataset(unified_pairs_path=cfg["data"]["unified_pairs"], data_root=cfg["data"]["root"],
                             tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    def pairs_of(sp):
        return [p for p in full.pairs if p["domain"] in splits[sp]]
    base_collate = DimmgnCollate(tokenizer=tok, max_text_len=cfg["data"]["max_text_len"])
    collate = TextOnlyCollate(base_collate)   # <- forces text-only mask everywhere
    def loader(sp, shuffle):
        ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                               max_text_len=cfg["data"]["max_text_len"], pairs=pairs_of(sp))
        return DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=shuffle,
                          collate_fn=collate, num_workers=0)
    pretrain_legit = [p for p in pairs_of("pretrain") if p.get("label") == "legitimate"]
    pretrain_ds = DimmgnPairDataset(data_root=cfg["data"]["root"], tokenizer=tok,
                                    max_text_len=cfg["data"]["max_text_len"], pairs=pretrain_legit)
    loaders = (
        DataLoader(pretrain_ds, batch_size=cfg["train"]["batch"], shuffle=True, collate_fn=collate, num_workers=0),
        loader("pretrain", True), loader("val", False), loader("test", False),
    )

    for seed in seeds:
        if str(seed) in raw["textonly"]:
            print(f"[Phase4d] skip text/{seed} (done)", flush=True)
            continue
        f1 = train_textonly(seed, cfg, loaders, device)
        raw["textonly"][str(seed)] = f1
        _save(raw)

    from scipy import stats
    vals = [raw["textonly"][str(s)] for s in seeds]
    a = np.array(vals, float); m, se = a.mean(), stats.sem(a)
    ci = float(se * stats.t.ppf(0.975, len(a) - 1)) if len(a) > 1 else 0.0
    raw["summary"] = {"textonly_mean": float(m), "textonly_ci95": ci}
    raw["done"] = True
    _save(raw)
    print(f"\n[Phase4d] DONE  TEXT-ONLY Δ_t = {m*100:.2f} ± {ci*100:.2f}  -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
