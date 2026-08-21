"""DI-MMGN top-level module. Paper ref: paper_outline.md §4.1–§4.3.

forward(batch) -> dict with delta, score, e_t1, e_t2, legit_proj, d_legit, d_deface.
Losses are computed OUTSIDE forward so the same forward serves train and eval.
"""
import torch
import torch.nn as nn
from .encoders import MultiModalEncoder
from .orthogonal import OrthogonalDecomposition


class DI_MMGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(cfg["model"]["d"], mode=cfg["model"]["orthogonality"])

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)              # [B,d]

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])   # [B,d]
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])   # [B,d]
        delta = e2 - e1                                          # [B,d]  paper §4.1
        legit_proj = self.ortho.legit_proj(delta)               # [B]
        score = self.ortho.score(delta)                          # [B]
        return {
            "delta": delta,
            "score": score,
            "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
        }

    @torch.no_grad()
    def after_step(self):
        """Call after each optimizer step to keep directions unit / orthogonal."""
        self.ortho.renormalize()


class DI_MMGN_Concat(nn.Module):
    """Fixed variant: detection uses [E(S_t); Δ_t] — both absolute content AND change.
    
    Root cause fix for the catastrophic FP on legitimate CMS mutations:
    the original DI_MMGN used only Δ_t (change), losing absolute content
    information needed to recognise legitimate pages. This variant
    concatenates E(S_t) with Δ_t → the classifier can learn:
    'page content looks legitimate (from E(S_t)) even though it changed.'
    
    The orthogonal decomposition (d_legit, d_deface) is retained for the
    contrastive pretraining of d_legit, but the detection score comes from
    a learned classifier on the 2d-dimensional combined representation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["model"]["d"]
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(d, mode=cfg["model"]["orthogonality"])
        # Classifier on [E(S_t); Δ_t] → score
        p = float(cfg["model"].get("dropout", 0.0))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, 1),
        )

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])   # [B,d]
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])   # [B,d]
        delta = e2 - e1                                          # [B,d]
        legit_proj = self.ortho.legit_proj(delta)               # [B] — contrastive target
        combined = torch.cat([e2, delta], dim=-1)               # [B,2d] — absolute + change
        score = self.classifier(combined).squeeze(-1)           # [B] — learned detection
        return {
            "delta": delta,
            "score": score,
            "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
        }

    @torch.no_grad()
    def after_step(self):
        self.ortho.renormalize()


# ═══ FUSION ABLATION VARIANTS (Gap 3: architecture depth) ═══

class DI_MMGN_CrossAttn(nn.Module):
    """Fusion variant: cross-attention between E(S_t) and Δ_t.
    
    Instead of naive concatenation, delta attends to the snapshot embedding:
    the model learns WHICH aspects of absolute content to focus on given the
    observed change. More expressive than concat; tests whether the simple
    concat is sufficient or attention helps.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["model"]["d"]
        p = float(cfg["model"].get("dropout", 0.0))
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(d, mode=cfg["model"]["orthogonality"])
        self.cross_attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True, dropout=p)
        self.classifier = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, 1))

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])
        delta = e2 - e1
        legit_proj = self.ortho.legit_proj(delta)
        # Cross-attention: delta (query) attends to e2 (key/value)
        attn_out, _ = self.cross_attn(delta.unsqueeze(1), e2.unsqueeze(1), e2.unsqueeze(1))
        combined = torch.cat([e2, attn_out.squeeze(1)], dim=-1)
        score = self.classifier(combined).squeeze(-1)
        return {
            "delta": delta, "score": score, "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
        }

    @torch.no_grad()
    def after_step(self):
        self.ortho.renormalize()


class DI_MMGN_Gated(nn.Module):
    """Fusion variant: learnable gate between E(S_t) and Δ_t.
    
    A sigmoid gate learns per-dimension how much to rely on absolute content
    vs temporal change: gated = g * e2 + (1-g) * delta. Interpretable —
    gate values show which embedding dimensions are content-driven vs change-driven.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["model"]["d"]
        p = float(cfg["model"].get("dropout", 0.0))
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(d, mode=cfg["model"]["orthogonality"])
        self.gate_proj = nn.Linear(d, d)
        self.classifier = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, 1))

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])
        delta = e2 - e1
        legit_proj = self.ortho.legit_proj(delta)
        # Learnable gate: g in [0,1] per dimension
        g = torch.sigmoid(self.gate_proj(e2))       # [B, d]
        gated = g * e2 + (1 - g) * delta             # [B, d]
        score = self.classifier(gated).squeeze(-1)
        return {
            "delta": delta, "score": score, "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
        }

    @torch.no_grad()
    def after_step(self):
        self.ortho.renormalize()


# ═══ CONFOUND CONTROL + INFERENCE-PATH ADAPTER (reviewer-driven experiments) ═══

class DI_MMGN_ConcatZeroDelta(nn.Module):
    """Parameter-matched confound control for the +change-vector claim.

    Identical to DI_MMGN_Concat (same encoder, same ortho, SAME classifier with
    the exact same parameter count) except the change-vector half of the input is
    replaced by zeros: combined = [E(S_t); 0]. Any F1 difference vs CONCAT is
    therefore attributable to the Δ_t signal itself, not to the extra 2d->d MLP
    capacity that concatenation introduces. Paper §6 (parameter-matched control).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["model"]["d"]
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(d, mode=cfg["model"]["orthogonality"])
        p = float(cfg["model"].get("dropout", 0.0))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, 1),
        )

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])
        delta = e2 - e1
        legit_proj = self.ortho.legit_proj(delta)
        zero_delta = torch.zeros_like(delta)                # confound control: no Δ signal
        combined = torch.cat([e2, zero_delta], dim=-1)       # [B,2d], same classifier params
        score = self.classifier(combined).squeeze(-1)
        return {
            "delta": delta, "score": score, "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
        }

    @torch.no_grad()
    def after_step(self):
        self.ortho.renormalize()


class DI_MMGN_ConcatAdapter(nn.Module):
    """CONCAT with a learnable inference-path bias (constructive drift-adaptation).

    Identical to DI_MMGN_Concat but adds a per-dimension bias b to the snapshot
    embedding before concatenation: combined = [E(S_t) + b; Δ_t]. Unlike d_legit
    (which is NOT in the CONCAT inference path), b IS in the path, so updating b
    at deployment changes the score — making online adaptation empirically
    testable. b is initialised to zero, so an unadapted model is bit-for-bit
    equivalent to DI_MMGN_Concat. Paper §6 (inference-path adapter / RQ3 positive).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["model"]["d"]
        self.encoder = MultiModalEncoder(cfg)
        self.ortho = OrthogonalDecomposition(d, mode=cfg["model"]["orthogonality"])
        p = float(cfg["model"].get("dropout", 0.0))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, 1),
        )
        self.adapt_bias = nn.Parameter(torch.zeros(d))       # inference-path adapter

    def encode_snapshot(self, snap, mask):
        return self.encoder(snap, mask)

    def forward(self, batch):
        e1 = self.encode_snapshot(batch["t1"], batch["mask"])
        e2 = self.encode_snapshot(batch["t2"], batch["mask"])
        delta = e2 - e1
        legit_proj = self.ortho.legit_proj(delta)
        combined = torch.cat([e2 + self.adapt_bias, delta], dim=-1)   # b is in the path
        score = self.classifier(combined).squeeze(-1)
        return {
            "delta": delta, "score": score, "legit_proj": legit_proj,
            "e_t1": e1, "e_t2": e2,
            "d_legit": self.ortho.d_legit.detach(),
            "d_deface": self.ortho.d_deface.detach(),
            "ortho_residual": self.ortho.orthogonality_residual(),
            "adapt_bias": self.adapt_bias,
        }

    @torch.no_grad()
    def after_step(self):
        self.ortho.renormalize()


# Factory for fusion-strategy ablation
FUSION_REGISTRY = {
    "concat": DI_MMGN_Concat,
    "crossattn": DI_MMGN_CrossAttn,
    "gated": DI_MMGN_Gated,
}
