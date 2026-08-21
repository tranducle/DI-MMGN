"""DI-MMGN losses. Paper ref: paper_outline.md §4.2, §6.4 (τ sweep), §7.2 (soft ortho).

Composite = w_legit * ContrastiveLegitLoss + w_deface * DefaceAnchorLoss
            + w_ortho * OrthoPenalty (soft mode only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLegitLoss(nn.Module):
    """NT-Xent-style: pull legitimate Δ_t toward +d_legit, push defaced away.

    legit_proj(delta): signed projection of Δ_t onto d_legit.
    Label y: 1 = legitimate (should align with +d_legit), 0 = defaced.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, legit_proj, y):
        # maximize legit_proj for legitimate, minimize (push negative) for defaced
        target = y.float() * 2 - 1        # legitimate -> +1, defaced -> -1
        # cosine-like alignment loss on signed projection
        return -F.logsigmoid(target * legit_proj / self.tau).mean()


class DefaceAnchorLoss(nn.Module):
    """Few-shot anchor: maximize |proj_{d_deface}(Δ_t)| for defaced pairs."""
    def forward(self, score, y):
        # score is already |proj|; push defaced scores up, legitimate down
        return F.binary_cross_entropy_with_logits(score, y.float())


class OrthoPenalty(nn.Module):
    """Soft orthogonality penalty λ‖d_legit·d_deface‖². Paper §7.2."""
    def forward(self, residual, lam=1.0):
        return lam * residual.pow(2)


def composite_loss(legit_proj, score, y, ortho_residual, cfg):
    w = cfg["loss"]
    cl = ContrastiveLegitLoss(w["temperature"])
    da = DefaceAnchorLoss()
    op = OrthoPenalty()
    L = (w["w_legit"] * cl(legit_proj, y)
         + w["w_deface"] * da(score, y))
    if cfg["model"]["orthogonality"] == "soft":
        L = L + w["w_ortho"] * op(ortho_residual, 1.0)
    return L
