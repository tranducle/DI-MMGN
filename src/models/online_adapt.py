"""Online drift adaptation — paper_outline.md §4.3, §4.4.

O(n) per confirmed-legitimate update: one contrastive gradient step on d_legit,
then re-orthogonalize d_deface. This is the mechanism that delivers Post-Redesign
F1 Retention (paper §5.4) without a full retrain.
"""
import torch


@torch.no_grad()
def online_adapt(ortho, confirmed_legit_delta, lr=1e-2):
    """One online adaptation step. Returns timing-relevant O(d) op count.

    ortho: OrthogonalDecomposition
    confirmed_legit_delta: [d] or [B,d] — a confirmed-legitimate change vector.
    """
    d_legit = ortho.d_legit.detach().clone()
    d_deface = ortho.d_deface.detach().clone()
    # gradient of -cos(delta, d_legit) w.r.t. d_legit (pull delta toward d_legit)
    if confirmed_legit_delta.dim() == 1:
        confirmed_legit_delta = confirmed_legit_delta.unsqueeze(0)
    proj = confirmed_legit_delta @ d_legit            # [B]
    # grad ∝ -mean(delta) scaled; take a normalized step along the mean delta direction
    g = confirmed_legit_delta.mean(0)
    g = g / g.norm().clamp(min=1e-8)
    d_legit = d_legit + lr * g
    d_legit = d_legit / d_legit.norm().clamp(min=1e-8)
    # re-orthogonalize d_deface
    d_deface = d_deface - (d_deface @ d_legit) * d_legit
    d_deface = d_deface / d_deface.norm().clamp(min=1e-8)
    ortho.d_legit.copy_(d_legit)
    ortho.d_deface.copy_(d_deface)
    return ortho
