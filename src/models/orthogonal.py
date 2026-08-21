"""Orthogonal Direction Decomposition — the core DI-MMGN contribution.

Paper ref: paper_outline.md §4.2.
  - d_legit: learned via self-supervised contrastive on benign temporal pairs.
  - d_deface: constrained strictly orthogonal to d_legit, anchored by few-shot
    defacements. Detection score = |proj_{d_deface}(Δ_t)|.
  - Soft mode (default): add penalty λ‖d_legit·d_deface‖² (paper §7.2).
  - Hard mode: project d_deface onto the orthogonal complement after each step.

These are the two learnable unit directions in R^d. Paper §6.3 ablation toggles
the orthogonality constraint to prove it is load-bearing.
"""
import torch
import torch.nn as nn


class OrthogonalDecomposition(nn.Module):
    def __init__(self, d=256, mode="soft"):
        super().__init__()
        self.d = d
        self.mode = mode  # "soft" | "hard"
        # random unit init
        v = torch.randn(2, d)
        v = v / v.norm(dim=1, keepdim=True)
        self.d_legit = nn.Parameter(v[0])
        self.d_deface = nn.Parameter(v[1])

    @torch.no_grad()
    def renormalize(self):
        """Keep both directions on the unit sphere; in hard mode also re-orthogonalize."""
        self.d_legit.div_(self.d_legit.norm().clamp(min=1e-8))
        if self.mode == "hard":
            self.d_deface.sub_((self.d_deface @ self.d_legit) * self.d_legit)
        self.d_deface.div_(self.d_deface.norm().clamp(min=1e-8))

    def orthogonality_residual(self):
        """|d_legit . d_deface| — should be ~0 (hard) or shrinking (soft)."""
        return (self.d_legit @ self.d_deface).abs()

    def score(self, delta):
        """Detection score = |proj_{d_deface}(Δ_t)|. Paper §4.2."""
        # delta: [B,d]
        return (delta @ self.d_deface).abs()      # [B]

    def legit_proj(self, delta):
        """Signed projection onto d_legit — used by the contrastive loss."""
        return delta @ self.d_legit                # [B]
