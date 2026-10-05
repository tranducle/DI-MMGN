# RTFDD 2026 Reproduction Assessment

Paper: **RTFDD: A Robust Two-Stage Framework For Web Defacement Detection**

## Finding
The deep-learning stage is materially the same three-modality architecture family as DefacementFusion:
HTML structural encoder + BiLSTM text + EfficientNetB0 image -> Transformer fusion -> MLP.

RTFDD adds a first-stage signature detector with a paper-specific database of 20,255 signatures extracted from the authors' 39,100 defaced pages.

## Reproduction blocker
The 20,255-signature corpus is not part of the available paper package and the exact signature extraction/matching representation is not specified enough to reconstruct the authors' database exactly.

## Decision
- Cite RTFDD as a latest reported system.
- Do **not** treat a locally invented signature corpus as an exact RTFDD reproduction.
- Primary task-specific SOTA reimplementation: DefacementFusion 2025.
- Secondary high-fidelity comparator: BiLSTM+EfficientNet 2021.
