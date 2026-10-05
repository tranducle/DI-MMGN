# Ablation, Sensitivity, and Robustness Plan

## Reviewer-oriented assessment
The current paper already contains the ablations that directly support its **actual novelty claim**. Adding many new architecture ablations would create scope without closing the main reviewer risk.

## Existing ablations — do not rerun
| Experiment | What it tests | Status | Priority |
|---|---|---|---|
| ZERO-Delta matched control | incremental value of direct temporal input vs same capacity/training | complete | must retain |
| CONCAT vs gated vs cross-attention | fusion design choice | complete | must retain |
| orthogonality on/off | whether auxiliary orthogonality is load-bearing | complete | must retain |
| attack-family / benign-source subgroup analysis | failure boundary | complete | must retain |

## New must-run robustness control if external validation proceeds
| Name | What it tests | Expected if method is robust | Priority |
|---|---|---|---|
| Frozen no-HTTP inference mask | external DMoS lacks HTTP; separates modality loss from domain shift | internal recall/F1 decreases modestly rather than collapses | 1 |
| Frozen text+DOM mask, conditional | fallback if archived screenshot coverage <95% | internal performance remains interpretable | 1 if visual unavailable |

These are **robustness/missing-modality controls**, not component-contribution ablations, because the model is not retrained.

## Optional component ablations
| Name | What it tests | Priority |
|---|---|---|
| Retrain without HTTP | marginal value of HTTP modality | 3 |
| Retrain without visual | marginal value of visual modality | 3 |
| Retrain without DOM | marginal value of DOM modality | 3 |
| Retrain text-only temporal | whether multimodality adds beyond temporal text | 3 |

Run these only if the revised manuscript explicitly claims each modality contributes. The current contribution is the state-plus-change representation, so full leave-one-modality-out retraining is not required to close the external-validity gap.

## Sensitivity analysis
### Must freeze
- classification threshold = 0.5 for external evaluation;
- archival search window and max captures;
- clean-screen criteria;
- visual-availability cutoff (95%);
- minimum cohort size and host-count gates.

### Avoid
- choosing a threshold on DMoS test labels;
- relaxing clean-screen rules after seeing recovered counts;
- reporting the best of multiple external modality masks.

## Failure-mode analyses
1. Per-host external recall.
2. Recall vs predecessor age/gap.
3. Recall vs current-page keyword intensity.
4. Recall on subtle/low-keyword real defacements where available.
5. Missing-modality gap: full internal vs no-HTTP/text+DOM internal.
6. External-vs-internal recall gap with host-aware uncertainty.

## Unnecessary ablations
- larger fusion hyperparameter sweeps: not central after CONCAT/gated/cross-attention evidence.
- more orthogonality weights: current negative result already narrows the claim.
- more GNN architectures: GIN/GAT/GraphSAGE plus task-specific SOTA now cover the baseline question.
- retraining on DMoS: would destroy the clean external-transfer claim.
