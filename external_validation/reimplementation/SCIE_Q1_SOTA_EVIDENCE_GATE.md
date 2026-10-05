# SCIE Q1 Evidence Coverage Gate — SOTA Reimplementation Extension

## Verdict
**PASS WITH CLAIM NARROWING**

Official comparator training on LWDED-v4 may proceed. Results may support only bounded claims about task-specific comparators reimplemented under the same frozen LWDED-v4 family-blocked protocol. They may not be used to claim external/deployment generalization.

## Claim-to-Experiment Matrix

| Claim | Evidence required | Planned/available experiment | Coverage | Missing evidence | Claim boundary |
|---|---|---|---|---|---|
| DI-MMGN improves over task-specific static defacement methods under the same benchmark | At least one recent task-specific comparator, identical split/test cohort, multiseed evaluation | DefacementFusion 2025 paper-faithful reimplementation + BiLSTM/EfficientNet 2021 high-fidelity reimplementation on frozen LWDED-v4 | Covered after official runs | Results pending | “On LWDED-v4, compared with our paper-faithful/high-fidelity reimplementations…” |
| The gain is not due only to weak/generic baselines | Strong multimodal defacement comparator | DefacementFusion: HTML + text + screenshot + Transformer fusion | Covered after official runs | Exact authors' source/checkpoint unavailable | Do not call implementation official or exact |
| DI-MMGN generalizes beyond the author-built benchmark | Independent real malicious cohort | DMoS public real-defacement HTML release downloaded and inspected | Partial/planned | No benign cohort, paired clean predecessor, screenshot, or HTTP provenance in public package | No external-generalization claim until a defensible DMoS protocol is completed |

## Coverage Audit

| Dimension | Status | Reviewer risk | Required action |
|---|---|---|---|
| Dataset/split discipline | PASS | Low | Reuse frozen LWDED-v4 family split and exact pair IDs |
| Comparator breadth | PASS-planned | Medium until results exist | Run 2021 + 2025 task-specific comparators |
| Multiseed uncertainty | PASS-planned | Medium until results exist | Seeds 42/43/44; same test cohort |
| Mechanism controls | PASS | Low | Existing ZERO-Delta and fusion ablations |
| External validation | PARTIAL | High for broad generalization | Build a separate DMoS external protocol before any external claim |
| Reproducibility | PASS-planned | Medium | Preserve SPEC/config/code/checkpoints/predictions/logs |
| Exact source fidelity | NOT AVAILABLE | Medium | Label DefacementFusion “paper-faithful reimplementation”; record every assumption |

## Tiered Decisions
- **Tier 1 must-run now:** BiLSTM+EfficientNet 2021 and DefacementFusion 2025 on frozen LWDED-v4; 3 seeds; aligned predictions; integrity gate.
- **Tier 1 separate external track:** DMoS independent real-defacement validation protocol.
- **Tier 2:** BERT+BiLSTM 2025 text-only recent comparator if compute/time permits.
- **Tier 3:** RTFDD exact reproduction only if its 20,255-signature corpus/source becomes available.

## Required Scientific Gates
| Stage | Gate | Pass criteria |
|---|---|---|
| Static-current adapter | S0 | 4,678 rows; zero missing HTML/screenshots; frozen split/labels match LWDED-v4 |
| 2021 implementation smoke | S1 | Real batch forward/backward; finite loss/gradients; soft-vote output |
| DefacementFusion smoke | S2 | HTML/Text/Image/Fusion/MLP all activate with finite gradients |
| Mini-training sanity | S3 | Validation predictions non-degenerate; loss finite; artifacts saved |
| Official comparator results | S4 | 3/3 seeds per comparator; 936 aligned test IDs; metrics recomputable exactly |
| Result-to-claim | G8 extension | Only bounded same-benchmark SOTA wording permitted |

## Launch Decision
**Official comparator training may start only after S3 mini-training sanity passes.**
