# Empirical Standards Design Plan — DI-MMGN External Validation Rescue v2

Date: 2026-10-02

## Chosen standard stack
- Primary standard: **Engineering research** — DI-MMGN is a novel technical artifact/method whose claims depend on empirical evaluation.
- Additional evaluation standard: **Data science / benchmarking** — predictive detection performance, frozen train/validation/test protocols, fair baselines, repeated seeds, and uncertainty matter directly.
- Supplements: **sampling**, **open science**, **ethics (secondary data)**, **ethics (engineering research)**, and **information visualization**.
- IRR/IRA supplement becomes active only if manual clean/defaced adjudication is used in protocol v2.
- Human-participant experiment standard is not applicable.

## Evidence already strong enough to reuse
1. LWDED-v4 family-blocked internal evaluation: 4,678 pairs; 936 held-out test pairs; 36 registrable-domain families; three official seeds.
2. Internal mechanism evidence: parameter-matched ZERO-Delta control and paired family bootstrap.
3. Fusion/orthogonality boundary evidence: CONCAT, gated, cross-attention, soft orthogonality.
4. Existing static baselines: GraphSAGE, GAT, GIN, TextBiLSTM, Snapshot-MM.
5. Recent task-specific reimplementations completed under the frozen LWDED-v4 split:
   - BiLSTM+EfficientNet 2021: 0.79341 +/- 0.00509 Macro-F1.
   - DefacementFusion 2025: 0.78050 +/- 0.00805 Macro-F1.
   - DI-MMGN CONCAT: 0.87480 +/- 0.00389 Macro-F1.
   - S4 integrity and S5 paired comparison gates PASS.
6. DMoS v1 acquisition is a valid **negative scientific result**, not disposable work:
   - 498 usable real-defaced records.
   - 114/498 had pre-collection captures across 15 hosts.
   - 449 archived captures fetched and screened with zero final fetch errors.
   - 52 clean captures.
   - only 9 positive temporal pairs across 9 hosts and 14 benign temporal pairs across 12 hosts.
   - G1C verdict: FAIL_REDESIGN.

## Essential evidence still missing
- Independent real-malicious temporal evidence large enough to support an external-generalization claim.
- A predeclared protocol for any redesigned archival recovery, frozen before official evaluation.
- If external evaluation lacks HTTP and/or visual modalities, an internal modality-matched control so domain shift is not confounded with modality loss.
- If manual archival labels are introduced, explicit labeling rules, blinded review where feasible, agreement reporting, and adjudication.
- A reproducibility record for the redesigned acquisition: query method, URL variants, archival window, capture-selection rule, hashes, exclusions, and failure reasons.

## Claim-to-evidence boundaries
| Claim | Current permission |
|---|---|
| State-plus-change representation improves LWDED-v4 performance | Supported internally |
| Direct classifier access to Delta adds value vs parameter-matched ZERO-Delta | Supported internally |
| DI-MMGN outperforms the two task-specific reimplementations on LWDED-v4 | Supported within benchmark only |
| Frozen DI-MMGN detects independent real defacement incidents | **Not yet supported** |
| DI-MMGN generalizes to deployment conditions / diverse real incidents | **Not supported** |
| Full external binary superiority (recall + specificity + Macro-F1) | **Not supported** unless a full same-source external cohort passes |

## Reviewer risks to pre-empt
| Risk | Why reviewers care | Pre-emption |
|---|---|---|
| Post-hoc lowering of G1C threshold | Would convert a failed gate into cherry-picked evidence | Preserve v1 FAIL_REDESIGN permanently; define v2 before running |
| Contaminated historical “clean” snapshots | Invalidates temporal labels | Strict clean rules + optional manual adjudication + hashes |
| Source/domain imbalance | 99 DMoS captured pages are concentrated in scripts.mit.edu | Host-level caps/stratification in reporting and host-cluster uncertainty |
| Missing HTTP on DMoS | External drop could be modality loss, not domain shift | Internal no-HTTP matched control using frozen checkpoints |
| Tiny positive-only cohort | Recall estimate may be unstable | Minimum n/host gate; CI and per-host reporting; no binary metrics without negatives |
| External test tuning | Inflates generalization evidence | threshold=0.5 frozen; no DMoS retraining/calibration |
| Dataset access ambiguity | Reviewer cannot reproduce | independent source verification before claiming second dataset |

## Submission readiness checklist
- [x] Internal data/split/result gates passed.
- [x] Strong within-benchmark task-specific comparison added.
- [x] DMoS v1 failure preserved.
- [ ] External protocol v2 frozen and hashed before full acquisition.
- [ ] External attack cohort >=50 pairs across >=5 hosts (minimum) OR claim remains benchmark-bounded.
- [ ] Strong target: >=100 attack pairs across >=8 hosts.
- [ ] Internal modality-matched control reported.
- [ ] External CIs use host-aware resampling; per-host results reported.
- [ ] No full binary external metrics unless same-source benign cohort passes.
- [ ] Data/code/protocol artifacts versioned for release.
