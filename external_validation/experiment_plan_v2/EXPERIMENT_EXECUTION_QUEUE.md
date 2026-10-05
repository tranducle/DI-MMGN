# Experiment Execution Queue — External Validation Rescue v2

| ID | Priority | Task | Gate / acceptance | Output | Depends on |
|---|---:|---|---|---|---|
| R0 | 0 | Freeze DMoS v1 negative result and hashes | v1 FAIL_REDESIGN preserved | v1 provenance note | done |
| R1 | 1 | Write/freeze DMoS-v2 protocol JSON/MD | protocol hash created before acquisition | DMOS_V2_PROTOCOL.md/json | R0 |
| R2 | 1 | Implement CDX/deep-history enumerator; 5-record mechanical smoke | retrieval + replay + hashing works; no metric inspection | V2_G1A_SMOKE.json | R1 |
| R3 | 1 | Deep salvage all 99 current-positive captured-but-no-clean records | >=50 total positives & >=5 hosts for minimum; >=100 & >=8 strong | V2_G1B_PRIMARY_SALVAGE.jsonl/json | R2 |
| R4 | 2 | Conditional archive enumeration of 309 current-positive no-capture pages | execute only under predeclared continuation condition | V2_G1C_SECONDARY_DISCOVERY.json | R3 |
| R5 | 1 | Label-quality audit of retained positive pairs | no ambiguous retained pairs; agreement recorded if 2 raters | V2_LABEL_AUDIT.csv/json | R3/R4 |
| R6 | 1 | Build immutable external pair/snapshot manifests | dedup + hashes + host IDs PASS | G2_COHORT_BUILD_v2.json | R5 |
| R7 | 1 | Internal frozen no-HTTP mask control, seeds 42/43/44 | predictions aligned; integrity PASS | G2_5_MODALITY_MATCHED.json | R6 |
| R8 | 1 | Build external text/DOM/visual features | finite; render coverage rule enforced | G3_EXTERNAL_FEATURES_v2.json | R6 |
| R9 | 1 | Frozen DI-MMGN external eval | no retrain/calibration; 3 seeds; aligned predictions | G4_EXTERNAL_FROZEN_EVAL_v2.json | R7/R8 |
| R10 | 1 | Host+seed bootstrap and failure analysis | CI + per-host table; positive-only metric restrictions | G5_EXTERNAL_EVIDENCE_v2.json | R9 |
| R11 | 2 | External task-specific comparator recall | same external current pages; no tuning on test | G6_EXTERNAL_COMPARATORS.json | R10 strong-target preferred |
| R12 | 1 | Update manuscript + responses + limitations | claim wording passes G8 | revised main.tex / responses | R10 or documented failure |
| F1 | 1 | Literature-hunter search for second temporal-capable dataset | verified access/provenance/fit register | DATASET_DISCOVERY_AND_FIT_REGISTER.md | parallel with R1-R5 |

## Supervisor policy
- New supervisor namespace/version; never overwrite v1 state.
- Every stage writes state + immutable gate JSON before advancing.
- Mechanical failures may retry automatically.
- Scientific FAIL_REDESIGN stops forward execution.
- G3/G4/G5 cannot run unless the v2 cohort gate authorizes them.
- No threshold or clean-screen change may be made after official v2 acquisition begins.
