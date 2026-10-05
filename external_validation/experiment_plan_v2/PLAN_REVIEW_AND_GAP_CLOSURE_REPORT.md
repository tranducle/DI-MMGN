# Plan Review and Gap Closure Report — DI-MMGN External Rescue v2

## Review stance
Combined harsh reviewer + methodology auditor + statistical reviewer + reproducibility reviewer.

## Main findings
1. **The paper is no longer weak on baselines/SOTA.** The recent task-specific reimplementations materially close that complaint.
2. **The central unresolved risk is external validity.** Current manuscript wording already acknowledges it, but reviewer pressure remains because all malicious LWDED-v4 examples are controlled/reconstructed synthetic.
3. **DMoS v1 should not be rerun unchanged.** Its G1C failure is scientific: only 9 positive temporal pairs and 14 benign pairs survive a conservative clean-screen despite 449 fetched archival captures.
4. **The strongest rescue route is a predeclared DMoS-v2 deep-history protocol**, because 99 current-positive pages already have archival evidence but no clean predecessor under the shallow v1 search.
5. **The external evaluation code must be tightened for a positive-only cohort.** Attack recall/FNR should be primary; binary metrics are not authorized without adequate benign external data.
6. **A modality-matched internal control is mandatory** because DMoS lacks the original HTTP modality.
7. **A second dataset search is still required as a contingency**; current large Zone-H/Alexa-derived datasets in the literature are not yet verified as accessible temporal corpora.

## Reviewer-attack closure matrix
| Reviewer attack | Closure action |
|---|---|
| “Only one synthetic benchmark” | DMoS-v2 real-malicious temporal attack-recall validation; second-dataset search |
| “You changed the gate after failure” | preserve v1 FAIL_REDESIGN; version/hash v2 before execution |
| “Wayback predecessor may already be compromised” | deep history + zero-keyword clean rule + hash checks + manual audit |
| “External performance drop is just missing HTTP” | frozen internal no-HTTP control |
| “Positive-only set cannot establish specificity” | do not report specificity/Macro-F1 unless full external benign gate passes |
| “External threshold was tuned on test” | fixed 0.5, no DMoS calibration |
| “Host imbalance makes CI invalid” | host-aware hierarchical/cluster bootstrap + per-host reporting |
| “Why no more ablations?” | existing matched Delta/fusion/orthogonality evidence directly covers claimed mechanism; new modality robustness only where external protocol requires it |

## Gate decision
**decision: conditional_execute**

### Allowed now
- freeze protocol v2;
- implement and smoke-test deep archival enumeration;
- run the primary salvage acquisition under the frozen protocol;
- run literature/dataset discovery in parallel;
- prepare internal modality-matched control code.

### Blocked
- official external G2/G3/G4/G5 evidence promotion until v2 cohort gate passes;
- manuscript claims of real-malicious transfer until G5 passes;
- full external binary claims unless full benign cohort gate passes.

## Smallest set of changes required before official external evaluation
1. Freeze DMoS-v2 acquisition/clean-screen rules.
2. Obtain >=50 positive pairs across >=5 hosts or abandon DMoS recall-only route.
3. Add label-quality audit.
4. Add modality-matched internal control.
5. Restrict positive-only metrics to attack recall/FNR + host-aware CI.
6. Preserve all v1 failures and v2 provenance.

## Gate ledger
| Stage Boundary | Gate ID | Report | Verdict | Next Step Allowed? |
|---|---|---|---|---|
| LWDED data -> internal experiments | v4 G0/G1 | existing v4 gates | PASS | yes |
| Internal baseline/mechanism -> results | v4 G2/G3/G4/G6 | v4 gate artifacts | PASS | yes |
| SOTA runs -> SOTA claim | S4/S5 | reimplementation/S4_SOTA_RESULT_INTEGRITY.json + S5_SOTA_COMPARISON_SUMMARY.json | PASS | yes, benchmark-bounded |
| DMoS provenance -> archive search | G0/G1A | DMoS v1 artifacts | PASS | yes |
| DMoS archive discovery -> cohort | G1B | G1B_wayback_discovery.json | PASS | yes |
| DMoS cohort -> external features | G1C | G1C_DMoS_TEMPORAL_COHORT.json | FAIL_REDESIGN | **no** |
| Redesign -> v2 acquisition | V2 protocol gate | to create/freeze | PENDING | planning/smoke only |
