# Complete Experiment Rescue Plan — DI-MMGN v4

## 1. Rescue objective
Close the remaining high-risk reviewer gap — **single-benchmark / no independent real-malicious validation** — without weakening predeclared scientific gates or rerunning valid internal evidence.

This is a post-experiment rescue plan. Existing valid evidence is reused, not restarted.

## 2. Current experiment disposition
| Evidence | Decision | Reason |
|---|---|---|
| 30-run LWDED-v4 matrix | reuse_as_is | final internal G2/G3/G4/G6 gates passed |
| ZERO-Delta matched control | reuse_as_is | directly isolates classifier access to temporal change |
| Fusion + orthogonality experiments | reuse_as_is | sufficient for current narrow architectural claims |
| SOTA reimplementations | reuse_as_is | 3 seeds each; integrity + paired bootstrap passed |
| DMoS G0/G1A/G1B | reuse_as_is | valid provenance and archival-discovery evidence |
| DMoS G1C v1 | reuse_as_negative_result | legitimate FAIL_REDESIGN; must not be overwritten |
| DMoS G3/G4/G5 v1 | exclude/not-run | G1C did not authorize them |

## 3. Central evidence contracts

### E1 — Independent real-malicious transfer
**Question:** Does the frozen DI-MMGN detector retain attack sensitivity on independently collected real defacement incidents?

**Minimum authorized evidence:**
- >=50 real positive temporal pairs,
- >=5 independent host groups,
- frozen checkpoints (seeds 42/43/44),
- fixed threshold 0.5,
- no DMoS retraining/calibration,
- host-aware confidence interval,
- modality-matched internal control.

**Strong target:** >=100 real positive pairs across >=8 host groups.

**Allowed wording if minimum passes:** “On an independently sourced real-defacement temporal cohort, the frozen model achieved X attack recall under [available modalities].”

**Not allowed:** deployment readiness, universal generalization, or full binary superiority.

### E2 — Full external binary discrimination
Requires:
- >=100 positive pairs across >=8 hosts,
- >=100 same-source benign clean-to-clean pairs across >=8 hosts,
- identical acquisition/screening rules,
- full confusion-matrix metrics and host-aware uncertainty.

If this condition is not met, Macro-F1, specificity, and external false-positive claims are not authorized.

## 4. DMoS protocol v2 — deep archival salvage

### 4.1 Preserve v1
Keep all v1 files and the FAIL_REDESIGN report immutable. Protocol v2 uses a new directory and new artifact names.

### 4.2 Candidate pool
DMoS v1 contains:
- 417 current pages with positive defacement-keyword signal.
- 108 of these have at least one archived capture.
- 9 already have a clean predecessor.
- **99 current-positive pages have captures but no clean predecessor** — primary salvage pool.
- 309 current-positive pages had no usable v1 capture — secondary pool only if needed.

### 4.3 Acquisition redesign
Use an archival enumeration method that returns the available capture timeline (prefer CDX-style enumeration when accessible) instead of repeatedly requesting only the closest snapshot.

Frozen v2 rules before official acquisition:
1. exact inferred URL first;
2. only deterministic URL variants: http/https, www/non-www when host-equivalent, trailing-slash normalization, and archived redirect target when traceable;
3. never substitute a different path merely because it exists;
4. search up to 10 years before DMoS collection time;
5. consider at most 24 temporally diverse pre-collection captures per page;
6. status 200 only;
7. store original archive timestamp, replay URL, requested URL, effective URL, response status, SHA-256, and failure reason.

### 4.4 Clean predecessor rule
A candidate predecessor is clean-eligible only when all are true:
- pre-collection capture;
- archive bytes >=1,000;
- visible text >=100 chars;
- DMoS defacement-keyword hits = 0;
- archived HTML SHA-256 differs from the current DMoS defaced HTML;
- no mechanical retrieval error.

For a positive temporal pair:
- current DMoS page must have positive_keyword_hits >=1;
- t1 = latest eligible clean predecessor;
- t2 = released real DMoS defaced page.

Do **not** loosen keyword_hits=0 after observing v2 outcomes.

### 4.5 Label-quality audit
For any final external-positive cohort intended for the paper:
- manually review every retained t1/t2 pair if cohort <=100; otherwise review all positives plus a random audit of excluded candidates;
- preferably use two independent raters blinded to DI-MMGN predictions;
- record clean/defaced/ambiguous;
- adjudicate disagreement;
- report Cohen’s kappa or percentage agreement when two raters are used;
- ambiguous pairs are excluded before model inference.

### 4.6 Salvage gates
**V2-G1a (mechanical smoke):** 5 records, successful enumeration/replay/hashing; no performance inspection.

**V2-G1b (primary salvage):** exhaust all 99 primary-pool records under the frozen v2 acquisition protocol.
- PASS_ATTACK_MIN if total valid positives (existing + recovered) >=50 and >=5 hosts.
- PASS_ATTACK_STRONG if >=100 and >=8 hosts.
- otherwise FAIL_REDESIGN_DMoS.

**V2-G1c (secondary discovery, conditional):**
Run the 309 no-capture current-positive pages only if V2-G1b is close enough that additional exact-URL archive enumeration could plausibly cross the minimum, or if v1 failure reasons indicate index/API incompleteness rather than true archive absence.
The continuation condition must be written before V2-G1b results are inspected.

## 5. External evaluation after a passing attack gate

### G2 — External cohort build
- build immutable pair/snapshot manifests;
- deduplicate by URL/timestamp/hash;
- enforce host-group identity;
- hash manifests;
- freeze before inference.

### G2.5 — Modality-matched internal robustness control
Because DMoS cannot provide archival HTTP features:
- evaluate the same frozen CONCAT checkpoints on LWDED-v4 test with mask [text, DOM, visual, no-HTTP];
- if external rendering cannot reach >=95% pair coverage, use predeclared text+DOM mode and run the matching LWDED-v4 mask;
- this is a missing-modality robustness control, **not** a retrained ablation.

### G3 — External feature integrity
- identical pinned MiniLM text pipeline;
- identical DOM schema/caps;
- deterministic offline screenshots;
- CLIP only if render pair coverage >=95%;
- no HTTP fabrication;
- all feature files finite and hash-traceable.

### G4 — Frozen external evaluation
- checkpoints: CONCAT seeds 42/43/44;
- threshold: 0.5 fixed;
- no retraining, fine-tuning, calibration, or threshold selection on DMoS;
- positive-only route primary metric: **attack recall**;
- secondary: false-negative rate and per-host recall;
- do not headline precision, accuracy, defaced-F1, Macro-F1, or specificity on a positive-only cohort.

### G5 — Statistical evidence
- report seed mean +/- sample SD;
- hierarchical/cluster bootstrap over host group and seed;
- 20,000 replicates is acceptable if frozen before execution;
- report 95% CI for attack recall and generalization gap vs modality-matched internal recall;
- report host-level dispersion and worst-host recall;
- avoid p-values unless a specific confirmatory hypothesis is declared in advance.

## 6. Stronger top-Q1 package
If resources permit after minimum external validation:
1. run DefacementFusion 2025 and BiLSTM+EfficientNet 2021 on the same external current snapshots; compare **attack recall only** if no benign external cohort;
2. if same-source benign cohort becomes large enough, extend to full binary metrics;
3. add a second independently sourced external corpus or later temporal window;
4. add runtime/throughput for frozen external inference, not only training;
5. release the v4 manifests, configs, result JSON, prediction files, and protocol hashes.

## 7. Dataset fallback track
The DMoS route must not be the only contingency.

Existing literature assets point to large Alexa/Zone-H-style datasets (about 96k webpages in the 2021/2024/2025 line of work) and a 96,220-page 2026 RTFDD dataset, but current project files do not establish public access to the raw URL/timestamp corpus. Treat them as **candidate_needs_access_or_license_check**, not available datasets.

A literature/dataset search must verify:
- raw HTML availability;
- per-page source URL;
- capture timestamp;
- normal/defaced label provenance;
- whether same-site predecessor reconstruction is possible;
- license/terms and redistribution limits.

If none provides temporal provenance, classify it as useful for static comparison/stress testing only, not as a replacement for temporal external validation.

## 8. Failure-mode / negative-result contribution
If DMoS v2 still fails:
- retain the result as evidence that public defacement corpora often lack sufficient clean temporal provenance for paired evaluation;
- keep manuscript claims benchmark-bounded;
- add a concise reproducibility/limitations paragraph rather than fabricating a second-dataset result;
- optionally use DMoS static/crafted data only as supplementary stress analysis, clearly separated from the central temporal-generalization claim.

## 9. Minimum publishable vs strong package
| Package | Required evidence |
|---|---|
| Minimum revision-strengthening | SOTA comparison (done) + >=50 independent real attack pairs / >=5 hosts + frozen attack-recall evaluation + modality-matched internal control |
| Strong Top-Q1 | >=100 real attack pairs / >=8 hosts + ideally >=100 same-source benign pairs / >=8 hosts OR second independent temporal source + external comparator recall + host-aware uncertainty + artifact release |
