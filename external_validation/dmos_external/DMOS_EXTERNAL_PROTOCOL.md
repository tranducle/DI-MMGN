# DMoS External Validation Protocol — Frozen Before Acquisition

## Goal
Test whether frozen DI-MMGN-v4 transfers to an independently collected real-defacement corpus released with DMoS (USENIX Security 2021), without retraining on DMoS.

## Source
- Public DMoS English release.
- 500 released real-defaced HTML pages.
- 498 non-empty files.
- Original archive mtimes preserved and used as per-page collection timestamps.
- DMoS keyword list: 1,428 English illicit bigrams.
- Independent from LWDED-v4 construction.

## Primary temporal-pair construction
For each non-empty DMoS real-defaced page:
1. infer the original page URL from absolute/relative canonical metadata, DMoS release path, or documented fallback;
2. search Wayback for an archived capture strictly **before** the DMoS collection timestamp;
3. accept a same-URL predecessor only if it is HTML, non-trivial, and passes the conservative clean-screen below;
4. pair that archived predecessor as t1 with the released DMoS real-defaced HTML as t2.

No post-collection archive capture is allowed as t1.

## Conservative clean-screen
A predecessor is eligible only if:
- HTTP/archive retrieval succeeds;
- HTML >= 1,000 bytes and visible text >= 100 characters;
- archive timestamp < DMoS collection timestamp;
- DMoS illicit-bigram hit count = 0;
- capture digest is not duplicated within that URL's candidate set.

A DMoS positive is included in the primary temporal cohort only when its released page has >=1 DMoS illicit-bigram hit. This makes the clean-vs-real-defaced transition auditable rather than inferred only from file labels.

## Matched benign pair
When >=2 eligible clean captures exist for the same URL before collection:
- choose the two most recent distinct clean captures in chronological order;
- create clean->clean organic benign pair.

## Predeclared coverage gates
### G1A feasibility probe
Stratified sample of up to 60 non-empty released pages across host groups and URL-source types.
- PASS: >=20% have at least one pre-collection same-URL capture and >=5 host groups represented.
- PASS_WITH_WARNINGS: >=10% and >=3 host groups.
- FAIL_REDESIGN: below those thresholds.

### G1B main external-cohort gate
- FULL PASS: >=100 real-defaced temporal pairs across >=8 host groups **and** >=100 clean->clean benign pairs across >=8 host groups.
- PASS_ATTACK_RECALL_ONLY: >=50 real-defaced temporal pairs across >=5 host groups but benign target not met.
- FAIL_REDESIGN: <50 positive pairs or <5 host groups.

These thresholds are frozen before full acquisition.

## Modalities
External DMoS lacks historical HTTP metadata.
- text: derived with the pinned LWDED-v4 MiniLM pipeline;
- DOM: derived with the frozen LWDED-v4 graph schema;
- visual: deterministic offline render + pinned CLIP when render gate passes;
- HTTP: **missing and masked out**, never fabricated.

The frozen model's modality mask natively supports missing HTTP.

Visual-render gate is predeclared as follows: attempt deterministic offline rendering with the frozen v4 Chromium policy and repair retries. If both snapshots render successfully for >=95% of otherwise eligible pairs, and the remaining visually complete pairs still satisfy the applicable G1C sample/host threshold, the external evaluation uses only those visually complete pairs with a single cohort-wide mask `text+DOM+visual` and HTTP absent. Otherwise the entire external evaluation uses all G1C-authorized pairs with the cohort-wide `text+DOM only` mask. The internal modality-matched control always uses the same mask as the external cohort. We do not mix masks opportunistically per example.

## Modality-matched internal control
Before interpreting external performance, run the frozen DI-MMGN checkpoints on the original LWDED-v4 test set with the **same HTTP-disabled mask**. This quantifies the performance cost of missing HTTP separately from external-domain shift.

If visual rendering fails quality gates on the external cohort, visual is also masked and the internal control is rerun with the same text+DOM-only mask.

## Evaluation
- no DMoS retraining or threshold tuning;
- frozen seeds 42/43/44;
- original fixed 0.5 decision threshold;
- primary external metrics if full cohort exists: macro-F1, defaced F1, attack recall, legitimate specificity, accuracy;
- attack-recall-only protocol if G1B allows only positive cohort;
- family/host-clustered bootstrap for uncertainty when cohort size permits.

## Claim boundary
Passing this protocol supports **independent external real-defacement validation under missing-HTTP modality masking**.
It does not establish deployment readiness or universal OOD robustness.
