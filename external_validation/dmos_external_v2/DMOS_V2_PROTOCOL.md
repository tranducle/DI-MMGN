# DMoS External Temporal Validation Protocol v2

Frozen before official v2 acquisition: 2026-10-02.

## Objective
Attempt to recover enough independently sourced, real-defacement temporal pairs from the DMoS release to support a frozen-model external **attack-recall** validation. This protocol is a redesign after v1 legitimately ended at `FAIL_REDESIGN`; v1 artifacts are preserved and are not overwritten.

## Source and inherited v1 facts
- DMoS English real-defacement release: 500 files; 498 non-empty usable records.
- v1 found pre-collection archival captures for 114/498 pages across 15 hosts.
- v1 fetched/screened 449 archival captures with zero final fetch errors.
- v1 yielded only 9 positive temporal pairs across 9 hosts and 14 benign temporal pairs across 12 hosts.
- v1 gate remains `FAIL_REDESIGN`.

## Candidate pools
The v2 pool builder deterministically derives:
- existing positives: the 9 v1 positive pairs;
- primary salvage pool: current DMoS page has >=1 DMoS keyword hit, has at least one v1 archival capture, but no v1 clean predecessor; expected count 99;
- secondary pool: current DMoS page has >=1 DMoS keyword hit and had no v1 archival capture; expected count 309.

## Archival search
For each candidate:
1. Search exact inferred URL first.
2. Deterministic URL variants only: http/https scheme swap; www/non-www host variant; trailing-slash normalization when syntactically applicable.
3. Never substitute another path merely because it exists.
4. Search from 10 years before the DMoS collection timestamp through the collection date.
5. Query Wayback CDX-style timeline metadata for status-200 captures and collapse exact duplicate digests when supported.
6. Merge/deduplicate all variant results by digest when available, otherwise timestamp+original URL.
7. If <=24 unique captures exist, select all. If >24 exist, select exactly 24 capture indices spaced uniformly across the chronological timeline, including oldest and newest.
8. Fetch selected captures newest-to-oldest. Stop after two clean captures are found or the selected set is exhausted.
9. Preserve requested URL, selected original archive URL, timestamp, digest, replay URL, final URL, status, hash, bytes, visible characters, keyword hits, and error reason.

## Clean predecessor rule
A fetched candidate is clean-eligible only if all are true:
- timestamp precedes the DMoS collection time;
- replay status is HTTP 200;
- archive bytes >= 1,000;
- visible text >= 100 characters;
- DMoS keyword hits = 0;
- archive SHA-256 differs from the released DMoS current-page SHA-256;
- no mechanical retrieval error.

The criterion `keyword_hits == 0` is frozen and must not be relaxed after outcomes are observed.

For positive temporal pairs:
- released current DMoS page must have `positive_keyword_hits >= 1`;
- t1 is the newest clean-eligible capture among the predeclared sampled captures;
- t2 is the released real DMoS page.

If two clean captures are recovered, they may also form one clean-to-clean benign pair, using the older and newer clean captures in temporal order.

## Gates

### V2-G0 protocol/provenance
PASS only when:
- v1 `G1C_DMoS_TEMPORAL_COHORT.json` still exists and says `FAIL_REDESIGN`;
- protocol SHA-256 matches the frozen hash file;
- pool counts match expected provenance.

### V2-G1A mechanical smoke
Run exactly five predeclared primary-pool records spanning known captured hosts.
PASS when:
- five output records are written;
- no unhandled exception;
- at least one record successfully obtains archive timeline metadata;
- at least one replay fetch returns HTTP 200.
This gate is mechanical only; recovered-label counts are not promoted as evidence.

### V2-G1B primary salvage
Run all 99 primary-pool records under the frozen protocol.
Let `p_new` be newly recovered positive pairs and combine them with the 9 inherited v1 positives.
- `PASS_ATTACK_STRONG`: >=100 total positive pairs and >=8 host groups.
- `PASS_ATTACK_MIN`: >=50 total positive pairs and >=5 host groups.
- `CONTINUE_SECONDARY`: total <50, but `p_new >=10`.
- otherwise `FAIL_REDESIGN_DMoS`.

The `p_new >=10` continuation rule is frozen before execution. Its rationale is that 10/99 primary recovery extrapolated naively to 309 secondary candidates gives roughly 31 additional candidates, making the 50-pair minimum at least arithmetically plausible. This is a continuation heuristic, not a performance claim.

### V2-G1C secondary salvage
Executed only after `CONTINUE_SECONDARY`.
Run all 309 secondary-pool records with identical rules.
Final gate:
- `PASS_FULL_COHORT`: >=100 positive pairs across >=8 hosts AND >=100 benign pairs across >=8 hosts.
- `PASS_ATTACK_STRONG`: >=100 positive pairs across >=8 hosts.
- `PASS_ATTACK_MIN`: >=50 positive pairs across >=5 hosts.
- otherwise `FAIL_REDESIGN_DMoS`.

No threshold is weakened after execution.

## External evaluation authorization
- G2/G3/G4/G5 may run only after a v2 cohort gate returns a PASS verdict.
- No DMoS retraining, fine-tuning, calibration, or threshold selection.
- Frozen DI-MMGN CONCAT checkpoints for seeds 42/43/44.
- Classification threshold fixed at 0.5.
- DMoS lacks the original HTTP modality. A frozen internal no-HTTP control is required before external-domain-shift interpretation.
- If visual rendering coverage <95%, the predeclared fallback is text+DOM with a matching frozen internal mask control. The better of the two masks must not be selected using external labels.

## Metric boundary
If only a positive external cohort passes:
- primary: attack recall;
- secondary: false-negative rate and per-host recall;
- host-aware uncertainty.
Do not headline external Macro-F1, specificity, accuracy, or precision as binary-task evidence.

Full binary metrics are authorized only if `PASS_FULL_COHORT`.

## Label-quality audit
The DMoS current label is inherited from the released real-defacement corpus; predecessor labels are determined by the frozen objective clean-screen above. A human/independent review package will be generated for retained pairs before manuscript claim promotion. Pair selection must never use DI-MMGN predictions. If later review materially disputes the deterministic labels, external claim promotion is blocked rather than silently removing outcome-unfavorable pairs.

## Statistical plan
- evaluate three frozen seeds: 42, 43, 44;
- report seed mean and sample SD;
- hierarchical/cluster bootstrap over seed and DMoS host group;
- 20,000 bootstrap replicates, RNG seed 20261002;
- report 95% intervals;
- compare external attack recall with the modality-matched internal attack recall as a generalization-gap analysis, not as a significance contest.

## Failure path
If v2 fails the final cohort gate, preserve the negative result, keep manuscript claims benchmark-bounded, and continue the parallel search for another temporal-capable external dataset. DMoS may still be used for clearly labeled supplementary static/stress analysis, not as a substitute for paired temporal validation.
