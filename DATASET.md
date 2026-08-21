# LWDED — Large-scale Website Defacement Evaluation Dataset (v2)

**DOI:** [10.5281/zenodo.22051075](https://doi.org/10.5281/zenodo.22051075) · **License:** CC BY 4.0 · **Version:** v2.0 (2026-08-21)

The canonical dataset release used by *DI-MMGN: Benign-Update Tolerant Website Defacement Detection via Concatenated Multi-Modal Change Vectors*.

## Summary

| Property | Value |
|---|---|
| Temporal pairs | **10,446** ($S_{t-1}, S_t$) |
| Domains | 195 (unique) |
| Split policy | domain-blocked: 138 pretrain / 19 validation / 38 test |
| Labels | `legitimate` (4,864) · `defaced` (5,582) |
| Attack vectors | overt/wholesale (3,963) · semantic (615) · hidden (387) · structural (617) |
| Legit mutations | 751 synthetic CMS-style mutations + 4,113 organic changes |
| Modalities | text (384-d) for all pairs; **2,730 pairs** additionally carry DOM graph (70-d node features), CLIP visual (512-d), HTTP (32-d) |

Wikipedia (7,167 pairs, 75.8% of the pretrain split) is confined to pretraining by the
domain-blocked policy in `splits.json`; a programmatic leakage audit enforces zero domain overlap
across splits.

## Package contents (on Zenodo)

| Archive | Contents |
|---|---|
| `01_lwded_v2_pairs_metadata.zip` | `unified_pairs.jsonl` (10,446 records), `manifest.jsonl`, 10,446 per-sample metadata JSONs, pipeline logs |
| `02_lwded_v2_dom_graphs.zip` | 4,224 torch-serialised DOM graphs |
| `03_lwded_v2_visuals_http.zip` | 4,224 CLIP visual vectors + 4,224 HTTP signal vectors |
| `04_cms_eval_and_splits.zip` | CMS Offline Mutation corpus (10 templates × 4 attack vectors) + `splits.json` |
| `SHA256SUMS.txt` | per-file SHA-256 (post-extraction verification) |
| `CHECKSUMS_ZIPS.txt` | per-archive SHA-256 |
| `README.md` | full dataset documentation (schemas, statistics, provenance, ethics) |

## Record schema (`unified_pairs.jsonl`, one JSON per line)

```json
{
  "domain": "163.com",
  "label": "defaced",
  "attack_type": "structural",
  "sample_json": "v2/samples/defaced/163.com/20260430092400__structural__c5e86c.json",
  "delta_dom_graph": {"t1": "v2/graphs/163.com/raw__20260430092400.pt",  "t2": "..."},
  "delta_visual":    {"t1": "v2/visuals/163.com/raw__20260430092400.npy", "t2": "..."},
  "delta_http":      {"t1": "v2/http/163.com/raw__20260430092400.npy",    "t2": "..."}
}
```

A modality is available for a pair iff both `t1` and `t2` are non-null — this is exactly the rule
implemented by `src/data/modality_mask.py`, so results reproduce from this package + this repo.

## Reproduction

```bash
pip install -r requirements.txt
python src/train_dimmgn.py --config src/config/default.yaml   # streams v2/unified_pairs.jsonl
python src/evaluate.py                                        # per-seed Test-F1
```

Training seeds: 42 / 43 / 44. Model checkpoints (`*.pt`) are not tracked in this repository.

## Known limitation (disclosed)

Within each (template, attack vector) of the CMS Offline Mutation corpus, the 5 numbered variants
are byte-identical deterministic outputs of the seeded mutation engine (documented in the
package's `CMS_EVAL_MANIFEST.md`). Distinct seeded variants are future work.

## Citation

```bibtex
@misc{lwded2026,
  author = {Le, Tran Duc},
  doi = {10.5281/zenodo.22051075},
  howpublished = {Zenodo},
  note = {Version 2.0. doi: 10.5281/zenodo.22051075},
  title = {{LWDED}: Large-scale Website Defacement Evaluation Dataset},
  year = {2026}
}
```