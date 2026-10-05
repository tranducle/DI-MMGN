# LWDED — Large-scale Website Defacement Evaluation Dataset (v4)

The canonical dataset release used by the revised version of *DI-MMGN: Benign-Update Tolerant Website Defacement Detection via Snapshot-Concatenated Change Vectors*.

## Summary (v4 Protocol)

| Property | Value |
|---|---|
| Temporal pairs | **4,678** ($S_{t-1}, S_t$) |
| Held-out Test Set | **936** pairs |
| External Validation Set | **161** independently sourced real temporal defacement pairs |
| Split policy | Registrable-domain-blocked |
| Modalities | Text, DOM graph, CLIP visual, HTTP signals (Full-modality) |

## Data Availability & Versioning Notice

The numerical results in the article are produced by the rebuilt **LWDED-v4** protocol. The v4 reproducibility artifact package preserves the frozen pair and split manifests, modality-provenance records, feature schemas, experiment scripts, checkpoint-hash manifest, all 30 internal per-pair prediction files, aggregate result files, and the statistical-analysis scripts used for the reported tables and figures. 

The post-training external-validation artifact package additionally preserves the frozen 161-pair real-defacement manifest, source and licensing provenance, external per-pair predictions, statistical summaries, sensitivity analyses, integrity checks, and claim-permission reports used for the external results.

**Important distinction:** 
The previously released LWDED v2 archive remains publicly available through its Digital Object Identifier (DOI) at [https://doi.org/10.5281/zenodo.22051075](https://doi.org/10.5281/zenodo.22051075). The v4 dataset/features and complete reproduction package have not yet been assigned a public archive identifier; therefore the v2 DOI should not be interpreted as reproducing the v4 headline results. A versioned public release of the redistributable v4 artifacts is required for independent end-to-end reproduction, and any release must preserve source-specific redistribution and licensing restrictions.

## Reproduction

```bash
pip install -r requirements.txt
python src/train_dimmgn.py --config src/config/default.yaml   
python src/evaluate.py                                        
```
