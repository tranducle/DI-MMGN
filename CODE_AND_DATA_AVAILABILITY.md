# Code and data availability

The source LWDED v2 dataset is available from Zenodo at https://doi.org/10.5281/zenodo.22051075.

This repository contains the corrected LWDED-v3 loader, model code, experiment configuration, data manifests, provenance records, aggregate results, and derived analysis summaries. Raw webpage content, screenshots, graph tensors, feature arrays, model checkpoints, and the per-sample prediction JSON files from the corrected run matrix are not included in the current staging copy.

The corrected v3 experiment depends on a companion binary feature release that preserves the relative paths recorded in `dataset_pipeline/v3/unified_pairs.jsonl`. Complete independent reproduction also requires the corrected per-sample prediction bundle for direct result reconciliation, or regeneration of those predictions from the released features and training protocol. The companion v3 release has not yet been assigned a public identifier. The final manuscript should cite the v3 data deposit once it is published. Until then, the v2 DOI alone should not be described as reproducing the corrected v3 headline results.
