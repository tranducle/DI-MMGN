# Data

## Source release

The corrected LWDED-v3 cohort is derived from the LWDED v2 research release:

- LWDED v2: https://doi.org/10.5281/zenodo.22051075

LWDED-v3 changes the official evaluation cohort and text representation. It removes legacy text-only examples from the official experiment, rebuilds organic benign pairs from consecutive clean snapshots, requires all four modalities for every official pair, and uses one pinned MiniLM text representation.

## Official v3 cohort

- temporal pairs: 3,279
- domains: 194
- defaced pairs: 1,971
- legitimate pairs: 1,308
- pretrain: 2,293 pairs from 137 domains
- validation: 360 pairs from 19 domains
- test: 626 pairs from 38 domains
- organic clean-to-clean pairs: 557
- CMS-style benign mutations: 751

All malicious examples are synthetic or reconstructed synthetic. Organic benign examples are archive-assumed legitimate rather than independently adjudicated maintenance events.

## Required local layout

The tracked manifests use paths relative to `dataset_pipeline/`. A complete local data installation must provide the referenced files under the following layout:

```text
dataset_pipeline/
  v2/
    graphs/
    visuals/
    http/
    samples/
  v3/
    text/
    unified_pairs.jsonl
    snapshot_manifest.jsonl
    text_manifest.jsonl
    splits.json
    text_encoder_provenance.json
```

The text-only baseline also requires:

```text
dataset_pipeline/v3/text_sequence_cache.npy
dataset_pipeline/v3/text_sequence_mask.npy
dataset_pipeline/v3/text_sequence_index.json
```

The repository tracks the small JSON/JSONL manifests but does not track the binary feature arrays.

## Integrity hashes

- official pair manifest: `241805316dc2b22a08c8047e57cacbeeb1ec1349c1904cbd79f4c71b14c78b96`
- snapshot manifest: `15fcf8648b37d0005441759821486db312400a148c14895c66ed76fb1bccd028`
- text manifest: `46f6d1d503784e51f4b97f119bbd59874783513c94a874341af96a52f3ebeaa1`

Additional source fingerprints and representation metadata are stored in `dataset_pipeline/v3/source_v2_fingerprint.json`, `text_encoder_provenance.json`, and `text_sequence_provenance.json`.

## Text representation

The corrected snapshot text representation uses:

- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- maximum length: 512 tokens
- visible-text preprocessing: BeautifulSoup extraction, whitespace collapse, then a 3,072-character cap before tokenization
- pooling: `last_hidden_state[:, 0]`
- output dimension: 384
- model mode: evaluation
- gradient tracking: disabled

## Distribution status

The v2 source archive is public at the DOI above. The corrected v3 binary feature extension is not included in this GitHub tree. It must receive a companion public deposit before the manuscript's final Code and Data Availability statement can claim complete public reproduction of the v3 results.
