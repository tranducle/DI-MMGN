# Reproducibility

## Environment used for the corrected runs

- Python 3.12.10
- PyTorch 2.6.0 with CUDA 12.4
- torch-geometric 2.7.0
- transformers 4.57.6
- sentence-transformers 5.2.3
- scikit-learn 1.7.2
- NumPy 2.3.5
- PyYAML 6.0.2
- beautifulsoup4 4.15.0
- lxml 5.4.0
- NVIDIA RTX 3090, 24 GB VRAM

Install the Python dependencies from `requirements.txt`. Use the CUDA-compatible PyTorch wheel appropriate for the local CUDA installation.

## Frozen experiment settings

The main configuration is `src/config/experiment_v3.yaml`.

Key settings:

- embedding dimension: 256
- DOM hidden dimension: 128
- DOM layers: 3
- dropout: 0.15
- learning rate: 1e-4
- batch size: 32
- gradient accumulation: 1
- auxiliary-direction phase: 30 epochs
- joint finetuning phase: 30 epochs
- seeds: 42, 43, 44
- checkpoint selection: validation macro-F1
- test threshold: 0.5 for the primary result

The parameter-matched CONCAT and ZERO-Delta models each have 618,881 trainable parameters.

## 1. Validate the data installation

After placing the binary LWDED features under `dataset_pipeline/`:

```bash
python scripts/smoke_dataset_loader.py
```

For a full dataset audit:

```bash
python scripts/validate_dataset.py
```

## 2. Static baselines

```bash
python src/run_v3_static_baselines.py --models graphsage gat gin snapshot_mm --seeds 42 43 44
```

This produces checkpoints, per-sample predictions, and baseline result JSON files under `results/`.

## 3. Temporal and fusion models

```bash
python src/run_v3_proposed.py --runs concat_none concat_soft zero_delta_none gated_none crossattn_none --seeds 42 43 44
```

The corrected Phase A objective is supervised. Legitimate pairs map to +1 on the auxiliary benign direction and defaced pairs map to -1.

## 4. Text baseline

Build the full token-sequence cache:

```bash
python src/build_text_sequence_cache_v3.py
python src/validate_text_sequence_cache_v3.py
```

Then run the MiniLM+BiLSTM baseline:

```bash
python src/run_v3_text_baseline.py --seeds 42 43 44
```

## 5. Post-hoc analysis

The corrected analysis historically used per-sample prediction artifacts from the 30-run core matrix, but those JSON files are not present in the current staging copy. After restoring them, or after regenerating them from the released features and frozen training protocol, recompute the paired benchmark analysis with:

```bash
python scripts/analyze_results.py
python scripts/analyze_errors.py
```

The domain-and-seed bootstrap uses 20,000 replicates and NumPy random seed 20260924.

The operating-point analysis requires the trained CONCAT-soft checkpoints:

```bash
python scripts/analyze_operating_points.py
```

Thresholds in that analysis are selected using validation data only and then frozen for test evaluation. The analysis is exploratory and is not presented as preregistered inference.

## Expected corrected headline results

Across seeds 42, 43, and 44:

- CONCAT-none macro-F1: 0.8666
- CONCAT-soft macro-F1: 0.8681
- ZERO-Delta macro-F1: 0.8117
- static multi-modal snapshot macro-F1: 0.7887
- MiniLM+BiLSTM macro-F1: 0.6929

The parameter-matched CONCAT-none minus ZERO-Delta difference is 0.0549 macro-F1. The post-hoc domain-and-seed bootstrap 95% interval is 0.0232 to 0.0901.

These values are benchmark-specific. They do not establish external real-world attack performance.
