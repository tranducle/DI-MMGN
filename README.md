# DI-MMGN

**Benign-Update Tolerant Website Defacement Detection via Snapshot-Concatenated Change Vectors**

## Main Idea

Traditional website defacement detectors learn a fixed normality distribution from static page snapshots. This approach is highly susceptible to false alarms on the modern web, where legitimate CMS updates, seasonal campaigns, and structural redesigns continuously shift the benign class. The model becomes unable to distinguish a legitimate redesign from a malicious defacement.

**DI-MMGN** reframes defacement detection as a *temporal change-vector analysis* problem. For each pair of temporal snapshots $(S_{t-1}, S_t)$, a shared multi-modal encoder computes a change vector $\Delta_t = E(S_t) - E(S_{t-1})$ across text, DOM graph, visual, and HTTP signals. 

However, relying strictly on the change vector discards the absolute context of the page, leading to a massive false-positive rate on legitimate CMS mutations. The core architectural innovation of DI-MMGN is **concatenating the static snapshot embedding with the temporal change vector** — $[E(S_t);\,\Delta_t]$. This provides the downstream classifier with both *what the page is* and *what changed*, reducing false positives on legitimate mutations by **5.8$\times$** while preserving near-perfect recall on hidden and structural defacements. 

## Code Structure

- `src/`: Contains the PyTorch 2.6 implementations for the DI-MMGN framework (rebuilt v4 reproducibility artifact package).
  - `train_dimmgn.py`: Script to train the core DI-MMGN concatenated architecture.
  - `evaluate.py`: Evaluation loop for generating F1 scores and testing false-positive rates on legitimate pairs.
  - `ablation_*.py`, `train_fusion_ablation.py`: Various ablations for modality and fusion operators.
  - `models/`: Neural network architecture definitions.

## Data Availability

This repository contains the **v4 reproducibility artifact package** and the **post-training external-validation artifact package**. The numerical results are produced by the rebuilt **LWDED-v4 protocol**, which is a full-modality, registrable-family-blocked protocol containing 4,678 temporal pairs and a 936-pair held-out test set. It also includes an independent evaluation on 161 independently sourced real temporal defacement pairs.

**Notice:** The previously released LWDED v2 archive remains publicly available through its Digital Object Identifier (DOI) at [https://doi.org/10.5281/zenodo.22051075](https://doi.org/10.5281/zenodo.22051075). The v4 dataset/features and complete reproduction package have not yet been assigned a public archive identifier. Therefore, the v2 DOI should not be interpreted as reproducing the v4 headline results. A versioned public release of the redistributable v4 artifacts is required for independent end-to-end reproduction.

## Usage

1. Create a Python environment.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline (requires LWDED-v4 data placement):
   ```bash
   python src/train_dimmgn.py --config src/config/default.yaml
   ```

*Note: Model checkpoints (`*.pt`) and raw dataset files are not tracked in this repository.*
