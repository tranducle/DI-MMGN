# Paper-Fidelity Specification — DefacementFusion 2025

Paper: **DefacementFusion: A Robust Multi-Modal Defacement Detection** (VCRIS 2025, DOI 10.1109/VCRIS68011.2025.11250550)

## Explicitly specified by the paper
- Inputs: current webpage HTML structure, visible/text content, and current screenshot.
- HTML branch: HTMLDeface2vec based on BERT-base-uncased.
- DOM cleaning removes low-value nodes such as <script> and <style>; class/id attributes are preserved.
- DOM traversed in preorder and segmented into overlapping subtrees using a fixed token-size window m.
- Figure 2 assigns token metadata for Tag, Depth, Node index, Position, and Text.
- HTML pretraining: BERT-style MLM.
- Masking: 15% token positions selected; node-aware sampling; 80% [MASK], 10% random token, 10% unchanged.
- BERT-base-uncased: ~110M params, hidden size 768.
- Further HTML pretraining: 5 epochs, batch 16, LR 2e-5.
- Text branch: tokenizer + BiLSTM; paper explicitly states it follows Nguyen et al. 2021.
- Image branch: EfficientNetB0; screenshot 224x224, pixels normalized to [0,1]; follows Nguyen et al. 2021.
- Fusion: modality feature vectors H, T, I -> Transformer encoder/self-attention -> MLP -> benign/defaced.
- Classification: 10 epochs, batch 16, Adam, LR 2e-5, Binary Cross Entropy.
- HTML-only classification head description: 256 hidden units + ReLU -> 2 outputs.
- Original paper split: 80/20 on 57,120 benign + 39,100 defaced pages.

## Protocol adaptations for fair comparison
- Use the LWDED-v4 family-blocked pretrain/val/test split.
- Current snapshot S_t only.
- No delta, no prior snapshot, no HTTP branch.
- HTML MLM pretraining is allowed only on pretrain-family HTML; never val/test families.
- Validation macro-F1 used only for checkpoint selection; test evaluated once.
- Seeds 42/43/44.

## Missing from paper / required implementation assumptions
- Subtree token window m is not numerically specified; Figure 1's 5-token value is illustrative, not a declared training value.
  - Default: m=512 BERT tokens, stride=256.
- Exact formula/parameterization for tree positional embeddings is not specified.
  - Default: learned embeddings for depth, DOM node index (bucketed), and within-node token position, added to BERT input embeddings.
- Aggregation across multiple DOM subtrees is not specified.
  - Default: mean pool subtree [CLS] vectors.
- Text BiLSTM details are inherited from Nguyen et al. 2021:
  - 128 words, embedding 64, bidirectional output 128; SpatialDropout assumed 0.2.
- Common fusion dimensionality is not specified.
  - Default d_model=128; HTML 768->128 projection; text=128; image=128.
- Transformer depth/head count/FFN/dropout are not specified.
  - Default: 1 encoder layer, 4 heads, FFN=256, dropout=0.1.
- MLP fusion head dimensions are not specified.
  - Default: pooled transformer representation -> Dense 256/ReLU -> 2 logits, following the paper's two-layer classifier description.
- The paper does not explicitly state whether the HTML, BiLSTM, and EfficientNet branches are frozen during multimodal classification training. Figure 3 places all three branches inside the classification-training stage.
  - Default: initialize HTML from the MLM-pretrained checkpoint and fine-tune all three branches end-to-end with the fusion Transformer/MLP; report this as an implementation assumption.
- Paper says two output classes and Binary Cross Entropy; exact output/loss parameterization is ambiguous.
  - Default: two logits + BCEWithLogits on one-hot targets, recorded explicitly.
- Custom unlabeled HTML corpus and the authors' pretrained HTMLDeface2vec checkpoint are not public.
  - Re-pretrain from public BERT-base-uncased using LWDED-v4 **pretrain-family HTML only**.

## Execution precision decision
- Official end-to-end classification runs use **FP32**, not AMP.
- Reason: the S2B worst-case cached smoke (4 pages, 4 DOM windows/page) showed non-finite EfficientNet gradients under AMP even after GradScaler unscale, while the identical FP32 smoke passed all gradient-finiteness checks with approximately 5.64 GiB peak allocated GPU memory.
- This changes only numerical execution precision; architecture, data split, loss, learning rate, epochs, and effective batch size remain unchanged.

## Fidelity label
**Paper-faithful reimplementation, not an exact or official reproduction.**
All assumptions above must be reported if results enter the manuscript.
