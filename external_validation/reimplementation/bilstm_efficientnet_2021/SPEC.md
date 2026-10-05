# Paper-Fidelity Specification — Nguyen et al. 2021

Paper: **Detecting Website Defacement Attacks using Web-page Text and Image Features** (IJACSA 2021)

## Explicitly specified by the paper
- Inputs: current webpage text and current webpage screenshot.
- Text tokenization: TensorFlow Tokenizer-style word indexing.
- Sequence: first 128 consecutive words.
- Text model: Embedding -> SpatialDropout1D -> Bidirectional LSTM -> Dense(2)/Softmax.
- Figure 9 fixes embedding width at 64 and BiLSTM output width at 128.
- Image preprocessing: resize to 224x224, pixel values in [0,1].
- Image model: ImageNet-style EfficientNetB0 transfer learning, final FC removed.
- Figure 10: EfficientNetB0 output 1280 -> BatchNorm -> Dense(128) -> BatchNorm -> Dense(2).
- Fusion: soft voting, average text and image prediction probabilities.
- Original paper split: random 80/20 with class ratio preserved.
- Original dataset: 57,134 benign + 39,100 defaced webpages.

## Protocol adaptations required for a fair DI-MMGN comparison
- Use frozen LWDED-v4 family-blocked pretrain/val/test splits rather than the paper's random 80/20 split.
- Use the current snapshot S_t only; never expose S_(t-1) or delta.
- Fit the word vocabulary on the pretrain split only.
- Select checkpoints on the frozen validation split; evaluate test only after selection.
- Use seeds 42/43/44, matching DI-MMGN.

## Not specified / fidelity assumptions
- SpatialDropout rate: not stated. Default implementation assumption: 0.2.
- Vocabulary cap / OOV handling: not stated. Default: full pretrain vocabulary with <OOV>, min frequency 1.
- LSTM recurrent dropout: not stated. Default: 0.
- Optimizer, LR, epochs, batch size: not stated in the 2021 paper. For fair reimplementation, expose these in config and report them; do not attribute them to the authors.
- EfficientNet exact pretrained weight revision: not stated. Use torchvision EfficientNet_B0_Weights.IMAGENET1K_V1 and record this as an implementation choice.

## Fidelity label
**High-fidelity architectural reimplementation with explicitly documented training/framework assumptions.**
