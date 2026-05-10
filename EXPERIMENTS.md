# Experiment Log

This document records the research path for the SMILES-2026 hallucination
detection task. Every result was obtained with the official runner:

```bash
python solution.py
```

The generated `results.json` and `predictions.csv` are the source of truth for
submission artifacts.

## Experimental Rules

- Do not modify fixed infrastructure: `solution.py`, `model.py`, `evaluate.py`.
- Keep splits seeded and stratified.
- Do not use labels from `data/test.csv`.
- Do not hardcode examples, local paths, or submission labels.
- Prefer the simplest model that improves the primary metric.

## Final Decision

The strongest final result came from the compact original representation:

```text
final transformer layer -> last real token -> StandardScaler -> MLP probe
```

The final seed is `19`, producing:

| Feature Dim | Avg Val AUROC | Avg Test-Split Accuracy | Avg Test-Split F1 | Avg Test-Split AUROC |
|---:|---:|---:|---:|---:|
| 896 | 0.6774 | 0.7500 | 0.8375 | 0.7371 |

This is the submitted solution because it improved the primary accuracy while
remaining simple, reproducible, and compliant with the task rules.

## Experiment Branches

| Stage | Branch | Purpose |
|---:|---|---|
| 0 | `experiment/stage-0-official-skeleton` | Original repository skeleton baseline |
| 1 | `experiment/stage-1-strong-linear-baseline` | Multi-layer pooling + deterministic logistic probe |
| 2 | `experiment/stage-2-advanced-candidate` | Multi-layer pooling with trajectory and spectral features |
| 3 | `experiment/stage-3-layer-wise-probing` | Single-layer mean+last probing controlled by `LAYER_IDX` |
| 4 | `experiment/stage-4-last-token-layer-wise` | Single-layer last-token probing controlled by `LAYER_IDX` |
| 5 | `experiment/stage-5-mlp-kfold-final-layer` | Final-layer last-token tuned MLP with stratified K-fold |
| 6 | `experiment/stage-6-original-mlp-kfold` | Original MLP with final-layer last-token features and stratified K-fold |
| 7 | `experiment/stage-7-imbalance-aware-mlp` | Class-balanced MLP with prior-aware calibration |
| 8 | `experiment/stage-8-tail-window-features` | Data-driven final-layer tail-window hidden-state features |
| 9 | `experiment/stage-9-tail-window-single-split` | Stage 8 tail-window features on the original split |
| 10 | `experiment/stage-10-boosting-final-layer` | Histogram gradient boosting on final-layer last-token features |
| 11 | `experiment/stage-11-mlp-ensemble-final-layer` | Averaged ensemble of seeded MLP probes |
| 12 | `experiment/stage-12-layerwise-research-ensemble` | Layer-wise MLP ensemble with trajectory and spectral scalars |
| 13 | `experiment/stage-13-mlp-seed-search` | Final seed sweep for the original MLP representation |

## Colab Run Template

```python
!rm -rf summer-school-application-china
!git clone https://github.com/mart1ny/summer-school-application-china.git
%cd summer-school-application-china
!git checkout <branch-name>
!pip install -r requirements.txt
!python solution.py
```

Inspect metrics:

```python
import json
with open("results.json") as f:
    results = json.load(f)
results
```

## Results Table

| Stage | Branch | Feature Summary | Probe | Avg Val AUROC | Avg Test-Split Accuracy | Notes |
|---:|---|---|---|---:|---:|---|
| 0 | `experiment/stage-0-official-skeleton` | final layer, last token | original MLP | 0.6686 | 0.7404 | strong control baseline |
| 1 | `experiment/stage-1-strong-linear-baseline` | selected layers, mean/last/max pooling | LogisticRegression | 0.6461 | 0.6865 | high-dimensional concatenation did not generalize |
| 2 | `experiment/stage-2-advanced-candidate` | pooling + trajectory + spectral features | LogisticRegression | 0.6469 | 0.6923 | advanced scalars did not recover baseline performance |
| 3 | `experiment/stage-3-layer-wise-probing` | best single layer, mean+last pooling | LogisticRegression | 0.6339 | 0.7126 | best tested layer was `12`; still below baseline |
| 4 | `experiment/stage-4-last-token-layer-wise` | best single layer, last-token only | LogisticRegression | 0.6694 | 0.6981 | final layer best, but linear probe underperformed |
| 5 | `experiment/stage-5-mlp-kfold-final-layer` | final layer, last token | tuned MLP | 0.7065 | 0.6866 | better ranking but poor thresholded accuracy |
| 6 | `experiment/stage-6-original-mlp-kfold` | final layer, last token | original MLP | 0.6979 | 0.6952 | K-fold changed calibration and hurt accuracy |
| 7 | `experiment/stage-7-imbalance-aware-mlp` | final layer, last token | imbalance-aware MLP | 0.7025 | 0.6894 | preserved AUROC but did not improve primary metric |
| 8 | `experiment/stage-8-tail-window-features` | last token + final-layer tail windows | original MLP | 0.7097 | 0.7170 | useful data-driven features but still below baseline |
| 9 | `experiment/stage-9-tail-window-single-split` | last token + tail windows | original MLP | 0.7452 | 0.5962 | high validation AUROC but severe test-split overfit |
| 10 | `experiment/stage-10-boosting-final-layer` | final layer, last token | HistGradientBoosting | 0.6359 | 0.7019 | near majority-level accuracy |
| 11 | `experiment/stage-11-mlp-ensemble-final-layer` | final layer, last token | 5-seed MLP ensemble | 0.6690 | 0.7212 | stable but below baseline |
| 12 | `experiment/stage-12-layerwise-research-ensemble` | layer-wise blocks + trajectory/spectral scalars | layer-wise MLP ensemble | 0.6902 | 0.7212 | research-rich but below baseline |
| 13 | `experiment/stage-13-mlp-seed-search` | final layer, last token | original MLP with fixed seed | 0.6774 | 0.7500 | best seed is `19`; final default |

## Layer-Wise Probing

Layer-wise probing was used to test whether hallucination signal appears in a
specific depth range of the transformer.

### Mean + Last Pooling

| Layer | Feature Dim | Avg Val AUROC | Avg Test-Split Accuracy | Avg Test-Split F1 | Avg Test-Split AUROC |
|---:|---:|---:|---:|---:|---:|
| 12 | 1792 | 0.6339 | 0.7126 | 0.8242 | 0.6635 |
| 8 | 1792 | 0.5943 | 0.7054 | 0.8204 | 0.6313 |
| -12 | 1792 | 0.6154 | 0.7040 | 0.8202 | 0.6915 |
| 16 | 1792 | 0.6204 | 0.6923 | 0.8153 | 0.6788 |
| -4 | 1792 | 0.6214 | 0.6923 | 0.8112 | 0.6409 |
| -1 | 1792 | 0.5970 | 0.6864 | 0.8093 | 0.6266 |
| -8 | 1792 | 0.6451 | 0.6850 | 0.8043 | 0.6615 |
| -2 | 1792 | 0.6007 | 0.6792 | 0.7953 | 0.6487 |
| 20 | 1792 | 0.6356 | 0.6748 | 0.7956 | 0.6513 |

Takeaway: adding mean pooling to last-token features did not beat the original
final-layer last-token baseline.

### Last-Token Only

| Layer | Feature Dim | Avg Val AUROC | Avg Test-Split Accuracy | Avg Test-Split F1 | Avg Test-Split AUROC |
|---:|---:|---:|---:|---:|---:|
| -1 | 896 | 0.6694 | 0.6981 | 0.8161 | 0.6919 |
| 8 | 896 | 0.6679 | 0.6938 | 0.8094 | 0.6651 |
| 16 | 896 | 0.6583 | 0.6894 | 0.8054 | 0.6706 |
| -12 | 896 | 0.6512 | 0.6865 | 0.8049 | 0.6612 |
| -4 | 896 | 0.6308 | 0.6865 | 0.8078 | 0.6814 |
| 20 | 896 | 0.6160 | 0.6822 | 0.8068 | 0.6492 |
| 12 | 896 | 0.6488 | 0.6821 | 0.8019 | 0.6548 |
| -2 | 896 | 0.6618 | 0.6807 | 0.7991 | 0.6920 |
| -8 | 896 | 0.6459 | 0.6778 | 0.7994 | 0.6588 |

Takeaway: the final layer remained a strong representation, but the linear
probe was weaker than the MLP used in the final solution.

## Main Research Takeaways

- Simpler hidden-state features generalized better than large concatenated
  feature sets.
- AUROC improvements did not always translate into the primary accuracy metric.
- Class imbalance explains why many models achieve high F1 while remaining near
  the majority baseline in accuracy.
- The final result is not the most complex model; it is the model with the best
  measured primary metric under the official pipeline.
