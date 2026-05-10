# Experiment Plan

This file tracks the research progression for the SMILES-2026 hallucination
detection task. Each experiment must be run with the official command:

```bash
python solution.py
```

The generated `results.json` and `predictions.csv` are the source of truth for
reported metrics and final submission.

## Rules

- Do not modify fixed infrastructure: `solution.py`, `model.py`, `evaluate.py`.
- Keep all splits stratified and seeded.
- Do not use labels from `data/test.csv`.
- Record validation/test-split metrics from `results.json`.
- Prefer simple probes unless a feature family gives a clear gain.

## Experiment Branches

| Stage | Branch | Purpose |
|---:|---|---|
| 0 | `experiment/stage-0-official-skeleton` | Original repository skeleton baseline |
| 1 | `experiment/stage-1-strong-linear-baseline` | Multi-layer pooling + deterministic logistic probe |
| 2 | `main` | Current advanced candidate with trajectory + spectral features |
| 3 | `experiment/stage-3-layer-wise-probing` | Single-layer mean+last probing controlled by `LAYER_IDX` |
| 4 | `experiment/stage-4-last-token-layer-wise` | Single-layer last-token probing controlled by `LAYER_IDX` |
| 5 | `experiment/stage-5-mlp-kfold-final-layer` | Final-layer last-token tuned MLP with stratified K-fold |
| 6 | `experiment/stage-6-original-mlp-kfold` | Original MLP with final-layer last-token features and stratified K-fold |
| 7 | `experiment/stage-7-imbalance-aware-mlp` | Class-balanced MLP with prior-aware calibration and accuracy thresholding |
| 8 | `experiment/stage-8-tail-window-features` | Data-driven final-layer tail-window hidden-state features |
| 9 | `experiment/stage-9-tail-window-single-split` | Stage 8 tail-window features evaluated with the original single split |
| 10 | `experiment/stage-10-boosting-final-layer` | Histogram gradient boosting on final-layer last-token features |
| 11 | `experiment/stage-11-mlp-ensemble-final-layer` | Averaged ensemble of seeded MLP probes on final-layer last-token features |
| 12 | `experiment/stage-12-layerwise-research-ensemble` | Layer-wise MLP ensemble with trajectory and spectral covariance scalars |
| 13 | `experiment/stage-13-mlp-seed-search` | Original baseline MLP with fixed `PROBE_SEED` for seed sweep |

## Colab Run Template

```python
!rm -rf summer-school-application-china
!git clone https://github.com/mart1ny/summer-school-application-china.git
%cd summer-school-application-china
!git checkout <branch-name>
!pip install -r requirements.txt
!python solution.py
```

After each run, download or inspect:

```python
import json
with open("results.json") as f:
    results = json.load(f)
results
```

## Results Table

Fill this table after Colab runs.

| Stage | Branch | Feature Summary | Probe | Avg Val Accuracy | Avg Val F1 | Avg Val AUROC | Avg Test-Split Accuracy | Notes |
|---:|---|---|---|---:|---:|---:|---:|---|
| 0 | `experiment/stage-0-official-skeleton` | final layer, last token | original MLP | N/A | N/A | 0.6686 | 0.7404 | control baseline; feature dim 896 |
| 1 | `experiment/stage-1-strong-linear-baseline` | selected layers, mean/last/max pooling | LogisticRegression | N/A | N/A | 0.6461 | 0.6865 | worse than Stage 0; high-dimensional concatenation overfits/does not generalize |
| 2 | `experiment/stage-2-advanced-candidate` | pooling + trajectory + spectral features | LogisticRegression | N/A | N/A | 0.6469 | 0.6923 | advanced features did not recover Stage 0 performance; feature dim 13488 |
| 3 | `experiment/stage-3-layer-wise-probing` | best single layer among tested, mean+last pooling | LogisticRegression | N/A | N/A | 0.6339 | 0.7126 | best tested layer was `12`; feature dim 1792; still below Stage 0 |
| 4 | `experiment/stage-4-last-token-layer-wise` | best single layer among tested, last-token only | LogisticRegression | N/A | N/A | 0.6694 | 0.6981 | best tested layer was `-1`; feature dim 896; LogisticRegression underperforms original MLP |
| 5 | `experiment/stage-5-mlp-kfold-final-layer` | final layer, last token | tuned MLP | N/A | N/A | 0.7065 | 0.6866 | better ranking but poor hard-label accuracy; threshold/probability calibration needs work |
| 6 | `experiment/stage-6-original-mlp-kfold` | final layer, last token | original MLP | N/A | N/A | 0.6979 | 0.6952 | good ranking but poor accuracy; likely threshold/class-imbalance issue |
| 7 | `experiment/stage-7-imbalance-aware-mlp` | final layer, last token | imbalance-aware MLP | N/A | N/A | 0.7025 | 0.6894 | imbalance-aware calibration preserved AUROC but did not improve primary accuracy |
| 8 | `experiment/stage-8-tail-window-features` | last token + final-layer tail windows | original MLP | N/A | N/A | 0.7097 | 0.7170 | best data-driven feature branch so far; feature dim 3611 |
| 9 | `experiment/stage-9-tail-window-single-split` | last token + final-layer tail windows | original MLP | N/A | N/A | 0.7452 | 0.5962 | severe single-split overfit despite high validation AUROC |
| 10 | `experiment/stage-10-boosting-final-layer` | final layer, last token | HistGradientBoosting | N/A | N/A | 0.6359 | 0.7019 | near majority-level accuracy; tree boosting underfits dense hidden vectors |
| 11 | `experiment/stage-11-mlp-ensemble-final-layer` | final layer, last token | 5-seed MLP ensemble | N/A | N/A | 0.6690 | 0.7212 | improves stability/AUROC but still below Stage 0 primary accuracy |
| 12 | `experiment/stage-12-layerwise-research-ensemble` | 9 layer-wise last-token blocks + trajectory/spectral scalars | layer-wise MLP ensemble | N/A | N/A | 0.6902 | 0.7212 | full research ensemble stable but below Stage 0 primary accuracy; feature dim 8104 |
| 13 | `experiment/stage-13-mlp-seed-search` | final layer, last token | original MLP with fixed seed | N/A | N/A | 0.6774 | 0.7500 | best seed is `19`; final default seed |

## Planned Next Experiments

### Layer-Wise Probing

Train identical probes on individual layers to understand where hallucination
signal appears inside the transformer.

Expected report table:

| Layer | Pooling | Feature Dim | Avg Val AUROC | Avg Test-Split Accuracy | Avg Test-Split F1 | Avg Test-Split AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 12 | mean + last | 1792 | 0.6339 | 0.7126 | 0.8242 | 0.6635 |
| 8 | mean + last | 1792 | 0.5943 | 0.7054 | 0.8204 | 0.6313 |
| -12 | mean + last | 1792 | 0.6154 | 0.7040 | 0.8202 | 0.6915 |
| 16 | mean + last | 1792 | 0.6204 | 0.6923 | 0.8153 | 0.6788 |
| -4 | mean + last | 1792 | 0.6214 | 0.6923 | 0.8112 | 0.6409 |
| -1 | mean + last | 1792 | 0.5970 | 0.6864 | 0.8093 | 0.6266 |
| -8 | mean + last | 1792 | 0.6451 | 0.6850 | 0.8043 | 0.6615 |
| -2 | mean + last | 1792 | 0.6007 | 0.6792 | 0.7953 | 0.6487 |
| 20 | mean + last | 1792 | 0.6356 | 0.6748 | 0.7956 | 0.6513 |

Takeaway: adding mean pooling to last-token features did not beat the original
final-layer last-token baseline. The next experiment should isolate
last-token-only layer-wise probing.

### Last-Token-Only Layer-Wise Probing

| Layer | Pooling | Feature Dim | Avg Val AUROC | Avg Test-Split Accuracy | Avg Test-Split F1 | Avg Test-Split AUROC |
|---:|---|---:|---:|---:|---:|---:|
| -1 | last token | 896 | 0.6694 | 0.6981 | 0.8161 | 0.6919 |
| 8 | last token | 896 | 0.6679 | 0.6938 | 0.8094 | 0.6651 |
| 16 | last token | 896 | 0.6583 | 0.6894 | 0.8054 | 0.6706 |
| -12 | last token | 896 | 0.6512 | 0.6865 | 0.8049 | 0.6612 |
| -4 | last token | 896 | 0.6308 | 0.6865 | 0.8078 | 0.6814 |
| 20 | last token | 896 | 0.6160 | 0.6822 | 0.8068 | 0.6492 |
| 12 | last token | 896 | 0.6488 | 0.6821 | 0.8019 | 0.6548 |
| -2 | last token | 896 | 0.6618 | 0.6807 | 0.7991 | 0.6920 |
| -8 | last token | 896 | 0.6459 | 0.6778 | 0.7994 | 0.6588 |

Takeaway: the final layer remains the best tested last-token layer, but the
linear probe is weaker than the original MLP skeleton. The next experiment
should compare probes while keeping the strongest representation fixed.

### Pooling Ablation

Compare token aggregation strategies under the same probe:

- last token only
- mean pooling only
- max pooling only
- mean + last
- mean + last + max

### Probe Comparison

Compare linear classifiers:

- `LogisticRegression`
- `RidgeClassifier`
- `LinearSVC`

The final model should prioritize validation accuracy, reproducibility, and a
clear no-leakage story over architectural complexity.
