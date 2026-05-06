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
| 1 | `experiment/stage-1-strong-linear-baseline` | selected layers, mean/last/max pooling | LogisticRegression | TBD | TBD | TBD | TBD | strong reproducible baseline |
| 2 | `main` | pooling + trajectory + spectral features | LogisticRegression | TBD | TBD | TBD | TBD | advanced research candidate |

## Planned Next Experiments

### Layer-Wise Probing

Train identical probes on individual layers to understand where hallucination
signal appears inside the transformer.

Expected report table:

| Layer | Pooling | Avg Val Accuracy | Avg Val F1 | Avg Val AUROC |
|---:|---|---:|---:|---:|
| TBD | mean + last | TBD | TBD | TBD |

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
