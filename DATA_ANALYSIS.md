# Data Analysis Notes

This document summarizes exploratory analysis of `data/dataset.csv` and
`data/test.csv`. It is used to guide feature engineering experiments without
touching fixed infrastructure files.

## Dataset Shape and Class Balance

Training set:

- rows: `689`
- truthful (`label=0`): `206`
- hallucinated (`label=1`): `483`
- majority-class accuracy: `0.7010`

Test set:

- rows: `100`
- labels are null

The dataset is imbalanced: hallucinated answers are about 70% of the labelled
training data. This makes raw accuracy easy to inflate with a majority-class
bias, so threshold calibration and class-specific behavior matter.

## Duplicate and Leakage Checks

Training duplicates:

- duplicate prompts: `0`
- duplicate responses: `14`

Test duplicates:

- duplicate prompts: `0`
- duplicate responses: `0`

Train/test overlap:

- prompt overlap: `0`
- response overlap: `2`

Important duplicate responses in train:

| Response Preview | Count | Labels |
|---|---:|---|
| `You are a helpful assistant.moid<|endoftext|>` | 9 | all `1` |
| `Unable to answer based on given context.<|endoftext|>` | 5 | all `0` |
| `You are a helpful assistant.moid.<|endoftext|>` | 3 | all `1` |

Two exact train/test response overlaps exist. This is useful as a data artifact
observation, but the solution should not hardcode response strings or use
`data/test.csv` labels. Any final solution must remain general.

## Strongest Label Correlations

The clearest signals are response-side behavior, not prompt length.

| Feature | Truthful Mean | Hallucinated Mean | Correlation With `label=1` |
|---|---:|---:|---:|
| `response_words` | 67.75 | 131.02 | `+0.2409` |
| `response_chars` | 417.67 | 790.41 | `+0.2367` |
| `response_sentences` | 4.71 | 8.69 | `+0.2161` |
| `response_commas` | 3.08 | 6.73 | `+0.2249` |
| `response_digits` | 3.22 | 7.92 | `+0.1383` |
| `overlap_per_response_word` | 0.3420 | 0.1977 | `-0.4095` |

Interpretation:

- hallucinated responses are substantially longer;
- hallucinated responses have lower normalized lexical overlap with context;
- prompt/context/question lengths are weak predictors by themselves.

## Binned Label Rates

### Response Length

| Response Word Quintile | Word Range | Hallucination Rate |
|---:|---:|---:|
| 0 | 2-18 | 0.623 |
| 1 | 18-43 | 0.572 |
| 2 | 43-88 | 0.650 |
| 3 | 88-206 | 0.761 |
| 4 | 207-453 | 0.899 |

The longest responses are hallucinated almost 90% of the time.

### Context Overlap Ratio

`overlap_per_response_word` is the number of response words of length at least
4 that also appear in the context, divided by response word count.

| Overlap Quintile | Ratio Range | Hallucination Rate |
|---:|---:|---:|
| 0 | 0.000-0.093 | 0.906 |
| 1 | 0.094-0.171 | 0.833 |
| 2 | 0.172-0.261 | 0.759 |
| 3 | 0.262-0.394 | 0.630 |
| 4 | 0.394-0.750 | 0.377 |

This is the strongest simple data signal: answers with low context overlap are
much more likely to be hallucinated.

## Simple Feature Baselines

Cross-validated logistic baselines on labelled data:

| Feature Set | CV Accuracy | CV F1 | CV AUROC |
|---|---:|---:|---:|
| Numeric structural/text features | 0.6821 | 0.7491 | 0.7214 |
| Response TF-IDF | 0.6589 | 0.7575 | 0.6549 |
| Prompt+response TF-IDF | 0.6372 | 0.7412 | 0.5858 |

Numeric response/context features alone have meaningful ranking power
(`AUROC=0.7214`) but mediocre hard-label accuracy. This matches the model
experiments: several probes rank examples reasonably but lose accuracy after
thresholding.

## Train/Test Distribution Shift

The unlabeled test set has slightly longer responses:

| Feature | Train Mean | Test Mean | Standardized Mean Diff |
|---|---:|---:|---:|
| `response_words` | 112.11 | 130.93 | `+0.156` |
| `response_chars` | 678.97 | 789.47 | `+0.153` |
| `overlap_per_response_word` | 0.2408 | 0.2279 | `-0.080` |

The shift is not huge, but it points in the hallucination-correlated direction:
test responses are longer and slightly lower-overlap on average.

## Implications for Feature Engineering

Direct lexical overlap is not available inside `aggregation.py`, because the
official function receives only hidden states and `attention_mask`, not raw text
or token ids. However, the data analysis suggests a hidden-state approximation:

- avoid mean pooling over the whole prompt+response;
- focus on the answer tail, since the response is appended after the prompt;
- use last-token features plus pooled features over the last `K` real tokens;
- add sequence-length scalar features from `attention_mask`.

Recommended Stage 8:

```text
final layer hidden states
+ last token
+ mean/max/std pooling over last 32 tokens
+ mean pooling over last 64 and 128 tokens
+ log sequence length / normalized sequence length
+ MLP probe
```

This targets the strongest data signals while staying within the allowed
participant files and avoiding raw-text leakage.
