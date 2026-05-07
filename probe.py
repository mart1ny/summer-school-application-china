"""
probe.py - Hallucination probe classifier (student-implemented).

Stage 10 experiment: gradient-boosted decision trees over the compact
final-layer last-token representation. This keeps the original feature
extraction and split strategy, changing only the classifier.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


RANDOM_STATE = 42


class HallucinationProbe(nn.Module):
    """Boosting probe compatible with the official evaluation interface."""

    def __init__(self) -> None:
        super().__init__()
        self._classifier: HistGradientBoostingClassifier | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch forward is unused for boosting, kept for nn.Module compatibility."""
        raise RuntimeError("HistGradientBoostingClassifier does not support forward().")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit a class-balanced histogram gradient boosting classifier."""
        y = y.astype(int)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        sample_weight = np.where(
            y == 1,
            len(y) / (2.0 * max(n_pos, 1)),
            len(y) / (2.0 * max(n_neg, 1)),
        )

        self._classifier = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            l2_regularization=0.15,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )
        self._classifier.fit(X, y, sample_weight=sample_weight)

        # Final predictions in solution.py are made without a validation set, so
        # use a training-only threshold as the deterministic fallback.
        self._threshold = self._best_threshold(self.predict_proba(X)[:, 1], y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune threshold on validation labels for primary accuracy."""
        self._threshold = self._best_threshold(
            self.predict_proba(X_val)[:, 1],
            y_val.astype(int),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with column 1 equal to P(label=1)."""
        if self._classifier is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        proba = self._classifier.predict_proba(X)
        classes = list(self._classifier.classes_)
        if 1 not in classes:
            prob_pos = np.zeros(X.shape[0], dtype=float)
        else:
            prob_pos = proba[:, classes.index(1)]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    @staticmethod
    def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Choose threshold for accuracy, with balanced-accuracy and F1 tie-breaks."""
        candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 91)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_balanced_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if (
                accuracy > best_accuracy
                or (
                    accuracy == best_accuracy
                    and balanced_accuracy > best_balanced_accuracy
                )
                or (
                    accuracy == best_accuracy
                    and balanced_accuracy == best_balanced_accuracy
                    and f1 > best_f1
                )
            ):
                best_accuracy = accuracy
                best_balanced_accuracy = balanced_accuracy
                best_f1 = f1
                best_threshold = float(threshold)

        return best_threshold
