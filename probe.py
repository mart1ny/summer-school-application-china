"""
probe.py - Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that predicts whether a
response is truthful (0) or hallucinated (1). Called from ``solution.py`` via
``evaluate.run_evaluation``. The public methods must keep their signatures.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


class HallucinationProbe(nn.Module):
    """Deterministic linear probe for high-dimensional hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._classifier: LogisticRegression | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5
        self._coef: torch.Tensor | None = None
        self._intercept: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for already standardized feature tensors."""
        if self._coef is None or self._intercept is None:
            raise RuntimeError(
                "Classifier has not been fitted yet. Call fit() before forward()."
            )
        return x @ self._coef.to(x.device) + self._intercept.to(x.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit a standardized L2-regularized logistic regression probe."""
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        y = y.astype(int)
        X_scaled = self._scaler.fit_transform(X)

        self._classifier = LogisticRegression(
            C=0.25,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        self._classifier.fit(X_scaled, y)

        self._coef = torch.from_numpy(self._classifier.coef_.reshape(-1)).float()
        self._intercept = torch.tensor(float(self._classifier.intercept_[0]))

        # ``solution.py`` fits the final probe without a validation set, so use
        # a training-only threshold as the reproducible default for predictions.
        train_probs = self._predict_positive_proba_scaled(X_scaled)
        self._threshold = self._best_threshold(train_probs, y)

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on validation labels for accuracy."""
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_threshold(probs, y_val.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with column 1 equal to P(label=1)."""
        if self._classifier is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        prob_pos = self._predict_positive_proba_scaled(X_scaled)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def _predict_positive_proba_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return P(label=1) for already standardized features."""
        if self._classifier is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        proba = self._classifier.predict_proba(X_scaled)
        classes = list(self._classifier.classes_)
        if 1 not in classes:
            return np.zeros(X_scaled.shape[0], dtype=float)
        return proba[:, classes.index(1)]

    @staticmethod
    def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Select a threshold using only the labels available to this fit."""
        candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 91)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if (accuracy > best_accuracy) or (
                accuracy == best_accuracy and f1 > best_f1
            ):
                best_accuracy = accuracy
                best_f1 = f1
                best_threshold = float(threshold)

        return best_threshold
