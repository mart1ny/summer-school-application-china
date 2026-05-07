"""
probe.py - Hallucination probe classifier (student-implemented).

Stage 11 experiment: an ensemble of lightweight MLP probes over the compact
final-layer last-token representation. The goal is to improve the strong
single-MLP baseline without increasing feature dimensionality.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


ENSEMBLE_SEEDS = (3, 11, 17, 29, 42)


class _SmallMLP(nn.Module):
    """Single hidden-layer probe matching the original skeleton capacity."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HallucinationProbe(nn.Module):
    """Averaged MLP ensemble compatible with the official evaluation interface."""

    def __init__(self) -> None:
        super().__init__()
        self._models = nn.ModuleList()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return average ensemble logits for compatibility with nn.Module."""
        if len(self._models) == 0:
            raise RuntimeError("Models have not been fitted yet. Call fit() first.")
        logits = torch.stack([model(x) for model in self._models], dim=0)
        return logits.mean(dim=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train several seeded MLPs and average their probabilities."""
        y = y.astype(int)
        X_scaled = self._scaler.fit_transform(X)

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._models = nn.ModuleList()
        for seed in ENSEMBLE_SEEDS:
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = _SmallMLP(X_scaled.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            for _ in range(200):
                optimizer.zero_grad()
                logits = model(X_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
            model.eval()
            self._models.append(model)

        # Final solution.py predictions do not call fit_hyperparameters, so keep
        # a deterministic training-only threshold fallback.
        self._threshold = self._best_threshold(self.predict_proba(X)[:, 1], y)
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on validation labels for accuracy."""
        self._threshold = self._best_threshold(
            self.predict_proba(X_val)[:, 1],
            y_val.astype(int),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probabilities."""
        if len(self._models) == 0:
            raise RuntimeError("Models have not been fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            probs = [torch.sigmoid(model(X_t)).numpy() for model in self._models]
        prob_pos = np.mean(np.stack(probs, axis=0), axis=0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    @staticmethod
    def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Choose threshold for primary accuracy with stable tie-breaks."""
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
