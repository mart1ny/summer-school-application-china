"""
probe.py - Layer-wise ensemble probe.

The feature vector starts with nine 896-dimensional layer blocks.  Each block
gets its own MLP probe, and their probabilities are averaged with an auxiliary
scalar-feature logistic probe.  This implements layer-wise probing and
multi-layer ensembling without fitting one unstable classifier on the full
concatenation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


N_LAYER_BLOCKS = 9
HIDDEN_DIM = 896
LAYER_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139)
SCALAR_WEIGHT = 0.20


class _LayerMLP(nn.Module):
    """Small MLP used as one layer-specific probe."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HallucinationProbe(nn.Module):
    """Layer-wise MLP ensemble with validation threshold tuning."""

    def __init__(self) -> None:
        super().__init__()
        self._layer_models = nn.ModuleList()
        self._layer_scalers: list[StandardScaler] = []
        self._scalar_scaler = StandardScaler()
        self._scalar_model: LogisticRegression | None = None
        self._threshold: float = 0.5
        self._n_layer_blocks: int = N_LAYER_BLOCKS
        self._hidden_dim: int = HIDDEN_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch forward is not used by sklearn-style evaluation."""
        raise RuntimeError("Use predict_proba/predict for the layer-wise ensemble.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit separate probes for each layer block and optional scalar block."""
        y = y.astype(int)
        self._infer_layout(X.shape[1])

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        y_t = torch.from_numpy(y.astype(np.float32))

        self._layer_models = nn.ModuleList()
        self._layer_scalers = []
        for block_id in range(self._n_layer_blocks):
            start = block_id * self._hidden_dim
            end = start + self._hidden_dim
            scaler = StandardScaler()
            X_block = scaler.fit_transform(X[:, start:end])
            X_t = torch.from_numpy(X_block).float()

            seed = LAYER_SEEDS[block_id % len(LAYER_SEEDS)]
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = _LayerMLP(self._hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
            model.train()
            for _ in range(160):
                optimizer.zero_grad()
                logits = model(X_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
            model.eval()

            self._layer_scalers.append(scaler)
            self._layer_models.append(model)

        scalar_X = self._scalar_features(X)
        if scalar_X.shape[1] > 0:
            sample_weight = np.where(
                y == 1,
                len(y) / (2.0 * max(n_pos, 1)),
                len(y) / (2.0 * max(n_neg, 1)),
            )
            scalar_scaled = self._scalar_scaler.fit_transform(scalar_X)
            self._scalar_model = LogisticRegression(
                C=0.5,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )
            self._scalar_model.fit(scalar_scaled, y, sample_weight=sample_weight)

        self._threshold = self._best_threshold(self.predict_proba(X)[:, 1], y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on validation labels for primary accuracy."""
        self._threshold = self._best_threshold(
            self.predict_proba(X_val)[:, 1],
            y_val.astype(int),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average layer-wise probabilities and scalar-feature probability."""
        if len(self._layer_models) == 0:
            raise RuntimeError("Models have not been fitted yet. Call fit() first.")

        layer_probs = []
        for block_id, (model, scaler) in enumerate(
            zip(self._layer_models, self._layer_scalers)
        ):
            start = block_id * self._hidden_dim
            end = start + self._hidden_dim
            X_block = scaler.transform(X[:, start:end])
            X_t = torch.from_numpy(X_block).float()
            with torch.no_grad():
                layer_probs.append(torch.sigmoid(model(X_t)).numpy())

        prob_pos = np.mean(np.stack(layer_probs, axis=0), axis=0)

        scalar_X = self._scalar_features(X)
        if self._scalar_model is not None and scalar_X.shape[1] > 0:
            scalar_scaled = self._scalar_scaler.transform(scalar_X)
            scalar_prob = self._scalar_model.predict_proba(scalar_scaled)[:, 1]
            prob_pos = (1.0 - SCALAR_WEIGHT) * prob_pos + SCALAR_WEIGHT * scalar_prob

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def _infer_layout(self, feature_dim: int) -> None:
        """Infer layer block layout while keeping Qwen's hidden size explicit."""
        if feature_dim < N_LAYER_BLOCKS * HIDDEN_DIM:
            raise ValueError(
                f"Expected at least {N_LAYER_BLOCKS * HIDDEN_DIM} features, got {feature_dim}."
            )
        self._n_layer_blocks = N_LAYER_BLOCKS
        self._hidden_dim = HIDDEN_DIM

    def _scalar_features(self, X: np.ndarray) -> np.ndarray:
        """Return trajectory/spectral scalar tail after layer blocks."""
        start = self._n_layer_blocks * self._hidden_dim
        return X[:, start:]

    @staticmethod
    def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Choose threshold for accuracy with balanced-accuracy tie-breaks."""
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
