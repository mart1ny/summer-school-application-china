"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


TAIL_WINDOWS = (16, 32, 64, 128)
EPS = 1e-6


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)

    layer = hidden_states[-1].float()          # (seq_len, hidden_dim)
    real_positions = mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())
    valid_tokens = layer[mask]

    last_token = layer[last_pos]

    # Data analysis showed that response-side behavior is much more predictive
    # than prompt-side length. Since the response is appended at the end, tail
    # windows approximate answer-only pooling without using raw text or token ids.
    tail_32 = valid_tokens[-min(32, valid_tokens.size(0)) :]
    tail_64 = valid_tokens[-min(64, valid_tokens.size(0)) :]

    tail_mean_32 = tail_32.mean(dim=0)
    tail_delta_32 = last_token - tail_mean_32
    tail_mean_64 = tail_64.mean(dim=0)

    scalar_features = _tail_scalar_features(valid_tokens, last_token)

    return torch.cat(
        [last_token, tail_mean_32, tail_delta_32, tail_mean_64, scalar_features],
        dim=0,
    ).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def _tail_scalar_features(
    valid_tokens: torch.Tensor,
    last_token: torch.Tensor,
) -> torch.Tensor:
    """Compact answer-tail statistics guided by the EDA length/overlap signals."""
    features: list[torch.Tensor] = []
    seq_len = torch.tensor(
        float(valid_tokens.size(0)),
        device=valid_tokens.device,
        dtype=valid_tokens.dtype,
    )
    max_len = torch.tensor(
        512.0,
        device=valid_tokens.device,
        dtype=valid_tokens.dtype,
    )
    features.extend([seq_len / max_len, torch.log1p(seq_len) / torch.log(max_len)])

    last_norm = torch.linalg.vector_norm(last_token).clamp_min(EPS)
    features.append(last_norm)

    for window in TAIL_WINDOWS:
        tail = valid_tokens[-min(window, valid_tokens.size(0)) :]
        mean = tail.mean(dim=0)
        norms = torch.linalg.vector_norm(tail, dim=1)
        features.extend(
            [
                torch.tensor(
                    float(tail.size(0)) / float(window),
                    device=valid_tokens.device,
                    dtype=valid_tokens.dtype,
                ),
                norms.mean(),
                norms.std(unbiased=False),
                norms.max(),
                F.cosine_similarity(last_token, mean, dim=0),
                torch.linalg.vector_norm(last_token - mean),
            ]
        )

    return torch.stack(features).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
