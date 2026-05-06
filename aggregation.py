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


SELECTED_LAYERS = (-1, -2, -4, -8, -12)


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

    real_positions = mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())

    features: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]          # (seq_len, hidden_dim)
        valid_tokens = layer[mask]                # (n_real_tokens, hidden_dim)

        mean_pool = valid_tokens.mean(dim=0)
        last_token = layer[last_pos]
        max_pool = valid_tokens.max(dim=0).values

        features.extend([mean_pool, last_token, max_pool])

    return torch.cat(features, dim=0).float()


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
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)

    features: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx][mask]
        token_norms = torch.linalg.vector_norm(layer, dim=1)
        features.extend(
            [
                token_norms.mean(),
                token_norms.std(unbiased=False),
                token_norms.max(),
            ]
        )

    seq_len = mask.sum().to(dtype=hidden_states.dtype)
    max_len = torch.tensor(mask.numel(), device=hidden_states.device, dtype=hidden_states.dtype)
    features.append(seq_len / max_len.clamp_min(1.0))

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
