"""
aggregation.py - Layer-wise aggregation and research features.

This stage exposes separate last-token representations for selected
transformer layers, followed by compact trajectory and spectral scalars.
``probe.py`` consumes the layer blocks as an ensemble rather than as one large
flat vector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


SELECTED_LAYERS = (-12, -8, -4, -2, -1, 8, 12, 16, 20)
TRAJECTORY_LAYERS = (-12, -8, -4, -2, -1)
SPECTRAL_LAYERS = (-12, -8, -4, -1)
MAX_SPECTRAL_TOKENS = 96
EPS = 1e-6


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return layer-wise last-token blocks plus compact research scalars."""
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)

    real_positions = mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())

    layer_vectors: list[torch.Tensor] = []
    trajectory_reps: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        vec = hidden_states[layer_idx, last_pos].float()
        layer_vectors.append(vec)
        if layer_idx in TRAJECTORY_LAYERS:
            trajectory_reps.append(vec)

    trajectory = _trajectory_features(torch.stack(trajectory_reps))
    spectral = _spectral_features(hidden_states, mask)
    seq_len = mask.sum().to(dtype=hidden_states.dtype).float()
    length_features = torch.stack(
        [
            seq_len / 512.0,
            torch.log1p(seq_len) / torch.log(torch.tensor(512.0, device=seq_len.device)),
        ]
    )

    return torch.cat([*layer_vectors, trajectory, spectral, length_features], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep optional geometric hook deterministic and compact."""
    del hidden_states, attention_mask
    return torch.zeros(0)


def _trajectory_features(reps: torch.Tensor) -> torch.Tensor:
    """Stability of last-token representation across selected layers."""
    deltas = reps[1:] - reps[:-1]
    consecutive_cos = F.cosine_similarity(reps[1:], reps[:-1], dim=1)
    delta_norms = torch.linalg.vector_norm(deltas, dim=1)
    endpoint_cos = F.cosine_similarity(reps[-1], reps[0], dim=0)
    endpoint_delta = torch.linalg.vector_norm(reps[-1] - reps[0])
    curvature = F.cosine_similarity(deltas[1:], deltas[:-1], dim=1)

    return torch.stack(
        [
            consecutive_cos.mean(),
            consecutive_cos.min(),
            consecutive_cos.max(),
            delta_norms.mean(),
            delta_norms.std(unbiased=False),
            delta_norms.sum(),
            endpoint_cos,
            endpoint_delta,
            curvature.mean(),
            curvature.min(),
        ]
    ).float()


def _spectral_features(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """EigenScore-inspired token covariance spectrum for selected layers."""
    features: list[torch.Tensor] = []
    device = hidden_states.device

    for layer_idx in SPECTRAL_LAYERS:
        tokens = hidden_states[layer_idx][mask].float()
        if tokens.size(0) > MAX_SPECTRAL_TOKENS:
            positions = torch.linspace(
                0,
                tokens.size(0) - 1,
                MAX_SPECTRAL_TOKENS,
                device=device,
            ).round().long()
            tokens = tokens[positions]

        centered = tokens - tokens.mean(dim=0, keepdim=True)
        denom = max(int(centered.size(0) - 1), 1)
        gram = (centered @ centered.T / denom).cpu()
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(EPS).flip(0)
        total = eigvals.sum().clamp_min(EPS)
        probs = eigvals / total
        entropy = -(probs * probs.log()).sum()
        effective_rank = entropy.exp()
        top = eigvals[0]
        second = eigvals[1] if eigvals.numel() > 1 else eigvals[0]
        tail = eigvals[-1]
        participation_ratio = total.square() / eigvals.square().sum().clamp_min(EPS)

        features.append(
            torch.stack(
                [
                    top.log(),
                    total.log(),
                    entropy,
                    effective_rank.log(),
                    (top / second.clamp_min(EPS)).log(),
                    (top / tail.clamp_min(EPS)).log(),
                    participation_ratio.log(),
                ]
            ).to(device=device)
        )

    return torch.cat(features).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
