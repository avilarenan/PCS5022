"""Leakage-safe residual-correlation evidence for selective Time-PEFT routing.

The public extractor deliberately accepts :class:`SupportView` rather than an
evaluation episode.  It performs exactly one frozen forecast and derives every
feature from the support residuals.  Correlations are computed per support
window, so adjacent examples are never joined by an artificial transition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from utility_peft.model import AdaptableForecaster
from utility_peft.types import EvidenceBundle, SupportView

FREQUENCY_ROUTING_FEATURES = (
    "residual_lag1_signed_autocorrelation",
    "residual_lag1_abs_autocorrelation",
    "residual_mean_signed_autocorrelation",
    "residual_mean_abs_autocorrelation",
    "residual_max_abs_autocorrelation",
    "residual_autocorrelation_nonstationarity",
    "residual_autocorrelation_decay",
)

CHANNEL_ROUTING_FEATURES = (
    "residual_mean_signed_channel_correlation",
    "residual_mean_abs_channel_correlation",
    "residual_channel_correlation_dispersion",
    "residual_correlation_nonstationarity",
    "residual_correlation_effective_rank",
    "residual_correlation_effective_rank_fraction",
    "residual_max_lagged_cross_correlation",
)


def extract_correlation_evidence(
    support: SupportView,
    model: AdaptableForecaster,
    *,
    device: str | torch.device = "cpu",
    max_lag: int = 8,
) -> EvidenceBundle:
    """Return support-only correlation evidence after one frozen model call.

    The model's training mode, parameter gradient flags, and original device
    are preserved.  Query tensors cannot be passed through this interface.
    Non-finite residual observations are ignored by the masked correlation
    calculations and never propagate to the returned feature values.
    """

    if type(support) is not SupportView:
        raise TypeError("Correlation evidence extraction accepts SupportView only")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")

    target_device = torch.device(device)
    first_parameter = next(model.parameters(), None)
    original_device = first_parameter.device if first_parameter is not None else target_device
    was_training = model.training
    original_flags = tuple(parameter.requires_grad for parameter in model.parameters())

    try:
        model.to(target_device)
        model.eval()
        x = support.x.to(target_device)
        y = support.y.to(target_device)
        mask = support.mask.to(target_device)
        with torch.no_grad():
            prediction = model.predict(x, mask, support.manifest.horizon)
        if prediction.shape != y.shape:
            raise RuntimeError(
                "Frozen forecast shape does not match the support target: "
                f"{tuple(prediction.shape)} != {tuple(y.shape)}"
            )
        residual = prediction.detach().float() - y.float()
        features = residual_correlation_features(residual, max_lag=max_lag)
    finally:
        model.train(was_training)
        model.to(original_device)
        for parameter, requires_grad in zip(model.parameters(), original_flags, strict=True):
            parameter.requires_grad_(requires_grad)

    manifest = support.manifest
    finite_residual = residual[torch.isfinite(residual)]
    if finite_residual.numel():
        frozen_mse = float(finite_residual.square().mean())
        frozen_mae = float(finite_residual.abs().mean())
        residual_std = float(finite_residual.std(unbiased=False))
    else:
        frozen_mse = frozen_mae = residual_std = 0.0
    features.update(
        {
            "channels": float(residual.shape[1]),
            "horizon": float(residual.shape[2]),
            "support_size": float(residual.shape[0]),
            "frozen_mse": frozen_mse,
            "frozen_mae": frozen_mae,
            "residual_std": residual_std,
            "residual_valid_fraction": float(torch.isfinite(residual).float().mean()),
            "horizon_lookback_ratio": manifest.horizon / max(support.x.shape[-1], 1),
        }
    )
    return EvidenceBundle.from_mapping(manifest.episode_id, _finite_feature_mapping(features))


def residual_correlation_features(residual: Tensor, *, max_lag: int = 8) -> dict[str, float]:
    """Compute vectorized windowed correlation features from ``[B, C, H]``.

    Each batch element is treated as a separate window.  Pairwise statistics
    use only jointly finite values.  A zero is returned when a statistic is not
    identifiable (for example, cross-channel correlation when ``C == 1`` or
    autocorrelation for a one-step horizon).
    """

    if residual.ndim != 3:
        raise ValueError("Residuals must have shape [batch, channels, horizon]")
    if residual.shape[0] == 0 or residual.shape[1] == 0 or residual.shape[2] == 0:
        raise ValueError("Residual dimensions must be non-empty")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")

    values = residual.detach().float()
    batch, channels, horizon = values.shape
    correlations = _windowed_cross_correlation(values, values)

    if channels > 1:
        off_diagonal = ~torch.eye(channels, dtype=torch.bool, device=values.device)
        windowed_pairs = correlations[:, off_diagonal]
        mean_signed_channel = windowed_pairs.mean()
        mean_abs_channel = windowed_pairs.abs().mean()
        channel_dispersion = windowed_pairs.abs().std(unbiased=False)
        if batch > 1:
            nonstationarity = windowed_pairs.std(dim=0, unbiased=False).mean()
        else:
            nonstationarity = values.new_zeros(())
    else:
        mean_signed_channel = values.new_zeros(())
        mean_abs_channel = values.new_zeros(())
        channel_dispersion = values.new_zeros(())
        nonstationarity = values.new_zeros(())

    effective_rank = _effective_correlation_rank(values)
    effective_rank_fraction = effective_rank / channels

    usable_lag = min(max_lag, horizon - 1)
    lagged_matrices: list[Tensor] = []
    autocorrelations: list[Tensor] = []
    for lag in range(1, usable_lag + 1):
        lagged = _windowed_cross_correlation(values[..., lag:], values[..., :-lag])
        lagged_matrices.append(lagged)
        autocorrelations.append(lagged.diagonal(dim1=-2, dim2=-1))

    if lagged_matrices and channels > 1:
        stacked_lagged = torch.stack(lagged_matrices, dim=0)
        off_diagonal = ~torch.eye(channels, dtype=torch.bool, device=values.device)
        max_lagged_cross = stacked_lagged[..., off_diagonal].abs().max()
    else:
        max_lagged_cross = values.new_zeros(())

    if autocorrelations:
        autocorrelation = torch.stack(autocorrelations, dim=0)  # [lag, B, C]
        absolute = autocorrelation.abs()
        lag1 = autocorrelation[0]
        mean_signed_autocorrelation = autocorrelation.mean()
        mean_abs_autocorrelation = absolute.mean()
        max_abs_autocorrelation = absolute.max()
        if batch > 1:
            per_window = absolute.mean(dim=(0, 2))
            autocorrelation_nonstationarity = per_window.std(unbiased=False)
        else:
            autocorrelation_nonstationarity = values.new_zeros(())
        autocorrelation_decay = absolute[0].mean() - absolute[-1].mean()
        lag1_signed = lag1.mean()
        lag1_abs = lag1.abs().mean()
    else:
        mean_signed_autocorrelation = values.new_zeros(())
        mean_abs_autocorrelation = values.new_zeros(())
        max_abs_autocorrelation = values.new_zeros(())
        autocorrelation_nonstationarity = values.new_zeros(())
        autocorrelation_decay = values.new_zeros(())
        lag1_signed = values.new_zeros(())
        lag1_abs = values.new_zeros(())

    return _finite_feature_mapping(
        {
            "residual_mean_signed_channel_correlation": mean_signed_channel,
            "residual_mean_abs_channel_correlation": mean_abs_channel,
            "residual_channel_correlation_dispersion": channel_dispersion,
            "residual_correlation_nonstationarity": nonstationarity,
            "residual_correlation_effective_rank": effective_rank,
            "residual_correlation_effective_rank_fraction": effective_rank_fraction,
            "residual_max_lagged_cross_correlation": max_lagged_cross,
            "residual_lag1_signed_autocorrelation": lag1_signed,
            "residual_lag1_abs_autocorrelation": lag1_abs,
            "residual_mean_signed_autocorrelation": mean_signed_autocorrelation,
            "residual_mean_abs_autocorrelation": mean_abs_autocorrelation,
            "residual_max_abs_autocorrelation": max_abs_autocorrelation,
            "residual_autocorrelation_nonstationarity": autocorrelation_nonstationarity,
            "residual_autocorrelation_decay": autocorrelation_decay,
        }
    )


def _windowed_cross_correlation(left: Tensor, right: Tensor) -> Tensor:
    """Masked correlation between every left/right channel in each window."""

    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("Correlation operands must have shape [batch, channels, time]")
    if left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-1]:
        raise ValueError("Correlation operands need matching batch and time dimensions")

    left_valid = torch.isfinite(left)
    right_valid = torch.isfinite(right)
    left_clean = torch.where(left_valid, left, torch.zeros_like(left)).float()
    right_clean = torch.where(right_valid, right, torch.zeros_like(right)).float()
    left_mask = left_valid.float()
    right_mask = right_valid.float()

    counts = torch.einsum("bit,bjt->bij", left_mask, right_mask)
    safe_counts = counts.clamp_min(1.0)
    left_sum = torch.einsum("bit,bjt->bij", left_clean, right_mask)
    right_sum = torch.einsum("bit,bjt->bij", left_mask, right_clean)
    cross_sum = torch.einsum("bit,bjt->bij", left_clean, right_clean)
    left_square_sum = torch.einsum("bit,bjt->bij", left_clean.square(), right_mask)
    right_square_sum = torch.einsum("bit,bjt->bij", left_mask, right_clean.square())

    covariance = cross_sum - left_sum * right_sum / safe_counts
    left_variance = (left_square_sum - left_sum.square() / safe_counts).clamp_min(0.0)
    right_variance = (right_square_sum - right_sum.square() / safe_counts).clamp_min(0.0)
    denominator = (left_variance * right_variance).sqrt()
    identifiable = (counts >= 2) & (denominator > 1e-12)
    output = torch.where(identifiable, covariance / denominator.clamp_min(1e-12), 0.0)
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def _effective_correlation_rank(values: Tensor) -> Tensor:
    """Entropy effective rank of a positive-semidefinite correlation Gram matrix."""

    channels = values.shape[1]
    flattened = values.permute(1, 0, 2).reshape(channels, -1)
    valid = torch.isfinite(flattened)
    clean = torch.where(valid, flattened, torch.zeros_like(flattened))
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    means = clean.sum(dim=1, keepdim=True) / counts
    centered = torch.where(valid, clean - means, torch.zeros_like(clean))
    norms = centered.square().sum(dim=1, keepdim=True).sqrt()
    standardized = torch.where(norms > 1e-12, centered / norms.clamp_min(1e-12), 0.0)
    gram = standardized @ standardized.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 1e-12:
        return values.new_zeros(())
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp().clamp(0.0, float(channels))


def _finite_feature_mapping(values: Mapping[str, float | Tensor]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, value in values.items():
        number = float(value)
        output[str(name)] = number if math.isfinite(number) else 0.0
    return output
