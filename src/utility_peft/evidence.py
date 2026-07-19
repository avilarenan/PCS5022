"""Support-only structural, mismatch, and one-backward-pass evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from utility_peft.model import AdaptableForecaster
from utility_peft.types import EvidenceBundle, SupportView

TIME_PEFT_COMPLEXITY_FEATURES = frozenset(
    {
        "spectral_entropy",
        "transfer_entropy_mean",
    }
)
STRUCTURAL_FEATURES = frozenset(
    {
        "channels",
        "lookback",
        "horizon",
        "horizon_lookback_ratio",
        "support_size",
        "input_mean",
        "input_std",
        "input_skewness",
        "input_kurtosis",
        "missing_fraction",
        "lag1_autocorrelation",
        "spectral_entropy",
        "top_frequency_mass",
        "mean_abs_channel_correlation",
        "transfer_entropy_mean",
        "transfer_entropy_max",
        "transfer_entropy_asymmetry",
        "linear_trend",
    }
)
MISMATCH_FEATURES = frozenset(
    {
        "frozen_mse",
        "frozen_mae",
        "residual_bias",
        "residual_std",
        "prediction_std",
        "target_std",
    }
)
FEATURE_SETS = frozenset({"complexity", "structure", "structure_mismatch", "full"})


def extract_evidence(
    support: SupportView,
    model: AdaptableForecaster,
    *,
    device: str | torch.device = "cpu",
    include_gradient_probe: bool = True,
) -> EvidenceBundle:
    """Extract controller inputs without accepting an evaluation/query object."""

    if type(support) is not SupportView:
        raise TypeError("Evidence extraction accepts SupportView only")
    target_device = torch.device(device)
    original_device = next(model.parameters()).device
    model = model.to(target_device)
    x = support.x.to(target_device)
    y = support.y.to(target_device)
    mask = support.mask.to(target_device)
    horizon = support.manifest.horizon
    features = _structural_features(support)
    features.update(model.backbone.source_statistics())

    was_training = model.training
    model.eval()
    with torch.no_grad():
        prediction = model.predict(x, mask, horizon)
        residual = prediction.float() - y.float()
        features.update(
            {
                "frozen_mse": float(residual.square().mean()),
                "frozen_mae": float(residual.abs().mean()),
                "residual_bias": float(residual.mean()),
                "residual_std": float(residual.std(unbiased=False)),
                "prediction_std": float(prediction.float().std(unbiased=False)),
                "target_std": float(y.float().std(unbiased=False)),
            }
        )
    if include_gradient_probe:
        features.update(_gradient_probe(model, x, y, mask, horizon))
    model.train(was_training)
    model.to(original_device)
    return EvidenceBundle.from_mapping(support.manifest.episode_id, features)


def _structural_features(support: SupportView) -> dict[str, float]:
    x = support.x.float()
    mask = support.mask
    valid = mask[:, None, :].expand_as(x)
    values = x[valid]
    centered = values - values.mean()
    variance = centered.square().mean().clamp_min(1e-12)
    skewness = centered.pow(3).mean() / variance.pow(1.5)
    kurtosis = centered.pow(4).mean() / variance.square()

    lagged_left = x[..., 1:]
    lagged_right = x[..., :-1]
    autocorrelation = _correlation(lagged_left.flatten(), lagged_right.flatten())
    spectrum = torch.fft.rfft(x, dim=-1)
    power = spectrum.abs().square().mean(dim=(0, 1))
    probabilities = power / power.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    entropy /= math.log(max(probabilities.numel(), 2))
    top_count = max(1, math.ceil(probabilities.numel() / 4))
    top_frequency_mass = probabilities.topk(top_count).values.sum()

    channel_correlation = 0.0
    transfer_entropy_mean = 0.0
    transfer_entropy_max = 0.0
    transfer_entropy_asymmetry = 0.0
    if x.shape[1] > 1:
        channels = x.permute(1, 0, 2).reshape(x.shape[1], -1)
        correlation = torch.corrcoef(channels)
        off_diagonal = ~torch.eye(x.shape[1], dtype=torch.bool, device=correlation.device)
        channel_correlation = float(torch.nan_to_num(correlation[off_diagonal]).abs().mean())
        transfer_entropy = lagged_transfer_entropy(x)
        transfer_entropy_mean = float(transfer_entropy[off_diagonal].mean())
        transfer_entropy_max = float(transfer_entropy[off_diagonal].max())
        transfer_entropy_asymmetry = float(
            (transfer_entropy - transfer_entropy.T).abs()[off_diagonal].mean()
        )
    trend = (x[..., -1].mean() - x[..., 0].mean()) / max(x.shape[-1] - 1, 1)
    manifest = support.manifest
    return {
        "channels": float(x.shape[1]),
        "lookback": float(x.shape[2]),
        "horizon": float(manifest.horizon),
        "horizon_lookback_ratio": manifest.horizon / x.shape[2],
        "support_size": float(x.shape[0]),
        "input_mean": float(values.mean()),
        "input_std": float(values.std(unbiased=False)),
        "input_skewness": float(skewness),
        "input_kurtosis": float(kurtosis),
        "missing_fraction": float((~valid).float().mean()),
        "lag1_autocorrelation": float(autocorrelation),
        "spectral_entropy": float(entropy),
        "top_frequency_mass": float(top_frequency_mass),
        "mean_abs_channel_correlation": channel_correlation,
        "transfer_entropy_mean": transfer_entropy_mean,
        "transfer_entropy_max": transfer_entropy_max,
        "transfer_entropy_asymmetry": transfer_entropy_asymmetry,
        "linear_trend": float(trend),
    }


def lagged_transfer_entropy(x: Tensor, *, lag: int = 1, ridge: float = 1e-6) -> Tensor:
    """Estimate directed Gaussian transfer entropy between every channel pair.

    This is conditional mutual information ``I(Y_t; X_{t-lag} | Y_{t-lag})``
    under a linear-Gaussian model. Windows remain separate when lagged samples
    are formed, so no artificial transition is introduced between examples.
    """

    if x.ndim != 3:
        raise ValueError("Transfer entropy expects [batch, channels, time]")
    if lag <= 0 or lag >= x.shape[-1]:
        raise ValueError("Transfer-entropy lag must be inside the lookback window")
    values = x.detach().to(device="cpu", dtype=torch.float64)
    channels = values.shape[1]
    output = torch.zeros((channels, channels), dtype=torch.float64)
    for target in range(channels):
        current = values[:, target, lag:].reshape(-1)
        target_past = values[:, target, :-lag].reshape(-1)
        reduced = torch.stack((torch.ones_like(target_past), target_past), dim=1)
        reduced_variance = _linear_residual_variance(reduced, current, ridge)
        for source in range(channels):
            if source == target:
                continue
            source_past = values[:, source, :-lag].reshape(-1)
            full = torch.stack(
                (torch.ones_like(target_past), target_past, source_past), dim=1
            )
            full_variance = _linear_residual_variance(full, current, ridge)
            ratio = reduced_variance / full_variance.clamp_min(ridge)
            output[source, target] = (0.5 * ratio.clamp_min(1.0).log()).clamp_min(0.0)
    return output.float()


def select_feature_mapping(
    evidence: dict[str, float] | Mapping[str, float], feature_set: str
) -> dict[str, float]:
    """Filter one evidence row for a preregistered ablation."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown evidence feature set: {feature_set}")
    values = dict(evidence)
    if feature_set == "complexity":
        allowed = TIME_PEFT_COMPLEXITY_FEATURES
    elif feature_set == "structure":
        allowed = STRUCTURAL_FEATURES
    elif feature_set == "structure_mismatch":
        allowed = STRUCTURAL_FEATURES | MISMATCH_FEATURES
    else:
        return values
    return {name: value for name, value in values.items() if name in allowed}


def _gradient_probe(
    model: AdaptableForecaster,
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    horizon: int,
) -> dict[str, float]:
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    prediction = model.predict(x, mask, horizon)
    F.mse_loss(prediction.float(), y.float()).backward()
    squared_norms: dict[str, Tensor] = defaultdict(lambda: torch.zeros((), device=x.device))
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _parameter_group(name)
        squared_norms[group] = squared_norms[group] + parameter.grad.float().square().sum()
        counts[group] += parameter.numel()
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_flags[name])
    output: dict[str, float] = {}
    for group in ("head", "query", "value", "encoder_other"):
        norm = squared_norms[group].sqrt()
        output[f"gradient_{group}_l2"] = float(norm)
        output[f"gradient_{group}_rms"] = float(norm / math.sqrt(max(counts[group], 1)))
    return output


def _parameter_group(name: str) -> str:
    leafs = name.split(".")
    if "forecast_head" in name or ".head." in name:
        return "head"
    if "q" in leafs:
        return "query"
    if "v" in leafs:
        return "value"
    return "encoder_other"


def _correlation(left: Tensor, right: Tensor) -> Tensor:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return (left * right).sum() / denominator.clamp_min(1e-12)


def _linear_residual_variance(design: Tensor, target: Tensor, ridge: float) -> Tensor:
    gram = design.T @ design
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(gram + penalty, design.T @ target)
    residual = target - design @ coefficients
    return residual.square().mean().clamp_min(ridge)
