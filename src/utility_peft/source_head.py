"""Leakage-auditable source forecasting-head training."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utility_peft.data.datasets import DatasetSeries
from utility_peft.model import AdaptableForecaster
from utility_peft.utils import atomic_write_json, preprocessing_hash, seed_everything

SOURCE_HEAD_PREPROCESSING_SPEC = {
    "version": 1,
    "scaler": "source-train-channel-standard",
    "fit_split": "train",
    "missing": "zero-after-scaling",
}
SOURCE_HEAD_PREPROCESSING_HASH = preprocessing_hash(SOURCE_HEAD_PREPROCESSING_SPEC)


@dataclass(frozen=True, slots=True)
class SourceHeadTrainingConfig:
    updates: int = 2_000
    batch_size: int = 32
    validation_windows: int = 256
    validation_interval: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    bf16: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.updates,
            self.batch_size,
            self.validation_windows,
            self.validation_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Source-head update and batch settings must be positive")
        if self.validation_interval > self.updates:
            raise ValueError("validation_interval cannot exceed updates")


@dataclass(frozen=True, slots=True)
class SourceHeadMetrics:
    source_dataset: str
    source_dataset_sha256: str
    preprocessing_hash: str
    scaler_statistics_sha256: str
    horizon: int
    updates: int
    best_update: int
    best_validation_mse: float
    checkpoint_sha256: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    excluded_evaluation_datasets: tuple[str, ...]


def train_source_head(
    model: AdaptableForecaster,
    dataset: DatasetSeries,
    *,
    horizon: int,
    lookback: int,
    checkpoint_path: str | Path,
    evaluation_datasets: tuple[str, ...],
    config: SourceHeadTrainingConfig | None = None,
    device: str | torch.device = "cuda",
) -> SourceHeadMetrics:
    """Train a forecasting head using source train/validation timestamps only."""

    config = config or SourceHeadTrainingConfig()
    if dataset.name in set(evaluation_datasets):
        raise ValueError(
            f"Source-head dataset {dataset.name} is also an evaluation dataset"
        )
    train_split = dataset.split("train")
    validation_split = dataset.split("validation")
    _validate_window_range(train_split.start, train_split.end, lookback, horizon)
    _validate_window_range(validation_split.start, validation_split.end, lookback, horizon)

    seed_everything(config.seed)
    target_device = torch.device(device)
    model = model.to(target_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head_parameters = model.backbone.head_parameters()
    for parameter in head_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    values, scaler_hash = _standardize_from_train_split(
        dataset.values.float(), train_split.start, train_split.end
    )
    use_amp = config.bf16 and target_device.type == "cuda"
    best_loss = float("inf")
    best_update = 0
    best_state: dict[str, Tensor] | None = None

    model.train()
    for update in range(1, config.updates + 1):
        starts = torch.randint(
            train_split.start,
            train_split.end - lookback - horizon + 1,
            (config.batch_size,),
            generator=generator,
        )
        x, y, mask = _window_batch(values, starts, lookback, horizon, target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            prediction = model.predict(x, mask, horizon)
            loss = F.mse_loss(prediction.float(), y.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Source-head loss became non-finite at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(head_parameters, config.gradient_clip)
        optimizer.step()

        if update % config.validation_interval == 0:
            validation_loss = _validation_loss(
                model,
                values,
                start=validation_split.start,
                end=validation_split.end,
                lookback=lookback,
                horizon=horizon,
                windows=config.validation_windows,
                batch_size=config.batch_size,
                device=target_device,
                use_amp=use_amp,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_update = update
                best_state = copy.deepcopy(_head_state(model))
            model.train()

    if best_state is None:
        raise RuntimeError("Source-head training produced no validation checkpoint")
    _load_head_state(model, best_state)
    target = Path(checkpoint_path)
    model.backbone.save_source_head(target)
    checkpoint_hash = _sha256(target)
    metrics = SourceHeadMetrics(
        source_dataset=dataset.name,
        source_dataset_sha256=dataset.sha256,
        preprocessing_hash=SOURCE_HEAD_PREPROCESSING_HASH,
        scaler_statistics_sha256=scaler_hash,
        horizon=horizon,
        updates=config.updates,
        best_update=best_update,
        best_validation_mse=best_loss,
        checkpoint_sha256=checkpoint_hash,
        train_start=train_split.start,
        train_end=train_split.end,
        validation_start=validation_split.start,
        validation_end=validation_split.end,
        excluded_evaluation_datasets=tuple(sorted(evaluation_datasets)),
    )
    atomic_write_json(target.with_suffix(".metrics.json"), asdict(metrics))
    return metrics


def validate_source_head_provenance(
    checkpoint_path: str | Path,
    *,
    horizon: int,
    evaluation_datasets: tuple[str, ...],
) -> SourceHeadMetrics:
    """Reject missing, stale, or target-contaminated source-head checkpoints."""

    checkpoint = Path(checkpoint_path)
    metadata_path = checkpoint.with_suffix(".metrics.json")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Source-head checkpoint does not exist: {checkpoint}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Source-head provenance does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metrics = SourceHeadMetrics(**json.load(handle))
    if metrics.horizon != horizon:
        raise ValueError(
            f"Source-head horizon {metrics.horizon} does not match requested horizon {horizon}"
        )
    if metrics.preprocessing_hash != SOURCE_HEAD_PREPROCESSING_HASH:
        raise ValueError("Source-head preprocessing does not match the active implementation")
    evaluation = set(evaluation_datasets)
    if metrics.source_dataset in evaluation:
        raise ValueError(
            f"Source-head provenance leaks evaluation dataset {metrics.source_dataset}"
        )
    if set(metrics.excluded_evaluation_datasets) != evaluation:
        raise ValueError("Source-head exclusion set does not match the configured pilot datasets")
    if metrics.checkpoint_sha256 != _sha256(checkpoint):
        raise ValueError("Source-head checkpoint hash does not match its provenance")
    return metrics


def _window_batch(
    values: Tensor,
    starts: Tensor,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    windows = torch.stack(
        [values[:, int(start) : int(start) + lookback + horizon] for start in starts]
    )
    x = windows[..., :lookback].to(device)
    y = windows[..., lookback:].to(device)
    mask = torch.isfinite(x).all(dim=1)
    return torch.nan_to_num(x), torch.nan_to_num(y), mask


def _standardize_from_train_split(
    values: Tensor, start: int, end: int
) -> tuple[Tensor, str]:
    fitted = values[:, start:end]
    mean = torch.nanmean(fitted, dim=1, keepdim=True)
    centered = fitted - mean
    std = torch.sqrt(torch.nanmean(centered.square(), dim=1, keepdim=True)).clamp_min(1e-6)
    standardized = torch.nan_to_num((values - mean) / std)
    digest = hashlib.sha256()
    digest.update(mean.detach().cpu().contiguous().numpy().tobytes())
    digest.update(std.detach().cpu().contiguous().numpy().tobytes())
    return standardized, digest.hexdigest()


@torch.no_grad()
def _validation_loss(
    model: AdaptableForecaster,
    values: Tensor,
    *,
    start: int,
    end: int,
    lookback: int,
    horizon: int,
    windows: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    latest = end - lookback - horizon
    starts = torch.from_numpy(np.linspace(start, latest, windows, dtype=np.int64))
    squared_error = 0.0
    elements = 0
    model.eval()
    for offset in range(0, starts.numel(), batch_size):
        x, y, mask = _window_batch(
            values,
            starts[offset : offset + batch_size],
            lookback,
            horizon,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            prediction = model.predict(x, mask, horizon)
        squared_error += float((prediction.float() - y.float()).square().sum())
        elements += y.numel()
    return squared_error / max(elements, 1)


def _head_state(model: AdaptableForecaster) -> dict[str, Tensor]:
    head_ids = {id(parameter) for parameter in model.backbone.head_parameters()}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.backbone.named_parameters()
        if id(value) in head_ids
    }


def _load_head_state(model: AdaptableForecaster, state: dict[str, Tensor]) -> None:
    parameters = dict(model.backbone.named_parameters())
    if state.keys() - parameters.keys():
        raise RuntimeError("Source-head state does not match the backbone")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _validate_window_range(start: int, end: int, lookback: int, horizon: int) -> None:
    if end - start < lookback + horizon:
        raise ValueError(
            f"Split [{start}, {end}) cannot provide lookback {lookback} and horizon {horizon}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
