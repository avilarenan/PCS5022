"""Paper-style L versus LFC Time-PEFT reproduction utilities.

This module implements the conventional target-train/validation/test protocol
described by the Time-PEFT paper.  It is deliberately separate from the
few-shot episode evaluator: hyperparameters are selected on validation data and
the official test split is evaluated exactly once for the selected checkpoint.

The implementation is a paper-specified reproduction, not a claim of parity
with unavailable official training code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from utility_peft.actions import TIME_PEFT_ACTION_BY_ID
from utility_peft.data.datasets import DatasetSeries, DatasetSplit
from utility_peft.model import AdaptableForecaster, count_parameters, model_for_action
from utility_peft.utils import implementation_hash

PAPER_METHOD_IDS = ("L", "LFC")
_COMPLEX_FAMILIES = frozenset({"synthetic", "chaotic", "medical"})
_TRIAL_CACHE_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class SmokeCaps:
    """Explicit reductions for quick integration checks.

    A production run should leave every field as ``None``.  Keeping these
    limits in a named object makes shortened runs visible in result records.
    """

    train_windows: int | None = None
    validation_windows: int | None = None
    test_windows: int | None = None
    batches_per_epoch: int | None = None
    evaluation_batches: int | None = None
    epochs: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Smoke cap {name} must be positive when specified")

    @property
    def active(self) -> bool:
        return any(getattr(self, name) is not None for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class TimePEFTReproductionConfig:
    """Training controls for the paper-style L/LFC comparison."""

    lookback: int = 96
    horizon: int = 96
    learning_rates: tuple[float, ...] = (1e-3, 1e-4, 1e-5)
    batch_size: int = 128
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    weight_decay: float = 0.01
    gradient_clip: float | None = None
    seeds: tuple[int, ...] = (0,)
    device: str = "auto"
    precision: str = "fp32"
    complex_split_override_70_10_20: bool = True
    smoke_caps: SmokeCaps | None = None

    def __post_init__(self) -> None:
        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError("lookback and horizon must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 1 <= self.max_epochs <= 100:
            raise ValueError("max_epochs must be between 1 and the paper maximum of 100")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be finite and positive when specified")
        if not self.learning_rates:
            raise ValueError("learning_rates cannot be empty")
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("learning_rates must not contain duplicates")
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.learning_rates):
            raise ValueError("Every learning rate must be finite and positive")
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if self.precision != "fp32":
            raise ValueError("The reproduction core currently requires explicit FP32 precision")
        if self.device != "auto":
            try:
                torch.device(self.device)
            except (RuntimeError, TypeError) as error:
                raise ValueError(f"Invalid device {self.device!r}") from error


@dataclass(frozen=True, slots=True)
class TrainStandardizer:
    """Per-channel statistics fitted exclusively on the target train split."""

    mean: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 2 or self.scale.shape != self.mean.shape:
            raise ValueError("Standardizer mean and scale must have shape [channels, 1]")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("Standardizer statistics must be finite")
        if (self.scale <= 0).any():
            raise ValueError("Standardizer scales must be positive")

    def transform(self, values: Tensor) -> Tensor:
        """Apply the train-fitted transform while preserving missing inputs."""

        if values.ndim != 2 or values.shape[0] != self.mean.shape[0]:
            raise ValueError("Values do not match the fitted channel count")
        statistics_device = self.mean.device
        if values.device != statistics_device:
            values = values.to(statistics_device)
        return (values - self.mean) / self.scale


@dataclass(frozen=True, slots=True)
class SplitWindows:
    """Lazy windows whose forecast targets are contained in one named split."""

    split_name: str
    values: Tensor
    target_starts: Tensor
    split_start: int
    split_end: int
    lookback: int
    horizon: int

    def __post_init__(self) -> None:
        if self.split_name not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split {self.split_name!r}")
        if self.values.ndim != 2 or not self.values.dtype.is_floating_point:
            raise ValueError("Window values must be floating point [channels, time]")
        if self.target_starts.ndim != 1 or self.target_starts.dtype != torch.long:
            raise ValueError("target_starts must be a one-dimensional long tensor")
        if self.target_starts.numel() == 0:
            raise ValueError(f"Split {self.split_name} produced no forecasting windows")
        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError("lookback and horizon must be positive")
        if not bool((self.target_starts[1:] > self.target_starts[:-1]).all()):
            raise ValueError("target_starts must be strictly increasing")
        if int(self.target_starts[0]) - self.lookback < 0:
            raise ValueError("A window requests context before the beginning of the series")
        if int(self.target_starts[0]) < self.split_start:
            raise ValueError("A target begins before its named split")
        if int(self.target_starts[-1]) + self.horizon > self.split_end:
            raise ValueError("A target ends after its named split")
        if (
            self.split_name == "train"
            and int(self.target_starts[0]) - self.lookback < self.split_start
        ):
            raise ValueError("Training context must remain inside the train split")

    def __len__(self) -> int:
        return self.target_starts.numel()

    @property
    def context_starts(self) -> Tensor:
        return self.target_starts - self.lookback

    @property
    def target_ends(self) -> Tensor:
        return self.target_starts + self.horizon

    def batch(
        self,
        indices: Tensor | Sequence[int],
        *,
        device: str | torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Materialize selected ``x``, ``y``, and observed-time masks."""

        selected = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if selected.numel() == 0:
            raise ValueError("A window batch cannot be empty")
        if int(selected.min()) < 0 or int(selected.max()) >= len(self):
            raise IndexError("Window batch index is out of range")
        starts = self.target_starts[selected]
        x = torch.stack(
            [
                self.values[:, int(start) - self.lookback : int(start)]
                for start in starts
            ]
        )
        y = torch.stack(
            [self.values[:, int(start) : int(start) + self.horizon] for start in starts]
        )
        _assert_finite(y, f"{self.split_name} targets")
        mask = torch.isfinite(x).all(dim=1)
        target_device = torch.device(device)
        return (
            torch.nan_to_num(x).to(target_device),
            y.to(target_device),
            mask.to(target_device),
        )


@dataclass(frozen=True, slots=True)
class ReproductionWindows:
    """Train-normalized windows for all three chronological partitions."""

    standardizer: TrainStandardizer
    train: SplitWindows
    validation: SplitWindows
    test: SplitWindows


@dataclass(frozen=True, slots=True)
class TuningWindows:
    """Train and validation windows; intentionally contains no test view."""

    standardizer: TrainStandardizer
    train: SplitWindows
    validation: SplitWindows


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One completed epoch in a trial's train/validation curve."""

    epoch: int
    train_mse: float
    validation_mse: float
    validation_mae: float
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Validation-only outcome for one method, seed, and AdamW learning rate."""

    dataset: str
    dataset_sha256: str
    method_id: str
    seed: int
    learning_rate: float
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    validation_mse: float | None
    validation_mae: float | None
    trainable_parameters: int
    total_parameters: int
    train_windows: int
    validation_windows: int
    initialization_fingerprint: str
    batch_order_seed: int
    status: str = "ok"
    error: str | None = None
    epoch_metrics: tuple[EpochMetrics, ...] = ()
    elapsed_time_s: float = field(default=0.0, compare=False)
    peak_cuda_memory_mb: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "failed"}:
            raise ValueError("Trial status must be 'ok' or 'failed'")
        if self.status == "ok":
            if self.validation_mse is None or self.validation_mae is None:
                raise ValueError("Successful trials require validation metrics")
            if not math.isfinite(self.validation_mse) or not math.isfinite(self.validation_mae):
                raise ValueError("Successful trial validation metrics must be finite")
            if self.error is not None:
                raise ValueError("Successful trials cannot contain an error")
        else:
            if self.error is None:
                raise ValueError("Failed trials require an error message")
            if self.validation_mse is not None or self.validation_mae is not None:
                raise ValueError("Failed trials cannot publish selection metrics")
        if not math.isfinite(self.elapsed_time_s) or self.elapsed_time_s < 0:
            raise ValueError("Trial elapsed time must be finite and non-negative")
        if not math.isfinite(self.peak_cuda_memory_mb) or self.peak_cuda_memory_mb < 0:
            raise ValueError("Trial peak CUDA memory must be finite and non-negative")
        for metric in self.epoch_metrics:
            values = (metric.train_mse, metric.validation_mse, metric.validation_mae)
            if metric.epoch <= 0 or metric.optimizer_steps <= 0:
                raise ValueError("Epoch metrics require positive epoch and step counts")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Epoch curves must contain finite metrics")


@dataclass(frozen=True, slots=True)
class TrialCacheIdentity:
    """Complete identity of one resumable method/seed/LR trial."""

    schema_version: int
    dataset: str
    dataset_sha256: str
    split_ranges: tuple[tuple[str, int, int], ...]
    config_fingerprint: str
    method_id: str
    seed: int
    learning_rate: float
    template_fingerprint: str


@dataclass(slots=True)
class _CachedTrial:
    identity: TrialCacheIdentity
    record: TrialRecord
    checkpoint: dict[str, Tensor] | None


@dataclass(slots=True)
class SelectedCheckpoint:
    """Selected validation checkpoint containing trainable tensors only."""

    record: TrialRecord
    trainable_state: dict[str, Tensor]


@dataclass(slots=True)
class MethodTuning:
    """Common-LR selection and selected seed checkpoints for one method."""

    method_id: str
    selected_learning_rate: float
    selection_mean_validation_mse: float
    trials: tuple[TrialRecord, ...]
    checkpoints: tuple[SelectedCheckpoint, ...]


@dataclass(slots=True)
class TuningResult:
    """Serializable tune-stage artifact that has never evaluated test data."""

    dataset: str
    dataset_family: str
    dataset_sha256: str
    config: TimePEFTReproductionConfig
    split_policy: str
    split_ranges: tuple[tuple[str, int, int], ...]
    methods: tuple[MethodTuning, ...]

    @property
    def trial_records(self) -> tuple[TrialRecord, ...]:
        return tuple(trial for method in self.methods for trial in method.trials)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata, excluding checkpoint tensors."""

        return {
            "dataset": self.dataset,
            "dataset_family": self.dataset_family,
            "dataset_sha256": self.dataset_sha256,
            "config": asdict(self.config),
            "split_policy": self.split_policy,
            "split_ranges": self.split_ranges,
            "methods": [
                {
                    "method_id": method.method_id,
                    "selected_learning_rate": method.selected_learning_rate,
                    "selection_mean_validation_mse": (
                        method.selection_mean_validation_mse
                    ),
                    "trials": [asdict(trial) for trial in method.trials],
                    "checkpoint_seeds": [
                        checkpoint.record.seed for checkpoint in method.checkpoints
                    ],
                }
                for method in self.methods
            ],
        }


@dataclass(frozen=True, slots=True)
class MethodRecord:
    """Selected validation trial and its single untouched-test evaluation."""

    dataset: str
    dataset_family: str
    dataset_sha256: str
    method_id: str
    seed: int
    selected_learning_rate: float
    selected_epoch: int
    selection_mean_validation_mse: float
    validation_mse: float
    validation_mae: float
    test_mse: float
    test_mae: float
    trainable_parameters: int
    total_parameters: int
    test_windows: int
    test_evaluations: int
    smoke_caps_active: bool
    trials: tuple[TrialRecord, ...]


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """Complete L/LFC records for one dataset and configured seed set."""

    dataset: str
    dataset_family: str
    dataset_sha256: str
    lookback: int
    horizon: int
    precision: str
    split_policy: str
    split_ranges: tuple[tuple[str, int, int], ...]
    records: tuple[MethodRecord, ...]
    smoke_caps: SmokeCaps | None

    @property
    def trial_records(self) -> tuple[TrialRecord, ...]:
        return tuple(trial for record in self.records for trial in record.trials)


@dataclass(frozen=True, slots=True)
class MethodAggregate:
    """Across-run summary for one method."""

    method_id: str
    runs: int
    mean_validation_mse: float
    mean_test_mse: float
    std_test_mse: float
    mean_test_mae: float
    std_test_mae: float
    selected_learning_rates: tuple[tuple[float, int], ...]


@dataclass(slots=True)
class _FittedTrial:
    record: TrialRecord
    checkpoint: dict[str, Tensor] | None


class TrialCache:
    """Atomic directory-backed cache for completed tuning trials."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, identity: TrialCacheIdentity) -> _FittedTrial | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cached, _CachedTrial):
            raise TypeError(f"Trial cache entry {path} has an unexpected payload")
        if cached.identity != identity:
            raise ValueError(f"Trial cache identity mismatch in {path}")
        _validate_cached_trial(cached)
        return _FittedTrial(record=cached.record, checkpoint=cached.checkpoint)

    def store(self, identity: TrialCacheIdentity, fitted: _FittedTrial) -> Path:
        cached = _CachedTrial(
            identity=identity,
            record=fitted.record,
            checkpoint=fitted.checkpoint,
        )
        _validate_cached_trial(cached)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(identity)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{target.stem}-",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(cached, temporary)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def _path(self, identity: TrialCacheIdentity) -> Path:
        encoded = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return self.root / f"trial-{digest}.pt"


@dataclass(frozen=True, slots=True)
class _Metrics:
    mse: float
    mae: float
    examples: int


def fit_train_standardizer(series: DatasetSeries) -> TrainStandardizer:
    """Fit channel-wise mean and scale without reading validation or test values."""

    train = series.split("train")
    fitted = series.values.detach().to(device="cpu", dtype=torch.float32)[
        :, train.start : train.end
    ]
    finite_counts = torch.isfinite(fitted).sum(dim=1, keepdim=True)
    if bool((finite_counts == 0).any()):
        bad = torch.nonzero(finite_counts.squeeze(1) == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Train split has no finite observations for channels {bad}")
    mean = torch.nanmean(fitted, dim=1, keepdim=True)
    centered = fitted - mean
    scale = torch.sqrt(torch.nanmean(centered.square(), dim=1, keepdim=True)).clamp_min(1e-6)
    return TrainStandardizer(mean=mean, scale=scale)


def build_split_windows(
    series: DatasetSeries,
    split_name: str,
    *,
    lookback: int,
    horizon: int,
    standardizer: TrainStandardizer | None = None,
    max_windows: int | None = None,
) -> SplitWindows:
    """Build lazy windows with targets wholly inside ``split_name``.

    Validation and test windows may obtain their lookback from timestamps before
    their split boundary.  Training windows keep both context and target inside
    the train split.
    """

    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive when specified")
    split = series.split(split_name)
    normalized_name = split.name
    earliest_target = (
        split.start + lookback
        if normalized_name == "train"
        else max(split.start, lookback)
    )
    latest_target = split.end - horizon
    if latest_target < earliest_target:
        context_rule = "inside train" if normalized_name == "train" else "from preceding history"
        raise ValueError(
            f"Split {normalized_name} cannot provide lookback {lookback} ({context_rule}) "
            f"and horizon {horizon}"
        )
    starts = torch.arange(earliest_target, latest_target + 1, dtype=torch.long)
    if max_windows is not None and starts.numel() > max_windows:
        starts = starts[_evenly_spaced_indices(starts.numel(), max_windows)]

    raw_values = series.values.detach().to(device="cpu", dtype=torch.float32)
    _assert_window_targets_finite(
        raw_values,
        starts,
        horizon,
        dataset=series.name,
        split_name=normalized_name,
    )
    values = standardizer.transform(raw_values) if standardizer is not None else raw_values
    return SplitWindows(
        split_name=normalized_name,
        values=values,
        target_starts=starts,
        split_start=split.start,
        split_end=split.end,
        lookback=lookback,
        horizon=horizon,
    )


def prepare_tuning_windows(
    series: DatasetSeries,
    *,
    lookback: int,
    horizon: int,
    smoke_caps: SmokeCaps | None = None,
) -> TuningWindows:
    """Construct train/validation windows without reading the test split."""

    caps = smoke_caps or SmokeCaps()
    standardizer = fit_train_standardizer(series)
    return TuningWindows(
        standardizer=standardizer,
        train=build_split_windows(
            series,
            "train",
            lookback=lookback,
            horizon=horizon,
            standardizer=standardizer,
            max_windows=caps.train_windows,
        ),
        validation=build_split_windows(
            series,
            "validation",
            lookback=lookback,
            horizon=horizon,
            standardizer=standardizer,
            max_windows=caps.validation_windows,
        ),
    )


def prepare_reproduction_windows(
    series: DatasetSeries,
    *,
    lookback: int,
    horizon: int,
    smoke_caps: SmokeCaps | None = None,
) -> ReproductionWindows:
    """Fit train-only preprocessing and construct all split-safe windows."""

    caps = smoke_caps or SmokeCaps()
    tuning = prepare_tuning_windows(
        series,
        lookback=lookback,
        horizon=horizon,
        smoke_caps=smoke_caps,
    )
    return ReproductionWindows(
        standardizer=tuning.standardizer,
        train=tuning.train,
        validation=tuning.validation,
        test=build_split_windows(
            series,
            "test",
            lookback=lookback,
            horizon=horizon,
            standardizer=tuning.standardizer,
            max_windows=caps.test_windows,
        ),
    )


def run_time_peft_reproduction(
    template: AdaptableForecaster | Callable[[int], AdaptableForecaster],
    series: DatasetSeries,
    config: TimePEFTReproductionConfig | None = None,
    *,
    trial_cache: TrialCache | str | Path | None = None,
) -> ReproductionResult:
    """Compose the separate tune and untouched-test stages."""

    tuning = tune_time_peft(template, series, config, trial_cache=trial_cache)
    return test_time_peft(template, series, tuning)


def tune_time_peft(
    template: AdaptableForecaster | Callable[[int], AdaptableForecaster],
    series: DatasetSeries,
    config: TimePEFTReproductionConfig | None = None,
    *,
    trial_cache: TrialCache | str | Path | None = None,
) -> TuningResult:
    """Train all seed/LR trials and select common L/LFC learning rates.

    This function constructs only train and validation windows; it never
    constructs, batches, or evaluates a test window.
    """

    config = config or TimePEFTReproductionConfig()
    device = _resolve_device(config.device)
    cache = _resolve_trial_cache(trial_cache)
    _validate_template_source(template, config)
    templates = {
        seed: _template_for_seed(template, seed, series, config, device)
        for seed in config.seeds
    }
    template_fingerprints = {
        seed: _state_fingerprint(templates[seed]) for seed in config.seeds
    }
    config_fingerprint = _config_fingerprint(config)
    protocol_series, split_policy = _paper_protocol_series(series, config)
    split_ranges = tuple(
        (split.name, split.start, split.end) for split in protocol_series.splits
    )
    windows = prepare_tuning_windows(
        protocol_series,
        lookback=config.lookback,
        horizon=config.horizon,
        smoke_caps=config.smoke_caps,
    )
    methods: list[MethodTuning] = []
    for method_id in PAPER_METHOD_IDS:
        trial_records: dict[int, list[TrialRecord]] = {
            seed: [] for seed in config.seeds
        }
        selected_checkpoints: dict[int, SelectedCheckpoint] = {}
        selected_learning_rate = config.learning_rates[0]
        selected_mean = float("inf")
        for learning_rate in config.learning_rates:
            candidates: dict[int, _FittedTrial] = {}
            for seed in config.seeds:
                identity = TrialCacheIdentity(
                    schema_version=_TRIAL_CACHE_SCHEMA,
                    dataset=series.name,
                    dataset_sha256=series.sha256,
                    split_ranges=split_ranges,
                    config_fingerprint=config_fingerprint,
                    method_id=method_id,
                    seed=seed,
                    learning_rate=learning_rate,
                    template_fingerprint=template_fingerprints[seed],
                )
                fitted = cache.load(identity) if cache is not None else None
                if fitted is None:
                    fitted = _fit_trial(
                        templates[seed],
                        protocol_series,
                        windows,
                        method_id=method_id,
                        seed=seed,
                        learning_rate=learning_rate,
                        template_fingerprint=template_fingerprints[seed],
                        config=config,
                        device=device,
                    )
                    if cache is not None:
                        cache.store(identity, fitted)
                trial_records[seed].append(fitted.record)
                candidates[seed] = fitted
            if not all(_trial_succeeded(candidates[seed]) for seed in config.seeds):
                continue
            candidate_mean = sum(
                _successful_validation_mse(candidates[seed]) for seed in config.seeds
            ) / len(config.seeds)
            if candidate_mean < selected_mean:
                selected_mean = candidate_mean
                selected_learning_rate = learning_rate
                selected_checkpoints = {
                    seed: SelectedCheckpoint(
                        record=candidates[seed].record,
                        trainable_state=_successful_checkpoint(candidates[seed]),
                    )
                    for seed in config.seeds
                }
        if not selected_checkpoints:
            failures = "; ".join(
                f"{trial.method_id}/seed{trial.seed}/lr{trial.learning_rate:g}: "
                f"{trial.error or 'failed'}"
                for seed in config.seeds
                for trial in trial_records[seed]
                if trial.status == "failed"
            )
            raise RuntimeError(
                f"No learning rate completed successfully for every seed in {method_id}. "
                f"Failures: {failures or 'none recorded'}"
            )
        methods.append(
            MethodTuning(
                method_id=method_id,
                selected_learning_rate=selected_learning_rate,
                selection_mean_validation_mse=selected_mean,
                trials=tuple(
                    trial
                    for seed in config.seeds
                    for trial in trial_records[seed]
                ),
                checkpoints=tuple(selected_checkpoints[seed] for seed in config.seeds),
            )
        )
    return TuningResult(
        dataset=series.name,
        dataset_family=series.family,
        dataset_sha256=series.sha256,
        config=config,
        split_policy=split_policy,
        split_ranges=split_ranges,
        methods=tuple(methods),
    )


def test_time_peft(
    template: AdaptableForecaster | Callable[[int], AdaptableForecaster],
    series: DatasetSeries,
    tuning: TuningResult,
    *,
    device: str | None = None,
) -> ReproductionResult:
    """Evaluate each validation-selected seed checkpoint on test exactly once."""

    _validate_tuning_identity(tuning, series)
    config = tuning.config
    target_device = _resolve_device(device or config.device)
    _validate_template_source(template, config)
    protocol_series, split_policy = _paper_protocol_series(series, config)
    split_ranges = tuple(
        (split.name, split.start, split.end) for split in protocol_series.splits
    )
    if split_policy != tuning.split_policy or split_ranges != tuning.split_ranges:
        raise ValueError("Current reproduction splits do not match the tuning artifact")

    standardizer = fit_train_standardizer(protocol_series)
    caps = config.smoke_caps or SmokeCaps()
    test_windows = build_split_windows(
        protocol_series,
        "test",
        lookback=config.lookback,
        horizon=config.horizon,
        standardizer=standardizer,
        max_windows=caps.test_windows,
    )
    method_tuning = {method.method_id: method for method in tuning.methods}
    if set(method_tuning) != set(PAPER_METHOD_IDS):
        raise ValueError("Tuning artifact must contain exactly L and LFC")

    records: list[MethodRecord] = []
    for seed in config.seeds:
        seed_template = _template_for_seed(template, seed, series, config, target_device)
        template_fingerprint = _state_fingerprint(seed_template)
        for method_id in PAPER_METHOD_IDS:
            method = method_tuning[method_id]
            checkpoints = {
                checkpoint.record.seed: checkpoint for checkpoint in method.checkpoints
            }
            if set(checkpoints) != set(config.seeds):
                raise ValueError(f"Tuning checkpoints for {method_id} do not match seeds")
            records.append(
                _test_selected_checkpoint(
                    seed_template,
                    protocol_series,
                    test_windows,
                    method_id=method_id,
                    seed=seed,
                    selected=checkpoints[seed],
                    trials=tuple(trial for trial in method.trials if trial.seed == seed),
                    selection_mean_validation_mse=method.selection_mean_validation_mse,
                    template_fingerprint=template_fingerprint,
                    config=config,
                    device=target_device,
                )
            )
    return ReproductionResult(
        dataset=series.name,
        dataset_family=series.family,
        dataset_sha256=series.sha256,
        lookback=config.lookback,
        horizon=config.horizon,
        precision=config.precision,
        split_policy=split_policy,
        split_ranges=split_ranges,
        records=tuple(records),
        smoke_caps=config.smoke_caps,
    )


# Prevent pytest from collecting the public stage function when a test module
# imports it directly.
test_time_peft.__test__ = False


def save_tuning_result(tuning: TuningResult, path: str | Path) -> None:
    """Persist validation metadata and selected trainable checkpoints together."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuning, target)


def load_tuning_result(path: str | Path) -> TuningResult:
    """Load a tune-stage artifact onto CPU for later test evaluation."""

    result = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(result, TuningResult):
        raise TypeError("Tune artifact does not contain a TuningResult")
    return result


def aggregate_method_records(records: Iterable[MethodRecord]) -> tuple[MethodAggregate, ...]:
    """Aggregate independently produced method records without pooling windows."""

    grouped: dict[str, list[MethodRecord]] = defaultdict(list)
    for record in records:
        if record.method_id not in PAPER_METHOD_IDS:
            raise ValueError(f"Unexpected method {record.method_id!r}; expected L or LFC")
        _assert_record_finite(record)
        grouped[record.method_id].append(record)
    if not grouped:
        raise ValueError("At least one method record is required")

    aggregates: list[MethodAggregate] = []
    for method_id in PAPER_METHOD_IDS:
        method_records = grouped.get(method_id)
        if not method_records:
            continue
        validation = torch.tensor(
            [record.validation_mse for record in method_records], dtype=torch.float64
        )
        test_mse = torch.tensor([record.test_mse for record in method_records], dtype=torch.float64)
        test_mae = torch.tensor([record.test_mae for record in method_records], dtype=torch.float64)
        rates = Counter(record.selected_learning_rate for record in method_records)
        aggregates.append(
            MethodAggregate(
                method_id=method_id,
                runs=len(method_records),
                mean_validation_mse=float(validation.mean()),
                mean_test_mse=float(test_mse.mean()),
                std_test_mse=float(test_mse.std(correction=0)),
                mean_test_mae=float(test_mae.mean()),
                std_test_mae=float(test_mae.std(correction=0)),
                selected_learning_rates=tuple(sorted(rates.items(), reverse=True)),
            )
        )
    return tuple(aggregates)


def aggregate_reproduction_results(
    results: Iterable[ReproductionResult],
) -> tuple[MethodAggregate, ...]:
    """Flatten multiple dataset results and aggregate them by method."""

    return aggregate_method_records(record for result in results for record in result.records)


def _test_selected_checkpoint(
    template: AdaptableForecaster,
    series: DatasetSeries,
    test_windows: SplitWindows,
    *,
    method_id: str,
    seed: int,
    selected: SelectedCheckpoint,
    trials: tuple[TrialRecord, ...],
    selection_mean_validation_mse: float,
    template_fingerprint: str,
    config: TimePEFTReproductionConfig,
    device: torch.device,
) -> MethodRecord:
    if method_id not in PAPER_METHOD_IDS:
        raise ValueError(f"Paper reproduction supports only {PAPER_METHOD_IDS}, not {method_id!r}")
    if len(trials) != len(config.learning_rates):
        raise RuntimeError("Tuning records do not match the configured learning-rate grid")
    record = selected.record
    if record.method_id != method_id or record.seed != seed:
        raise ValueError("Selected checkpoint identity does not match method and seed")
    action = TIME_PEFT_ACTION_BY_ID[method_id]
    with _seeded_torch_rng(seed, device):
        model = model_for_action(template, action)
        if template_fingerprint != record.initialization_fingerprint:
            raise ValueError("Test-stage model initialization does not match the tune stage")
        model = model.to(device=device, dtype=torch.float32)
        _load_trainable_state(model, selected.trainable_state)
        test = _evaluate(
            model,
            test_windows,
            batch_size=config.batch_size,
            device=device,
            max_batches=_cap(config, "evaluation_batches"),
        )
    return MethodRecord(
        dataset=series.name,
        dataset_family=series.family,
        dataset_sha256=series.sha256,
        method_id=method_id,
        seed=seed,
        selected_learning_rate=record.learning_rate,
        selected_epoch=record.best_epoch,
        selection_mean_validation_mse=selection_mean_validation_mse,
        validation_mse=_successful_validation_mse(selected),
        validation_mae=_successful_validation_mae(selected),
        test_mse=test.mse,
        test_mae=test.mae,
        trainable_parameters=record.trainable_parameters,
        total_parameters=record.total_parameters,
        test_windows=test.examples,
        test_evaluations=1,
        smoke_caps_active=config.smoke_caps is not None and config.smoke_caps.active,
        trials=trials,
    )


def _fit_trial(
    template: AdaptableForecaster,
    series: DatasetSeries,
    windows: TuningWindows,
    *,
    method_id: str,
    seed: int,
    learning_rate: float,
    template_fingerprint: str,
    config: TimePEFTReproductionConfig,
    device: torch.device,
) -> _FittedTrial:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    trainable = 0
    total = 0
    best_mse = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_completed = 0
    epoch_metrics: list[EpochMetrics] = []
    epoch_limit = min(config.max_epochs, _cap(config, "epochs") or config.max_epochs)
    failure: FloatingPointError | torch.cuda.OutOfMemoryError | None = None
    action = TIME_PEFT_ACTION_BY_ID[method_id]
    try:
        with _seeded_torch_rng(seed, device):
            model = model_for_action(template, action)
            model = model.to(device=device, dtype=torch.float32)
            trainable, total = count_parameters(model)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError(f"Method {method_id} has no trainable parameters")
            # A trial has one LR and one AdamW optimizer. There are deliberately
            # no per-module groups with different learning rates.
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            stale_epochs = 0

            for epoch in range(1, epoch_limit + 1):
                model.train()
                order = torch.randperm(len(windows.train), generator=generator)
                max_batches = _cap(config, "batches_per_epoch")
                train_squared_error = 0.0
                train_elements = 0
                optimizer_steps = 0
                for batch_number, offset in enumerate(
                    range(0, len(order), config.batch_size), start=1
                ):
                    if max_batches is not None and batch_number > max_batches:
                        break
                    indices = order[offset : offset + config.batch_size]
                    x, y, mask = windows.train.batch(indices, device=device)
                    _assert_finite(y, "training targets")
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.predict(x, mask, config.horizon)
                    _assert_finite(prediction, f"{method_id} predictions at epoch {epoch}")
                    residual = prediction.float() - y.float()
                    loss = residual.square().mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"{method_id} loss became non-finite at epoch {epoch}, "
                            f"learning rate {learning_rate:g}"
                        )
                    train_squared_error += float(
                        residual.detach().square().sum(dtype=torch.float64)
                    )
                    train_elements += y.numel()
                    loss.backward()
                    if config.gradient_clip is not None:
                        gradient_norm = nn.utils.clip_grad_norm_(
                            parameters, config.gradient_clip
                        )
                        if not bool(torch.isfinite(gradient_norm)):
                            raise FloatingPointError(
                                f"{method_id} gradient norm became non-finite at epoch {epoch}, "
                                f"learning rate {learning_rate:g}"
                            )
                    else:
                        _assert_finite_gradients(parameters, method_id, epoch, learning_rate)
                    optimizer.step()
                    optimizer_steps += 1

                if train_elements == 0:
                    raise RuntimeError(f"Method {method_id} completed no training batches")
                validation = _evaluate(
                    model,
                    windows.validation,
                    batch_size=config.batch_size,
                    device=device,
                    max_batches=_cap(config, "evaluation_batches"),
                )
                train_mse = train_squared_error / train_elements
                if not math.isfinite(train_mse):
                    raise FloatingPointError(
                        f"{method_id} train MSE became non-finite at epoch {epoch}, "
                        f"learning rate {learning_rate:g}"
                    )
                epoch_metrics.append(
                    EpochMetrics(
                        epoch=epoch,
                        train_mse=train_mse,
                        validation_mse=validation.mse,
                        validation_mae=validation.mae,
                        optimizer_steps=optimizer_steps,
                    )
                )
                epochs_completed = epoch
                if validation.mse < best_mse - config.early_stopping_min_delta:
                    best_mse = validation.mse
                    best_mae = validation.mae
                    best_epoch = epoch
                    best_state = _trainable_cpu_state(model)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= config.early_stopping_patience:
                        break

            if best_state is None or best_epoch == 0:
                raise FloatingPointError(
                    f"Method {method_id} produced no finite validation checkpoint"
                )
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
        failure = error
        best_state = None
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_cuda_memory_mb = 0.0
    elapsed_time_s = time.perf_counter() - started
    successful = failure is None
    trial = TrialRecord(
        dataset=series.name,
        dataset_sha256=series.sha256,
        method_id=method_id,
        seed=seed,
        learning_rate=learning_rate,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        stopped_early=successful and epochs_completed < epoch_limit,
        validation_mse=best_mse if successful else None,
        validation_mae=best_mae if successful else None,
        trainable_parameters=trainable,
        total_parameters=total,
        train_windows=len(windows.train),
        validation_windows=len(windows.validation),
        initialization_fingerprint=template_fingerprint,
        batch_order_seed=seed,
        status="ok" if successful else "failed",
        error=None if successful else f"{type(failure).__name__}: {failure}",
        epoch_metrics=tuple(epoch_metrics),
        elapsed_time_s=elapsed_time_s,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
    )
    return _FittedTrial(record=trial, checkpoint=best_state)


@torch.no_grad()
def _evaluate(
    model: AdaptableForecaster,
    windows: SplitWindows,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
) -> _Metrics:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    examples = 0
    for batch_number, offset in enumerate(range(0, len(windows), batch_size), start=1):
        if max_batches is not None and batch_number > max_batches:
            break
        indices = torch.arange(offset, min(offset + batch_size, len(windows)))
        x, y, mask = windows.batch(indices, device=device)
        _assert_finite(y, f"{windows.split_name} targets")
        prediction = model.predict(x, mask, windows.horizon)
        _assert_finite(prediction, f"{windows.split_name} predictions")
        residual = prediction.float() - y.float()
        squared_error += float(residual.square().sum(dtype=torch.float64))
        absolute_error += float(residual.abs().sum(dtype=torch.float64))
        elements += y.numel()
        examples += y.shape[0]
    if elements == 0:
        raise RuntimeError(f"No {windows.split_name} examples were evaluated")
    mse = squared_error / elements
    mae = absolute_error / elements
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise FloatingPointError(f"{windows.split_name} metrics are non-finite")
    return _Metrics(mse=mse, mae=mae, examples=examples)


def _validate_template(
    template: AdaptableForecaster,
    series: DatasetSeries,
    config: TimePEFTReproductionConfig,
) -> None:
    if template.adapter_implementation not in {"paper", "paper_count_inferred"}:
        raise ValueError(
            "Time-PEFT reproduction requires adapter_implementation='paper' or "
            "'paper_count_inferred'"
        )
    if template.channels != series.values.shape[0]:
        raise ValueError(
            f"Template channel count {template.channels} does not match dataset channels "
            f"{series.values.shape[0]}"
        )
    max_horizon = getattr(template.backbone, "max_horizon", None)
    if max_horizon is not None and config.horizon > int(max_horizon):
        raise ValueError(
            f"Requested horizon {config.horizon} exceeds backbone maximum {max_horizon}"
        )


def _validate_tuning_identity(tuning: TuningResult, series: DatasetSeries) -> None:
    expected = (series.name, series.family, series.sha256)
    observed = (tuning.dataset, tuning.dataset_family, tuning.dataset_sha256)
    if observed != expected:
        raise ValueError("Tuning artifact does not match the requested dataset")
    if tuple(method.method_id for method in tuning.methods) != PAPER_METHOD_IDS:
        raise ValueError("Tuning artifact must contain L and LFC exactly once in order")
    for method in tuning.methods:
        if method.selected_learning_rate not in tuning.config.learning_rates:
            raise ValueError(f"Selected learning rate for {method.method_id} is outside the grid")
        if not math.isfinite(method.selection_mean_validation_mse):
            raise ValueError(f"Selection metric for {method.method_id} is non-finite")
        expected_trials = len(tuning.config.seeds) * len(tuning.config.learning_rates)
        if len(method.trials) != expected_trials:
            raise ValueError(f"Tuning trial grid for {method.method_id} is incomplete")
        if len(method.checkpoints) != len(tuning.config.seeds):
            raise ValueError(f"Selected checkpoints for {method.method_id} are incomplete")
        trial_grid = {
            (trial.seed, trial.learning_rate, trial.method_id) for trial in method.trials
        }
        expected_grid = {
            (seed, learning_rate, method.method_id)
            for seed in tuning.config.seeds
            for learning_rate in tuning.config.learning_rates
        }
        if trial_grid != expected_grid:
            raise ValueError(f"Tuning trial identities for {method.method_id} are invalid")
        checkpoint_seeds = {checkpoint.record.seed for checkpoint in method.checkpoints}
        if checkpoint_seeds != set(tuning.config.seeds):
            raise ValueError(f"Checkpoint seeds for {method.method_id} are invalid")
        if any(
            checkpoint.record.method_id != method.method_id
            or checkpoint.record.learning_rate != method.selected_learning_rate
            or checkpoint.record.status != "ok"
            for checkpoint in method.checkpoints
        ):
            raise ValueError(f"Selected checkpoints for {method.method_id} are inconsistent")


def _validate_template_source(
    template: AdaptableForecaster | Callable[[int], AdaptableForecaster],
    config: TimePEFTReproductionConfig,
) -> None:
    if isinstance(template, AdaptableForecaster):
        if len(config.seeds) != 1:
            raise ValueError(
                "Multiple seeds require a template factory callable(seed) so model "
                "initialization can vary reproducibly"
            )
    elif not callable(template):
        raise TypeError("template must be an AdaptableForecaster or callable(seed)")


def _template_for_seed(
    template: AdaptableForecaster | Callable[[int], AdaptableForecaster],
    seed: int,
    series: DatasetSeries,
    config: TimePEFTReproductionConfig,
    device: torch.device,
) -> AdaptableForecaster:
    if isinstance(template, AdaptableForecaster):
        candidate = template
    else:
        with _seeded_torch_rng(seed, device):
            candidate = template(seed)
    if not isinstance(candidate, AdaptableForecaster):
        raise TypeError("The template factory must return an AdaptableForecaster")
    _validate_template(candidate, series, config)
    return candidate


def _paper_protocol_series(
    series: DatasetSeries,
    config: TimePEFTReproductionConfig,
) -> tuple[DatasetSeries, str]:
    if not config.complex_split_override_70_10_20 or series.family not in _COMPLEX_FAMILIES:
        return series, "dataset-series"
    length = series.values.shape[1]
    train_end = int(length * 0.7)
    validation_end = length - int(length * 0.2)
    if not 0 < train_end < validation_end < length:
        raise ValueError(
            f"Dataset {series.name} is too short for a chronological 70/10/20 split"
        )
    overridden = DatasetSeries(
        name=series.name,
        family=series.family,
        values=series.values,
        splits=(
            DatasetSplit("train", 0, train_end),
            DatasetSplit("validation", train_end, validation_end),
            DatasetSplit("test", validation_end, length),
        ),
        sha256=series.sha256,
    )
    return overridden, "complex-70/10/20-override"


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_trial_cache(
    value: TrialCache | str | Path | None,
) -> TrialCache | None:
    if value is None or isinstance(value, TrialCache):
        return value
    return TrialCache(value)


def _config_fingerprint(config: TimePEFTReproductionConfig) -> str:
    payload = {
        "schema_version": _TRIAL_CACHE_SCHEMA,
        "config": asdict(config),
        "implementation_hash": implementation_hash(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cap(config: TimePEFTReproductionConfig, field: str) -> int | None:
    if config.smoke_caps is None:
        return None
    return getattr(config.smoke_caps, field)


def _evenly_spaced_indices(length: int, requested: int) -> Tensor:
    if requested >= length:
        return torch.arange(length)
    if requested == 1:
        return torch.zeros(1, dtype=torch.long)
    numerator = torch.arange(requested, dtype=torch.long) * (length - 1)
    return torch.div(numerator, requested - 1, rounding_mode="floor")


def _assert_window_targets_finite(
    values: Tensor,
    starts: Tensor,
    horizon: int,
    *,
    dataset: str,
    split_name: str,
) -> None:
    invalid = (~torch.isfinite(values)).any(dim=0).to(torch.int64)
    prefix = torch.cat((torch.zeros(1, dtype=torch.int64), invalid.cumsum(dim=0)))
    invalid_counts = prefix[starts + horizon] - prefix[starts]
    if bool((invalid_counts > 0).any()):
        bad_index = int(torch.nonzero(invalid_counts > 0, as_tuple=False)[0])
        target_start = int(starts[bad_index])
        target = values[:, target_start : target_start + horizon]
        coordinate = torch.nonzero(~torch.isfinite(target), as_tuple=False)[0]
        channel = int(coordinate[0])
        timestamp = target_start + int(coordinate[1])
        raise ValueError(
            f"Dataset {dataset} has a non-finite {split_name} target at "
            f"channel {channel}, timestamp {timestamp}"
        )


def _assert_finite(value: Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contain NaN or infinity")


def _trainable_cpu_state(model: nn.Module) -> dict[str, Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    for name, value in state.items():
        _assert_finite(value, f"checkpoint parameter {name}")
    return state


def _trial_succeeded(fitted: _FittedTrial) -> bool:
    return (
        fitted.record.status == "ok"
        and fitted.record.validation_mse is not None
        and fitted.record.validation_mae is not None
        and fitted.checkpoint is not None
    )


def _successful_validation_mse(fitted: _FittedTrial | SelectedCheckpoint) -> float:
    value = fitted.record.validation_mse
    if fitted.record.status != "ok" or value is None or not math.isfinite(value):
        raise RuntimeError("Selected trial does not have a finite validation MSE")
    return value


def _successful_validation_mae(fitted: _FittedTrial | SelectedCheckpoint) -> float:
    value = fitted.record.validation_mae
    if fitted.record.status != "ok" or value is None or not math.isfinite(value):
        raise RuntimeError("Selected trial does not have a finite validation MAE")
    return value


def _successful_checkpoint(fitted: _FittedTrial) -> dict[str, Tensor]:
    if not _trial_succeeded(fitted) or fitted.checkpoint is None:
        raise RuntimeError("Selected trial does not have a successful checkpoint")
    return fitted.checkpoint


def _validate_cached_trial(cached: _CachedTrial) -> None:
    identity = cached.identity
    record = cached.record
    record.__post_init__()
    if identity.schema_version != _TRIAL_CACHE_SCHEMA:
        raise ValueError("Trial cache schema is incompatible")
    observed = (
        record.dataset,
        record.dataset_sha256,
        record.method_id,
        record.seed,
        record.learning_rate,
        record.initialization_fingerprint,
    )
    expected = (
        identity.dataset,
        identity.dataset_sha256,
        identity.method_id,
        identity.seed,
        identity.learning_rate,
        identity.template_fingerprint,
    )
    if observed != expected:
        raise ValueError("Cached trial record does not match its identity")
    if record.status == "ok":
        if cached.checkpoint is None or not cached.checkpoint:
            raise ValueError("Successful cached trial is missing its checkpoint")
        for name, value in cached.checkpoint.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise TypeError("Cached checkpoint must map parameter names to tensors")
            _assert_finite(value, f"cached checkpoint parameter {name}")
    elif cached.checkpoint is not None:
        raise ValueError("Failed cached trial must not contain a checkpoint")


def _load_trainable_state(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if state.keys() != expected:
        missing = sorted(expected - state.keys())
        extra = sorted(state.keys() - expected)
        raise RuntimeError(
            f"Selected checkpoint does not match trainable parameters; "
            f"missing={missing}, extra={extra}"
        )
    with torch.no_grad():
        for name, value in state.items():
            target = parameters[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def _state_fingerprint(model: nn.Module) -> str:
    """Hash every template parameter and buffer for strict cache/test binding."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if tensor.numel():
            raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy()
            digest.update(raw)
    return digest.hexdigest()


def _assert_finite_gradients(
    parameters: Sequence[nn.Parameter],
    method_id: str,
    epoch: int,
    learning_rate: float,
) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(
                f"{method_id} gradients became non-finite at epoch {epoch}, "
                f"learning rate {learning_rate:g}"
            )


@contextmanager
def _seeded_torch_rng(seed: int, device: torch.device) -> Iterator[None]:
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def _assert_record_finite(record: MethodRecord) -> None:
    values = (
        record.selection_mean_validation_mse,
        record.validation_mse,
        record.validation_mae,
        record.test_mse,
        record.test_mae,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Method record for {record.method_id} contains non-finite metrics")
