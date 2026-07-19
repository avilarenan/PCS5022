"""Immutable public data contracts used across the Utility-PEFT pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard action constraints. ``None`` means unconstrained."""

    max_trainable_parameters: int | None = None
    max_trainable_fraction: float | None = None
    max_peak_memory_mb: float | None = None
    max_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.max_trainable_parameters,
            self.max_trainable_fraction,
            self.max_peak_memory_mb,
            self.max_wall_time_s,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("Budget limits must be non-negative")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One adaptation operation and its fixed training budget."""

    action_id: str
    modules: frozenset[str]
    rank: int | None = None
    alpha: int | None = None
    target_modules: tuple[str, ...] = ("q", "v")
    update_steps: int = 100
    budget: Budget = field(default_factory=Budget)
    controller_action: bool = True

    def __post_init__(self) -> None:
        allowed = {"head", "lora", "frequency", "channel", "fourierft", "full"}
        unknown = self.modules - allowed
        if unknown:
            raise ValueError(f"Unknown action modules: {sorted(unknown)}")
        if self.update_steps < 0:
            raise ValueError("update_steps must be non-negative")
        if "lora" in self.modules and (self.rank is None or self.rank <= 0):
            raise ValueError("LoRA actions require a positive rank")


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Reproducible, leakage-auditable description of one episode."""

    episode_id: str
    dataset: str
    dataset_family: str
    lookback: int
    horizon: int
    support_size: int
    query_size: int
    support_start: int
    support_end: int
    query_start: int
    query_end: int
    seed: int
    preprocessing_hash: str
    subject_id: str | None = None
    session_id: str | None = None
    source_hash: str = ""
    partition: str = "full"

    def __post_init__(self) -> None:
        if self.support_start < 0 or self.support_end <= self.support_start:
            raise ValueError("Invalid support interval")
        if self.query_start < self.support_end:
            raise ValueError("Query interval overlaps support interval")
        if self.query_end <= self.query_start:
            raise ValueError("Invalid query interval")
        if self.support_size <= 0 or self.query_size <= 0:
            raise ValueError("Episode sets must be non-empty")
        if not self.partition:
            raise ValueError("Episode partition must be non-empty")


@dataclass(frozen=True, slots=True)
class SupportView:
    """The only episode view accepted by selection-time components."""

    x: Tensor
    y: Tensor
    mask: Tensor
    manifest: EpisodeManifest

    def __post_init__(self) -> None:
        _validate_forecasting_tensors(self.x, self.y, self.mask, self.manifest.horizon)


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    """Support data plus query tensors reserved for evaluation code."""

    support: SupportView
    query_x: Tensor
    query_y: Tensor
    query_mask: Tensor

    def __post_init__(self) -> None:
        _validate_forecasting_tensors(
            self.query_x,
            self.query_y,
            self.query_mask,
            self.support.manifest.horizon,
        )


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Named support-only evidence consumed by the controller."""

    episode_id: str
    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("Evidence names and values must have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Evidence feature names must be unique")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))

    @classmethod
    def from_mapping(cls, episode_id: str, values: Mapping[str, float]) -> EvidenceBundle:
        ordered = sorted((str(name), float(value)) for name, value in values.items())
        return cls(
            episode_id=episode_id,
            names=tuple(name for name, _ in ordered),
            values=tuple(value for _, value in ordered),
        )


@dataclass(frozen=True, slots=True)
class UtilityRecord:
    """Immutable raw result for one episode/action/seed evaluation."""

    episode_id: str
    dataset: str
    dataset_family: str
    horizon: int
    action_id: str
    seed: int
    frozen_loss: float
    adapted_loss: float
    normalized_gain: float
    trainable_parameters: int
    stored_adapter_parameters: int
    total_parameters: int
    profiled_flops: float
    peak_memory_mb: float
    wall_time_s: float
    evidence: Mapping[str, float]
    config_hash: str
    model_revision: str
    preprocessing_hash: str
    status: str = "ok"
    error: str | None = None
    frozen_mae: float | None = None
    adapted_mae: float | None = None
    evidence_wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in {"ok", "failed"}:
            raise ValueError("Utility record status must be 'ok' or 'failed'")
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType({str(name): float(value) for name, value in self.evidence.items()}),
        )

    @property
    def key(self) -> tuple[str, int, str, str, int, str, str, str]:
        return (
            self.dataset,
            self.horizon,
            self.episode_id,
            self.action_id,
            self.seed,
            self.config_hash,
            self.model_revision,
            self.preprocessing_hash,
        )

    def utility(self, cost_weights: Mapping[str, float] | None = None) -> float:
        weights = cost_weights or {}
        costs = {
            "parameters": self.trainable_parameters / max(self.total_parameters, 1),
            "flops": self.profiled_flops,
            "memory": self.peak_memory_mb,
            "time": self.wall_time_s,
        }
        penalty = sum(weights.get(name, 0.0) * value for name, value in costs.items())
        return self.normalized_gain - penalty

    def to_flat_dict(self) -> dict[str, Any]:
        row = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if field_name != "evidence"
        }
        row["evidence"] = dict(self.evidence)
        return row


def _validate_forecasting_tensors(x: Tensor, y: Tensor, mask: Tensor, horizon: int) -> None:
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("Forecasting tensors must have shape [batch, channels, time]")
    if mask.ndim != 2 or mask.shape != (x.shape[0], x.shape[2]):
        raise ValueError("Mask must have shape [batch, lookback]")
    if x.shape[:2] != y.shape[:2] or y.shape[2] != horizon:
        raise ValueError("Input/output channel dimensions or forecast horizon do not match")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("Forecasting tensors must be floating point")
    if mask.dtype is not torch.bool:
        raise TypeError("Input masks must use torch.bool")
