"""Backbone contract used by adaptation and evidence code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class BackboneProtocol(Protocol):
    """Minimal interface required from a forecasting foundation model."""

    d_model: int

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        """Return representations shaped [batch, channels, patches, embedding]."""

    def predict(self, x: Tensor, mask: Tensor, horizon: int) -> Tensor:
        """Return forecasts shaped [batch, channels, horizon]."""

    def predict_from_embeddings(self, embeddings: Tensor, horizon: int) -> Tensor:
        """Apply the forecasting head to canonical encoder representations."""

    def adapter_targets(self) -> tuple[str, ...]:
        """Return target leaf names for encoder linear adapters."""

    def source_statistics(self) -> Mapping[str, float]:
        """Return model/source metadata usable as non-target evidence."""

    def head_parameters(self) -> list[nn.Parameter]:
        """Return forecasting-head parameters only."""

    def save_source_head(self, path: str | Path) -> None:
        """Persist only the forecasting head for leakage-auditable reuse."""
