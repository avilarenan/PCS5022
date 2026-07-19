from __future__ import annotations

import pytest
import torch

from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.episodes import build_episode
from utility_peft.model import AdaptableForecaster
from utility_peft.utils import seed_everything


@pytest.fixture
def series() -> torch.Tensor:
    time = torch.linspace(0, 20, 240)
    return torch.stack(
        (
            torch.sin(time),
            torch.cos(time * 0.7),
            torch.sin(time * 0.2) + 0.1 * time,
        )
    )


@pytest.fixture
def episode(series: torch.Tensor):
    return build_episode(
        series,
        dataset="synthetic",
        dataset_family="test",
        lookback=16,
        horizon=8,
        support_size=8,
        query_size=8,
        start=11,
        seed=3,
    )


@pytest.fixture
def template() -> AdaptableForecaster:
    seed_everything(7)
    return AdaptableForecaster(
        TinyBackbone(
            d_model=16,
            patch_len=4,
            depth=1,
            heads=2,
            max_horizon=16,
        )
    )
