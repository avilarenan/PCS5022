from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utility_peft.data.datasets import (
    DatasetSeries,
    DatasetSplit,
    available_datasets,
    load_dataset_manifest,
    load_dataset_series,
)


def test_lorenz_is_deterministic_and_has_exact_project_splits(tmp_path: Path) -> None:
    first = load_dataset_series("Lorenz", tmp_path, lorenz_length=2_000)
    second = load_dataset_series("lorenz-63", tmp_path, lorenz_length=2_000)

    assert first.name == "Lorenz"
    assert first.family == "synthetic"
    assert first.values.shape == (3, 2_000)
    assert torch.equal(first.values, second.values)
    assert first.sha256 == second.sha256
    assert [(split.name, split.start, split.end) for split in first.splits] == [
        ("train", 0, 1_200),
        ("validation", 1_200, 1_600),
        ("test", 1_600, 2_000),
    ]
    assert first.split("val") == first.split("validation")


@pytest.mark.parametrize(
    ("name", "alias", "channels"),
    [
        ("CellCycle", "cell_cycle", 6),
        ("DoublePendulum", "double-pendulum", 4),
        ("Hopfield", "hopfield-network", 6),
        ("LorenzCoupled", "coupled_lorenz", 6),
    ],
)
def test_compatible_chaotic_generators_are_finite_and_aliasable(
    tmp_path: Path, name: str, alias: str, channels: int
) -> None:
    canonical = load_dataset_series(name, tmp_path, synthetic_length=80)
    aliased = load_dataset_series(alias, tmp_path, synthetic_length=80)

    assert canonical.values.shape == (channels, 80)
    assert torch.isfinite(canonical.values).all()
    assert torch.equal(canonical.values, aliased.values)
    assert canonical.sha256 == aliased.sha256


def test_local_weather_csv_is_channels_first_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "custom-weather.csv"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=100, freq="10min"),
            **{f"signal-{index}": np.arange(100) + index for index in range(21)},
        }
    )
    frame.to_csv(path, index=False)

    dataset = load_dataset_series("weather", tmp_path, local_path=path)

    assert dataset.name == "Weather"
    assert dataset.values.shape == (21, 100)
    assert dataset.values.dtype == torch.float32
    assert dataset.split("train") == DatasetSplit("train", 0, 70)
    assert dataset.split("validation") == DatasetSplit("validation", 70, 80)
    assert dataset.split("test") == DatasetSplit("test", 80, 100)
    assert dataset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_ett_uses_fixed_official_month_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "ETTh1.csv"
    frame = pd.DataFrame(
        np.arange(14_400 * 7, dtype=np.float32).reshape(14_400, 7),
        columns=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    )
    frame.insert(0, "date", pd.date_range("2016-01-01", periods=14_400, freq="h"))
    frame.to_csv(path, index=False)

    dataset = load_dataset_series("ett_h1", tmp_path, local_path=path)

    assert dataset.values.shape == (7, 14_400)
    assert [(split.start, split.end) for split in dataset.splits] == [
        (0, 8_640),
        (8_640, 11_520),
        (11_520, 14_400),
    ]


def test_short_or_malformed_csv_has_actionable_errors(tmp_path: Path) -> None:
    wrong_channels = tmp_path / "wrong-weather.csv"
    pd.DataFrame({"date": ["2026-01-01"] * 20, "only": range(20)}).to_csv(
        wrong_channels, index=False
    )
    with pytest.raises(ValueError, match="requires 21 signal channels"):
        load_dataset_series("Weather", tmp_path, local_path=wrong_channels)

    short_ett = tmp_path / "short-ett.csv"
    pd.DataFrame(np.ones((100, 7))).to_csv(short_ett, index=False)
    with pytest.raises(ValueError, match="insufficient for configured boundaries"):
        load_dataset_series("ETTh1", tmp_path, local_path=short_ett)


def test_manifest_covers_time_peft_suite_and_source_only_electricity() -> None:
    manifest = load_dataset_manifest()
    expected = {
        "Lorenz",
        "CellCycle",
        "DoublePendulum",
        "Hopfield",
        "LorenzCoupled",
        "ECGCA115",
        "ECGCA515",
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "Weather",
        "Exchange",
        "Electricity",
    }
    assert set(available_datasets()) == expected
    for name in expected - {
        "Lorenz",
        "CellCycle",
        "DoublePendulum",
        "Hopfield",
        "LorenzCoupled",
    }:
        digest = manifest["datasets"][name]["sha256"]
        assert len(digest) == 64
    assert manifest["datasets"]["Electricity"]["role"].startswith("source-head-only")


def test_missing_public_dataset_error_explains_both_acquisition_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="download=True") as caught:
        load_dataset_series("Exchange", tmp_path)
    assert "local_path=" in str(caught.value)


def test_dataset_series_rejects_overlapping_partitions() -> None:
    values = torch.ones(2, 30)
    with pytest.raises(ValueError, match="overlap"):
        DatasetSeries(
            name="bad",
            family="test",
            values=values,
            splits=(
                DatasetSplit("train", 0, 15),
                DatasetSplit("validation", 14, 20),
                DatasetSplit("test", 20, 30),
            ),
            sha256="0" * 64,
        )


def test_unknown_dataset_lists_canonical_choices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="available datasets"):
        load_dataset_series("not-a-dataset", tmp_path)
