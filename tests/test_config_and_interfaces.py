from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from click import BadParameter
from typer.testing import CliRunner

from utility_peft.actions import ACTION_BY_ID
from utility_peft.backbones.base import BackboneProtocol
from utility_peft.backbones.moment import MOMENT_MODEL_REVISION
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.cli import (
    _configured_seed_tuple,
    _correlation_router_config,
    _validate_episode_cartesian_coverage,
    app,
)
from utility_peft.config import load_config


def test_backbone_protocol_and_canonical_representation_shape() -> None:
    backbone = TinyBackbone(d_model=16, patch_len=4, depth=1, heads=2, max_horizon=12)
    assert isinstance(backbone, BackboneProtocol)
    x = torch.randn(3, 2, 16)
    mask = torch.ones(3, 16, dtype=torch.bool)
    embeddings = backbone.encode(x, mask)
    assert embeddings.shape == (3, 2, 4, 16)
    assert backbone.predict(x, mask, 8).shape == (3, 2, 8)


def test_mvp_hydra_config_contains_exact_fixed_protocol() -> None:
    config = load_config("config", ["model=tiny", "device=cpu"])
    assert config.experiment.lookback == 96
    assert list(config.experiment.horizons) == [96, 192, 336]
    assert list(config.experiment.seeds) == [0, 1, 2]
    assert config.experiment.support_size == 64
    assert config.experiment.query_size == 128
    assert config.experiment.training.effective_batch_size == 32
    assert ACTION_BY_ID["A2"].rank == 8
    assert ACTION_BY_ID["A2"].alpha == 16
    assert ACTION_BY_ID["A2"].target_modules == ("q", "v")


def test_external_dependency_and_checkpoint_revisions_are_pinned() -> None:
    assert version("momentfm") == "0.1.5"
    assert version("transformers") == "4.54.1"
    assert version("peft") == "0.17.1"
    assert MOMENT_MODEL_REVISION == "5e44b0ea26376a176360f87831124e018f876d96"


def test_cli_exposes_required_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "prepare-data",
        "train-source-head",
        "reproduce-time-peft",
        "run-time-peft-reproduction",
        "run-correlation-benchmark",
        "reproduce",
        "generate-utilities",
        "train-controller",
        "evaluate-heldout",
        "build-report",
    ):
        assert command in result.stdout


def test_time_peft_reproduction_config_separates_paper_protocol_from_router() -> None:
    config = load_config("time_peft_reproduction")
    assert tuple(config.experiment.actions) == ("L", "LFC")
    assert tuple(config.experiment.horizons) == (96, 192, 336)
    assert tuple(config.experiment.seeds) == (0, 1, 2)
    assert len(config.experiment.datasets) == 7
    assert config.experiment.training.batch_size == 128
    assert config.experiment.training.max_epochs == 100
    assert tuple(config.experiment.training.learning_rates) == (0.001, 0.0001, 0.00001)
    assert config.experiment.training.gradient_clip is None
    assert config.model.source_head_checkpoint is None
    assert config.model.allow_random_head is True
    assert config.model.adapter_implementation == "paper_count_inferred"


def test_router_8h_config_wires_every_locked_correlation_knob() -> None:
    config = load_config("router_timepeft_8h")
    resolved = _correlation_router_config(config)
    assert resolved.regularization_c == config.correlation.regularization_c
    assert resolved.max_iter == config.correlation.max_iter
    assert resolved.bootstrap_samples == config.correlation.bootstrap_samples
    assert resolved.bootstrap_seed == config.correlation.bootstrap_seed
    assert resolved.bootstrap_confidence_level == config.correlation.bootstrap_confidence_level
    assert resolved.random_control_repeats == config.correlation.random_control_repeats
    assert resolved.frequency_features == tuple(config.correlation.frequency_features)
    assert resolved.channel_features == tuple(config.correlation.channel_features)


def test_episode_cartesian_coverage_rejects_balanced_total_with_missing_cell() -> None:
    def manifest(dataset: str, horizon: int, episode_id: str) -> SimpleNamespace:
        return SimpleNamespace(dataset=dataset, horizon=horizon, episode_id=episode_id)

    valid = [
        manifest(dataset, horizon, f"{dataset}-h{horizon}-{index}")
        for dataset in ("d0", "d1")
        for horizon in (96, 192)
        for index in range(2)
    ]
    _validate_episode_cartesian_coverage(
        valid,
        datasets={"d0", "d1"},
        horizons={96, 192},
        episodes_per_cell=2,
    )

    skewed = [item for item in valid if item.episode_id != "d1-h192-1"]
    skewed.append(manifest("d0", 96, "d0-h96-extra"))
    assert len(skewed) == len(valid)
    with pytest.raises(ValueError, match=r"d0/h96=3.*d1/h192=1"):
        _validate_episode_cartesian_coverage(
            skewed,
            datasets={"d0", "d1"},
            horizons={96, 192},
            episodes_per_cell=2,
        )


def test_episode_cartesian_coverage_rejects_duplicate_episode_ids() -> None:
    manifests = [
        SimpleNamespace(dataset="d0", horizon=96, episode_id="duplicate"),
        SimpleNamespace(dataset="d1", horizon=96, episode_id="duplicate"),
    ]
    with pytest.raises(ValueError, match="duplicate episode IDs: duplicate"):
        _validate_episode_cartesian_coverage(
            manifests,
            datasets={"d0", "d1"},
            horizons={96},
            episodes_per_cell=1,
        )


def test_configured_seed_tuple_is_nonempty_and_unique() -> None:
    assert _configured_seed_tuple(["2", 0]) == (2, 0)
    with pytest.raises(BadParameter, match="at least one"):
        _configured_seed_tuple([])
    with pytest.raises(BadParameter, match=r"duplicates: \[1\]"):
        _configured_seed_tuple([1, 1])
