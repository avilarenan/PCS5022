from __future__ import annotations

from importlib.metadata import version

import torch
from typer.testing import CliRunner

from utility_peft.actions import ACTION_BY_ID
from utility_peft.backbones.base import BackboneProtocol
from utility_peft.backbones.moment import MOMENT_MODEL_REVISION
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.cli import app
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
        "run-correlation-benchmark",
        "reproduce",
        "generate-utilities",
        "train-controller",
        "evaluate-heldout",
        "build-report",
    ):
        assert command in result.stdout
