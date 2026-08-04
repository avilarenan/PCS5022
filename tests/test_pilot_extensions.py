from __future__ import annotations

import math

import pytest
import torch
from typer.testing import CliRunner

from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.cli import app
from utility_peft.config import load_config
from utility_peft.controller import ControllerTrainingConfig
from utility_peft.data.datasets import load_dataset_series
from utility_peft.episodes import chronological_starts_in_range
from utility_peft.evidence import lagged_transfer_entropy, select_feature_mapping
from utility_peft.lodo import ABLATION_FEATURE_SETS, evaluate_leave_one_dataset_out
from utility_peft.model import AdaptableForecaster
from utility_peft.source_head import (
    SOURCE_HEAD_PREPROCESSING_HASH,
    SourceHeadTrainingConfig,
    train_source_head,
    validate_source_head_provenance,
)
from utility_peft.types import UtilityRecord


def test_official_partitions_and_episode_starts_remain_inside_test_split() -> None:
    dataset = load_dataset_series("Lorenz", "unused", lorenz_length=2_000)
    assert [(split.name, split.start, split.end) for split in dataset.splits] == [
        ("train", 0, 1_200),
        ("validation", 1_200, 1_600),
        ("test", 1_600, 2_000),
    ]
    test = dataset.split("test")
    starts = chronological_starts_in_range(
        test.start,
        test.end,
        lookback=16,
        horizon=8,
        support_size=8,
        query_size=8,
        episodes=3,
    )
    assert min(starts) >= test.start
    assert max(starts) < test.end


def test_lagged_transfer_entropy_recovers_directed_information_flow() -> None:
    generator = torch.Generator().manual_seed(4)
    source = torch.randn(32, 48, generator=generator)
    target = torch.zeros_like(source)
    target[:, 1:] = source[:, :-1] + 0.02 * torch.randn(
        32, 47, generator=generator
    )
    transfer = lagged_transfer_entropy(torch.stack((source, target), dim=1))
    assert transfer.shape == (2, 2)
    assert transfer[0, 1] > transfer[1, 0]
    assert transfer.diag().count_nonzero() == 0


def test_preregistered_feature_sets_exclude_model_aware_evidence() -> None:
    evidence = {
        "spectral_entropy": 0.3,
        "transfer_entropy_mean": 0.2,
        "lag1_autocorrelation": 0.5,
        "frozen_mse": 1.0,
        "gradient_query_l2": 2.0,
    }
    assert set(select_feature_mapping(evidence, "complexity")) == {
        "spectral_entropy",
        "transfer_entropy_mean",
    }
    assert "frozen_mse" not in select_feature_mapping(evidence, "structure")
    assert "frozen_mse" in select_feature_mapping(evidence, "structure_mismatch")
    assert "gradient_query_l2" not in select_feature_mapping(
        evidence, "structure_mismatch"
    )
    assert select_feature_mapping(evidence, "full") == evidence


def test_source_head_training_records_and_validates_dataset_exclusion(tmp_path) -> None:
    dataset = load_dataset_series("Lorenz", tmp_path, lorenz_length=1_000)
    model = AdaptableForecaster(
        TinyBackbone(d_model=8, patch_len=4, depth=1, heads=2, max_horizon=8)
    )
    checkpoint = tmp_path / "head-h8.pt"
    metrics = train_source_head(
        model,
        dataset,
        horizon=8,
        lookback=16,
        checkpoint_path=checkpoint,
        evaluation_datasets=("target-a", "target-b"),
        config=SourceHeadTrainingConfig(
            updates=2,
            batch_size=2,
            validation_windows=4,
            validation_interval=1,
            bf16=False,
        ),
        device="cpu",
    )
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".metrics.json").exists()
    assert metrics.source_dataset == "Lorenz"
    assert metrics.source_dataset_sha256 == dataset.sha256
    assert metrics.preprocessing_hash == SOURCE_HEAD_PREPROCESSING_HASH
    assert len(metrics.scaler_statistics_sha256) == 64
    restored = validate_source_head_provenance(
        checkpoint,
        horizon=8,
        evaluation_datasets=("target-a", "target-b"),
    )
    assert restored.checkpoint_sha256 == metrics.checkpoint_sha256
    with pytest.raises(ValueError, match="also an evaluation dataset"):
        train_source_head(
            model,
            dataset,
            horizon=8,
            lookback=16,
            checkpoint_path=tmp_path / "invalid.pt",
            evaluation_datasets=("Lorenz",),
            config=SourceHeadTrainingConfig(
                updates=1,
                batch_size=1,
                validation_windows=1,
                validation_interval=1,
                bf16=False,
            ),
            device="cpu",
        )


def test_pilot_config_has_preregistered_record_counts_and_claim_guard() -> None:
    config = load_config("pilot", ["model=tiny", "device=cpu"])
    groups = (
        len(config.experiment.datasets)
        * len(config.experiment.horizons)
        * config.experiment.episodes_per_dataset_horizon
    )
    assert groups * len(config.experiment.actions) * len(config.experiment.seeds) == 336
    assert groups * len(config.experiment.actions) * 3 == 504
    assert len(config.experiment.datasets) * len(config.experiment.horizons) == 8
    result = CliRunner().invoke(
        app,
        [
            "reproduce-time-peft",
            "--config",
            "pilot",
            "--protocol",
            "paper",
            "-o",
            "model=tiny",
            "-o",
            "device=cpu",
        ],
    )
    assert result.exit_code != 0
    assert "blocked until official parity" in result.stdout
    assert "is verified" in result.stdout


def test_nested_lodo_emits_ablation_and_matched_baseline_metrics(tmp_path) -> None:
    records = []
    for dataset_index, dataset in enumerate(("d0", "d1", "d2")):
        for episode_index in range(2):
            spectral = (dataset_index + episode_index) / 3
            evidence = {
                "spectral_entropy": spectral,
                "transfer_entropy_mean": 1 - spectral,
                "lag1_autocorrelation": 0.2 + spectral,
                "frozen_mse": 1.5 - spectral,
                "gradient_query_l2": 0.5 + spectral,
            }
            winner = "A3" if spectral >= 0.5 else "A4"
            for action_index in range(7):
                action = f"A{action_index}"
                gain = 0.8 if action == winner else 0.1 + action_index * 0.01
                for seed in (0, 1):
                    noisy_gain = gain + seed * 0.001
                    records.append(
                        _utility_record(
                            dataset,
                            f"{dataset}-episode-{episode_index}",
                            action,
                            noisy_gain,
                            evidence,
                            seed,
                        )
                    )
    result = evaluate_leave_one_dataset_out(
        records,
        tmp_path,
        config=ControllerTrainingConfig(
            hidden_size=8,
            action_embedding_size=2,
            dropout=0.0,
            epochs=2,
            seed=2,
            device="cpu",
        ),
        bootstrap_samples=100,
        expected_seeds=(0, 1),
    )
    assert len(result.folds) == 3
    assert set(result.folds[0].ablations) == set(ABLATION_FEATURE_SETS)
    assert math.isfinite(result.superiority.mean_relative_mse_difference)
    assert (tmp_path / "lodo_metrics.json").exists()
    for fold in result.folds:
        assert "d" in fold.heldout_dataset
        assert fold.episodes == 2


def test_nested_lodo_rejects_disjoint_action_seed_sets(tmp_path) -> None:
    evidence = {"spectral_entropy": 0.5}
    records = [
        _utility_record(
            "d0",
            "d0-episode-0",
            f"A{action_index}",
            0.1,
            evidence,
            action_index,
        )
        for action_index in range(7)
    ]

    with pytest.raises(ValueError, match=r"Exact seed pairing.*d0-episode-0"):
        evaluate_leave_one_dataset_out(records, tmp_path, bootstrap_samples=10)


def _utility_record(
    dataset: str,
    episode_id: str,
    action: str,
    gain: float,
    evidence: dict[str, float],
    seed: int,
) -> UtilityRecord:
    action_index = int(action[1:])
    frozen_loss = 2.0
    return UtilityRecord(
        episode_id=episode_id,
        dataset=dataset,
        dataset_family="test",
        horizon=96,
        action_id=action,
        seed=seed,
        frozen_loss=frozen_loss,
        adapted_loss=frozen_loss * (1 - gain),
        normalized_gain=gain,
        trainable_parameters=action_index * 10,
        stored_adapter_parameters=action_index * 10,
        total_parameters=1_000,
        profiled_flops=float(action_index),
        peak_memory_mb=float(action_index + 1),
        wall_time_s=0.1 * action_index,
        evidence=evidence,
        config_hash="pilot",
        model_revision="tiny",
        preprocessing_hash="preprocessing",
        frozen_mae=1.0,
        adapted_mae=1.0 - gain / 2,
        evidence_wall_time_s=0.01,
    )
