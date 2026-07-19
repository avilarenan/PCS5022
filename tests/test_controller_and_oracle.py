from __future__ import annotations

import json

import pytest

from utility_peft.controller import (
    ControllerBundle,
    ControllerTrainingConfig,
    train_controller,
)
from utility_peft.oracle import evaluate_oracle_gate, require_oracle_gate
from utility_peft.types import Budget, EvidenceBundle, UtilityRecord


def record(
    dataset: str,
    episode_id: str,
    action_id: str,
    gain: float,
    evidence: dict[str, float],
    *,
    seed: int = 0,
) -> UtilityRecord:
    action_number = int(action_id[1:])
    return UtilityRecord(
        episode_id=episode_id,
        dataset=dataset,
        dataset_family="test",
        horizon=8,
        action_id=action_id,
        seed=seed,
        frozen_loss=2.0,
        adapted_loss=2.0 * (1 - gain),
        normalized_gain=gain,
        trainable_parameters=action_number * 10,
        stored_adapter_parameters=action_number * 10,
        total_parameters=1_000,
        profiled_flops=float(action_number),
        peak_memory_mb=float(action_number),
        wall_time_s=float(action_number),
        evidence=evidence,
        config_hash="config",
        model_revision="tiny",
        preprocessing_hash="preprocessing",
    )


def test_controller_trains_persists_and_filters_by_budget(tmp_path) -> None:
    records = []
    for dataset_index, dataset in enumerate(("source-a", "source-b")):
        for episode_index in range(4):
            signal = (dataset_index * 4 + episode_index) / 7
            evidence = {
                "spectral_entropy": signal,
                "mean_abs_channel_correlation": 1 - signal,
            }
            gains = {
                "A0": 0.0,
                "A1": 0.8 - signal,
                "A2": signal,
            }
            for action_id, gain in gains.items():
                for seed in (0, 1):
                    records.append(
                        record(
                            dataset,
                            f"{dataset}-{episode_index}",
                            action_id,
                            gain + seed * 0.001,
                            evidence,
                            seed=seed,
                        )
                    )
    path = tmp_path / "controller.pt"
    bundle, metrics = train_controller(
        records,
        path,
        config=ControllerTrainingConfig(
            hidden_size=32,
            action_embedding_size=4,
            dropout=0.0,
            epochs=15,
            validation_dataset="source-b",
            seed=4,
        ),
    )
    assert path.exists()
    assert 1 <= metrics.best_epoch <= 15
    assert 0 <= metrics.validation_ndcg <= 1
    assert "dataset" not in bundle.feature_names

    evidence = EvidenceBundle.from_mapping(
        "deployment", {"spectral_entropy": 0.9, "mean_abs_channel_correlation": 0.1}
    )
    restored = ControllerBundle.load(path)
    assert restored.predict(evidence).keys() == bundle.predict(evidence).keys()
    assert restored.select(evidence, budget=Budget(max_trainable_parameters=0)) == "A0"


def test_controller_rejects_unknown_or_query_derived_feature(tmp_path) -> None:
    records = [
        record("source", f"episode-{index}", action, float(index), {"support": index})
        for index in range(2)
        for action in ("A0", "A1")
    ]
    bundle, _ = train_controller(
        records,
        tmp_path / "controller.pt",
        config=ControllerTrainingConfig(hidden_size=8, epochs=2),
    )
    unknown = EvidenceBundle.from_mapping("target", {"support": 1.0, "query_loss": 99.0})
    with pytest.raises(ValueError, match="unknown evidence"):
        bundle.predict(unknown)


def test_oracle_gate_requires_heterogeneous_adapter_winners_and_regret() -> None:
    records = []
    winners = ("A2", "A2", "A3", "A3", "A4", "A4")
    for episode_index, winner in enumerate(winners):
        for action_number in range(7):
            action_id = f"A{action_number}"
            for seed, noise in enumerate((-0.01, 0.0, 0.01)):
                gain = (1.0 if action_id == winner else 0.0) + noise
                records.append(
                    record(
                        "source",
                        f"episode-{episode_index}",
                        action_id,
                        gain,
                        {"support": float(episode_index)},
                        seed=seed,
                    )
                )
    result = evaluate_oracle_gate(records, bootstrap_samples=5_000, seed=9)
    assert result.passed
    assert result.heterogeneous_adapter_winners
    assert result.positive_fixed_action_regret
    assert set(result.winning_families) == {"channel", "frequency", "lora"}
    assert result.bootstrap_ci_low > 0


def test_failed_oracle_gate_blocks_controller_command(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        require_oracle_gate(path)


def test_oracle_gate_is_bound_to_run_and_model(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "config_hash": "expected",
                "model_revision": "model-a",
            }
        ),
        encoding="utf-8",
    )
    require_oracle_gate(path, config_hash="expected", model_revision="model-a")
    with pytest.raises(RuntimeError, match="different run"):
        require_oracle_gate(path, config_hash="other", model_revision="model-a")
