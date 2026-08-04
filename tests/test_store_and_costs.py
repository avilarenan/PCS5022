from __future__ import annotations

import dataclasses

import pytest

from utility_peft.costs import CostTable, filter_feasible
from utility_peft.store import UtilityStore
from utility_peft.types import Budget, UtilityRecord


def make_record(
    action_id: str = "A1",
    *,
    episode_id: str = "episode-1",
    dataset: str = "dataset",
    horizon: int = 8,
    gain: float = 0.5,
    trainable: int = 10,
    total: int = 100,
    flops: float = 20.0,
    memory: float = 4.0,
    wall_time: float = 2.0,
    seed: int = 0,
) -> UtilityRecord:
    return UtilityRecord(
        episode_id=episode_id,
        dataset=dataset,
        dataset_family="test",
        horizon=horizon,
        action_id=action_id,
        seed=seed,
        frozen_loss=2.0,
        adapted_loss=1.0,
        normalized_gain=gain,
        trainable_parameters=trainable,
        stored_adapter_parameters=trainable,
        total_parameters=total,
        profiled_flops=flops,
        peak_memory_mb=memory,
        wall_time_s=wall_time,
        evidence={"spectral_entropy": 0.3},
        config_hash="config",
        model_revision="revision",
        preprocessing_hash="preprocessing",
    )


def test_utility_arithmetic_preserves_raw_costs() -> None:
    record = make_record()
    utility = record.utility({"parameters": 1.0, "time": 0.1})
    assert utility == pytest.approx(0.5 - 0.1 - 0.2)
    assert record.profiled_flops == 20.0
    assert record.peak_memory_mb == 4.0


def test_utility_record_evidence_is_immutable() -> None:
    record = make_record()
    with pytest.raises(TypeError):
        record.evidence["new"] = 1.0  # type: ignore[index]


def test_budget_filtering_uses_hard_raw_limits() -> None:
    frozen = make_record("A0", trainable=0, flops=0.0, memory=1.0, wall_time=0.0)
    head = make_record("A1", trainable=20, flops=10.0, memory=5.0, wall_time=2.0)
    costs = CostTable.from_records([frozen, head])
    budget = Budget(max_trainable_parameters=10, max_peak_memory_mb=2.0)
    assert filter_feasible(["A0", "A1"], costs, budget) == ["A0"]
    assert costs.utility("A0", 0.2, {"parameters": 1.0}) == pytest.approx(0.2)


def test_parquet_store_is_partitioned_resume_safe_and_deduplicated(tmp_path) -> None:
    store = UtilityStore(tmp_path)
    record = make_record(dataset="source/name", horizon=96)
    assert store.append(record)
    assert not store.append(record)
    path = store.path_for(record)
    assert path.exists()
    assert "dataset=source_name" in str(path)
    assert "horizon=96" in str(path)
    loaded = store.records()
    assert len(loaded) == 1
    assert loaded[0].key == record.key
    assert dict(loaded[0].evidence) == dict(record.evidence)
    other_run = dataclasses.replace(record, config_hash="other-config")
    assert store.append(other_run)
    assert store.records(config_hash="config") == [loaded[0]]
    assert store.records(config_hash="other-config") == [other_run]
    assert {item.key for item in store.records(seeds={0})} == {
        loaded[0].key,
        other_run.key,
    }
    assert store.records(seeds={1}) == []


def test_record_key_covers_all_required_partitions() -> None:
    first = make_record()
    assert first.key != make_record(dataset="other").key
    assert first.key != make_record(horizon=12).key
    assert first.key != make_record(seed=1).key
