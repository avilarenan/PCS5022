from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch

from utility_peft.actions import ALL_ACTIONS
from utility_peft.backbones.moment import MomentBackbone
from utility_peft.data.datasets import load_dataset_series
from utility_peft.episodes import build_episode
from utility_peft.evaluator import TrainingConfig, evaluate_action
from utility_peft.evidence import extract_evidence
from utility_peft.model import AdaptableForecaster
from utility_peft.store import UtilityStore


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("UTILITY_PEFT_RUN_GPU_SMOKE") != "1",
    reason="set UTILITY_PEFT_RUN_GPU_SMOKE=1 to run the A40/MOMENT smoke test",
)
def test_every_action_on_one_etth1_episode_and_resume(tmp_path) -> None:
    assert torch.cuda.is_available()
    data_root = Path(os.environ.get("UTILITY_PEFT_DATA_ROOT", "data"))
    dataset = load_dataset_series("ETTh1", data_root)
    episode = build_episode(
        dataset.values,
        dataset="ETTh1",
        dataset_family="standard",
        lookback=96,
        horizon=96,
        support_size=2,
        query_size=2,
        start=0,
        seed=0,
    )
    backbone = MomentBackbone(
        lookback=96,
        horizon=96,
        allow_random_head=True,
    )
    template = AdaptableForecaster(backbone)
    evidence = extract_evidence(
        episode.support,
        template,
        device="cuda",
        include_gradient_probe=False,
    )
    config = TrainingConfig(
        effective_batch_size=1,
        query_batch_size=1,
        bf16=True,
        profile_flops=True,
    )
    store = UtilityStore(tmp_path / "utilities")
    for action in ALL_ACTIONS:
        smoke_action = dataclasses.replace(
            action, update_steps=0 if action.action_id == "A0" else 1
        )
        record = evaluate_action(
            template,
            episode,
            smoke_action,
            evidence,
            seed=0,
            config=config,
            config_hash="gpu-smoke",
            model_revision=backbone.revision + ":random-head",
            device="cuda",
        )
        assert record.status == "ok"
        assert record.total_parameters > 0
        assert record.trainable_parameters >= 0
        assert record.profiled_flops >= 0
        assert record.peak_memory_mb >= 0
        assert record.wall_time_s >= 0
        assert store.append(record)

    restarted = UtilityStore(tmp_path / "utilities")
    records = restarted.records()
    assert len(records) == len(ALL_ACTIONS)
    assert not restarted.append(records[0])
