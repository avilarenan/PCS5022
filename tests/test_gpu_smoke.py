from __future__ import annotations

import math
import os
from collections import Counter

import pytest
import torch

from utility_peft.actions import resolve_time_peft_actions
from utility_peft.backbones.moment import MOMENT_MODEL_REVISION, MomentBackbone
from utility_peft.correlation import (
    CHANNEL_ROUTING_FEATURES,
    FREQUENCY_ROUTING_FEATURES,
    extract_correlation_evidence,
)
from utility_peft.episodes import build_episode
from utility_peft.evaluator import TrainingConfig, evaluate_action
from utility_peft.model import AdaptableForecaster, model_for_action
from utility_peft.store import UtilityStore
from utility_peft.utils import seed_everything

PAPER_TRAINABLE_PARAMETERS = {
    "L": 1_327_200,
    "LF": 1_917_024,
    "LC": 8_110_176,
    "LFC": 8_700_000,
}


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("UTILITY_PEFT_RUN_GPU_SMOKE") != "1",
    reason="set UTILITY_PEFT_RUN_GPU_SMOKE=1 to run the paper-mode MOMENT CUDA smoke test",
)
def test_paper_time_peft_arms_on_real_moment_cuda_and_resume(
    tmp_path, monkeypatch
) -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    seed_everything(0)

    generator = torch.Generator().manual_seed(0)
    series = torch.randn(21, 640, generator=generator).mul_(0.05).cumsum(dim=-1)
    episode = build_episode(
        series,
        dataset="synthetic-21-channel",
        dataset_family="gpu-smoke",
        lookback=96,
        horizon=96,
        support_size=64,
        query_size=32,
        start=0,
        seed=0,
    )
    with pytest.warns(UserWarning, match="Only reconstruction head is pre-trained"):
        backbone = MomentBackbone(
            lookback=96,
            horizon=96,
            allow_random_head=True,
        )
    assert backbone.revision == MOMENT_MODEL_REVISION
    template = AdaptableForecaster(
        backbone,
        channels=21,
        adapter_implementation="paper",
        frequency_top_k=3,
        adapter_dropout=0.0,
    )
    actions = resolve_time_peft_actions(("L", "LF", "LC", "LFC"), update_steps=1)
    assert all(
        (action.rank, action.alpha, action.target_modules, action.update_steps)
        == (8, 32, ("q", "k", "v"), 1)
        for action in actions
    )

    injection_probe = model_for_action(template, actions[0])
    injected_leaves = Counter(
        name.rsplit(".", 1)[-1] for name in injection_probe.injected_modules
    )
    assert injected_leaves == {"q": 12, "k": 12, "v": 12}
    assert len(injection_probe.injected_modules) == 36
    del injection_probe

    evidence = extract_correlation_evidence(
        episode.support,
        template,
        device="cuda",
        max_lag=8,
    )
    evidence_values = evidence.as_dict()
    assert set(FREQUENCY_ROUTING_FEATURES) <= evidence_values.keys()
    assert set(CHANNEL_ROUTING_FEATURES) <= evidence_values.keys()
    assert all(math.isfinite(value) for value in evidence_values.values())

    optimizer_steps = 0
    bf16_autocast_calls = 0
    original_step = torch.optim.AdamW.step
    original_autocast = torch.autocast

    def counted_step(optimizer, *args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(optimizer, *args, **kwargs)

    def counted_autocast(*args, **kwargs):
        nonlocal bf16_autocast_calls
        if (
            kwargs.get("device_type") == "cuda"
            and kwargs.get("dtype") is torch.bfloat16
            and kwargs.get("enabled") is True
        ):
            bf16_autocast_calls += 1
        return original_autocast(*args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    monkeypatch.setattr(torch, "autocast", counted_autocast)
    config = TrainingConfig(
        effective_batch_size=32,
        query_batch_size=32,
        bf16=True,
        profile_flops=False,
    )
    store = UtilityStore(tmp_path / "utilities")
    records = []
    for action in actions:
        record = evaluate_action(
            template,
            episode,
            action,
            evidence,
            seed=0,
            config=config,
            config_hash="paper-gpu-smoke-v1",
            model_revision=f"{backbone.revision}:random-head",
            device="cuda",
        )
        assert record.status == "ok", record.error
        assert record.trainable_parameters == PAPER_TRAINABLE_PARAMETERS[action.action_id]
        assert 100_000_000 < record.total_parameters < 200_000_000
        assert 0.0 < record.peak_memory_mb < 12 * 1024
        assert record.wall_time_s > 0.0
        assert record.frozen_mae is not None
        assert record.adapted_mae is not None
        assert all(
            math.isfinite(value)
            for value in (
                record.frozen_loss,
                record.adapted_loss,
                record.frozen_mae,
                record.adapted_mae,
                record.normalized_gain,
                record.profiled_flops,
                record.peak_memory_mb,
                record.wall_time_s,
            )
        )
        assert store.append(record)
        records.append(record)
        torch.cuda.empty_cache()

    assert optimizer_steps == len(actions)
    assert bf16_autocast_calls == len(actions)
    assert [record.action_id for record in records] == ["L", "LF", "LC", "LFC"]

    restarted = UtilityStore(tmp_path / "utilities")
    assert len(restarted.records()) == len(actions)
    assert not restarted.append(records[0])
