from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from utility_peft.actions import ACTION_BY_ID
from utility_peft.evaluator import TrainingConfig, evaluate_action
from utility_peft.evidence import extract_evidence


def fast_training() -> TrainingConfig:
    return TrainingConfig(
        effective_batch_size=4,
        query_batch_size=4,
        bf16=False,
        profile_flops=False,
    )


def test_evidence_api_rejects_query_episode(template, episode) -> None:
    with pytest.raises(TypeError, match="SupportView only"):
        extract_evidence(episode, template)  # type: ignore[arg-type]


def test_query_tensors_cannot_change_evidence(template, episode) -> None:
    first = extract_evidence(episode.support, template, include_gradient_probe=False)
    changed_query = dataclasses.replace(
        episode,
        query_x=torch.randn_like(episode.query_x) * 1e6,
        query_y=torch.randn_like(episode.query_y) * 1e6,
    )
    second = extract_evidence(changed_query.support, template, include_gradient_probe=False)
    assert first == second


def test_one_backward_probe_restores_grad_flags(template, episode) -> None:
    before = {name: parameter.requires_grad for name, parameter in template.named_parameters()}
    evidence = extract_evidence(episode.support, template)
    after = {name: parameter.requires_grad for name, parameter in template.named_parameters()}
    assert before == after
    assert "gradient_query_l2" in evidence.names
    assert "gradient_value_l2" in evidence.names


def test_evaluator_runs_exact_updates_without_mutating_template(
    template, episode, monkeypatch
) -> None:
    evidence = extract_evidence(episode.support, template, include_gradient_probe=False)
    action = dataclasses.replace(ACTION_BY_ID["A1"], update_steps=3)
    original_step = torch.optim.AdamW.step
    steps = 0

    def counted_step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    initial = {name: value.clone() for name, value in template.state_dict().items()}
    record = evaluate_action(
        template,
        episode,
        action,
        evidence,
        seed=0,
        config=fast_training(),
        config_hash="config",
        model_revision="tiny",
        device="cpu",
    )
    assert record.status == "ok"
    assert record.trainable_parameters > 0
    assert record.wall_time_s > 0
    assert math.isfinite(record.normalized_gain)
    assert steps == 3
    for name, value in initial.items():
        assert torch.equal(value, template.state_dict()[name])


def test_a0_uses_no_training_and_preserves_loss(template, episode) -> None:
    evidence = extract_evidence(episode.support, template, include_gradient_probe=False)
    record = evaluate_action(
        template,
        episode,
        ACTION_BY_ID["A0"],
        evidence,
        seed=0,
        config=fast_training(),
        config_hash="config",
        model_revision="tiny",
        device="cpu",
    )
    assert record.trainable_parameters == 0
    assert record.frozen_loss == pytest.approx(record.adapted_loss)
    assert record.normalized_gain == pytest.approx(0.0)
    assert record.wall_time_s == 0.0


def test_second_nan_failure_creates_explicit_infeasible_record(template, episode) -> None:
    evidence = extract_evidence(episode.support, template, include_gradient_probe=False)
    invalid = dataclasses.replace(episode, query_y=torch.full_like(episode.query_y, torch.nan))
    record = evaluate_action(
        template,
        invalid,
        ACTION_BY_ID["A0"],
        evidence,
        seed=0,
        config=fast_training(),
        config_hash="config",
        model_revision="tiny",
        device="cpu",
    )
    assert record.status == "failed"
    assert record.error is not None
    assert "NaN or infinite" in record.error
