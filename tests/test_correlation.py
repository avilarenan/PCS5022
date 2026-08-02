from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from utility_peft import cli
from utility_peft.correlation import (
    extract_correlation_evidence,
    residual_correlation_features,
)


def test_extractor_uses_exactly_one_frozen_forward_and_restores_model(
    template, episode, monkeypatch
) -> None:
    calls = 0
    original_predict = template.predict

    def counted_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert not template.training
        assert not torch.is_grad_enabled()
        return original_predict(*args, **kwargs)

    template.train()
    before_flags = tuple(parameter.requires_grad for parameter in template.parameters())
    monkeypatch.setattr(template, "predict", counted_predict)
    evidence = extract_correlation_evidence(episode.support, template, max_lag=3)

    assert calls == 1
    assert template.training
    assert tuple(parameter.requires_grad for parameter in template.parameters()) == before_flags
    assert evidence.episode_id == episode.support.manifest.episode_id
    assert all(math.isfinite(value) for value in evidence.values)


def test_extractor_rejects_query_episode_and_query_changes_do_not_matter(template, episode) -> None:
    with pytest.raises(TypeError, match="SupportView only"):
        extract_correlation_evidence(episode, template)  # type: ignore[arg-type]

    first = extract_correlation_evidence(episode.support, template)
    changed = dataclasses.replace(
        episode,
        query_x=torch.randn_like(episode.query_x) * 1e9,
        query_y=torch.randn_like(episode.query_y) * 1e9,
    )
    second = extract_correlation_evidence(changed.support, template)
    assert first == second


def test_timed_extractor_matches_evidence_and_restores_template(template, episode) -> None:
    template.train()
    first_parameter = next(template.parameters())
    first_parameter.requires_grad_(False)
    original_device = first_parameter.device
    original_support_devices = tuple(
        tensor.device for tensor in (episode.support.x, episode.support.y, episode.support.mask)
    )
    original_flags = tuple(parameter.requires_grad for parameter in template.parameters())
    expected = extract_correlation_evidence(episode.support, template, max_lag=3)

    evidence, elapsed = cli._extract_correlation_evidence_timed(
        episode.support,
        template,
        device="cpu",
        max_lag=3,
    )

    assert evidence == expected
    assert elapsed >= 0.0
    assert template.training
    assert next(template.parameters()).device == original_device
    assert tuple(
        tensor.device for tensor in (episode.support.x, episode.support.y, episode.support.mask)
    ) == original_support_devices
    assert tuple(parameter.requires_grad for parameter in template.parameters()) == original_flags


def test_timed_extractor_excludes_template_device_transfers(
    template, episode, monkeypatch
) -> None:
    clock = [0.0]
    synthetic_device = [torch.device("cpu")]
    transfers: list[tuple[torch.device, torch.device]] = []
    support_transfers = 0
    marker = object()

    def tracked_to(device, *args, **kwargs):
        destination = torch.device(device)
        if destination != synthetic_device[0]:
            transfers.append((synthetic_device[0], destination))
            clock[0] += 10.0
            synthetic_device[0] = destination
        return template

    def fake_extract(support, model, *, device, max_lag):
        assert support is not episode.support
        assert max_lag == 3
        original_device = synthetic_device[0]
        model.to(device)
        clock[0] += 2.0
        model.to(original_device)
        return marker

    def tracked_tensor_to(tensor, device, *args, **kwargs):
        nonlocal support_transfers
        support_transfers += 1
        clock[0] += 10.0
        return tensor

    monkeypatch.setattr(template, "to", tracked_to)
    monkeypatch.setattr(torch.Tensor, "to", tracked_tensor_to)
    monkeypatch.setattr(cli, "extract_correlation_evidence", fake_extract)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    evidence, elapsed = cli._extract_correlation_evidence_timed(
        episode.support,
        template,
        device="cuda",
        max_lag=3,
    )

    assert evidence is marker
    assert elapsed == pytest.approx(2.0)
    assert support_transfers == 3
    assert transfers == [
        (torch.device("cpu"), torch.device("cuda")),
        (torch.device("cuda"), torch.device("cpu")),
    ]


def test_windowed_signed_correlations_and_nonstationarity_are_detected() -> None:
    time = torch.arange(1, 9, dtype=torch.float32)
    residual = torch.stack(
        (
            torch.stack((time, time)),
            torch.stack((time, -time)),
        )
    )
    features = residual_correlation_features(residual, max_lag=3)

    assert features["residual_mean_abs_channel_correlation"] == pytest.approx(1.0)
    assert features["residual_mean_signed_channel_correlation"] == pytest.approx(0.0)
    assert features["residual_correlation_nonstationarity"] == pytest.approx(1.0)
    assert features["residual_max_lagged_cross_correlation"] == pytest.approx(1.0)
    assert features["residual_lag1_abs_autocorrelation"] == pytest.approx(1.0)
    assert features["residual_correlation_effective_rank"] >= 1.0


@pytest.mark.parametrize(
    "residual",
    (
        torch.ones(2, 1, 5),
        torch.full((2, 3, 5), float("nan")),
        torch.tensor(
            [
                [[1.0, float("nan"), 1.0], [2.0, 2.0, float("inf")]],
                [[float("-inf"), 1.0, 1.0], [2.0, float("nan"), 2.0]],
            ]
        ),
    ),
)
def test_constants_single_channel_and_nonfinite_values_are_safe(residual) -> None:
    features = residual_correlation_features(residual, max_lag=2)
    assert features
    assert all(math.isfinite(value) for value in features.values())
    assert 0.0 <= features["residual_correlation_effective_rank_fraction"] <= 1.0
    if residual.shape[1] == 1:
        assert features["residual_mean_abs_channel_correlation"] == 0.0
        assert features["residual_max_lagged_cross_correlation"] == 0.0


def test_one_step_residual_has_zero_lagged_features() -> None:
    features = residual_correlation_features(torch.randn(3, 2, 1))
    assert features["residual_lag1_abs_autocorrelation"] == 0.0
    assert features["residual_max_lagged_cross_correlation"] == 0.0


def test_residual_feature_shape_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        residual_correlation_features(torch.randn(2, 3))
    with pytest.raises(ValueError, match="positive"):
        residual_correlation_features(torch.randn(2, 3, 4), max_lag=0)
