from __future__ import annotations

import math

import pytest

from utility_peft.correlation_benchmark import (
    CorrelationRouterConfig,
    RouterActionIds,
    evaluate_correlation_lodo,
    infer_router_action_ids,
    paper_time_peft_parameter_savings,
    train_correlation_router,
)
from utility_peft.types import UtilityRecord


def _records(*, one_class: bool = False, aliases: bool = False) -> list[UtilityRecord]:
    action_ids = (
        RouterActionIds(base="A2", frequency="A3", channel="A4", full="A5")
        if aliases
        else RouterActionIds()
    )
    rows: list[UtilityRecord] = []
    patterns = ((-2.0, -2.0), (-1.0, 2.0), (2.0, -1.0), (3.0, 3.0))
    for dataset_index, dataset in enumerate(("d0", "d1", "d2")):
        for episode_index, (frequency_score, channel_score) in enumerate(patterns):
            if one_class:
                frequency_score = channel_score = 2.0
            frequency_benefit = 0.20 if frequency_score > 0 else -0.10
            channel_benefit = 0.15 if channel_score > 0 else -0.08
            losses = {
                action_ids.base: 1.0,
                action_ids.frequency: 1.0 - frequency_benefit,
                action_ids.channel: 1.0 - channel_benefit,
                action_ids.full: 1.0 - frequency_benefit - channel_benefit,
            }
            evidence = {
                "residual_mean_abs_autocorrelation": frequency_score,
                "residual_lag1_abs_autocorrelation": frequency_score,
                "residual_mean_abs_channel_correlation": channel_score,
                "residual_correlation_nonstationarity": channel_score,
                "frozen_mse": 1.0,
                "residual_std": 1.0,
                "channels": 3.0,
                "horizon": 8.0,
            }
            costs = {
                action_ids.base: (100, 0.40, 1.0, 100.0),
                action_ids.frequency: (200, 0.65, 2.0, 150.0),
                action_ids.channel: (300, 0.75, 3.0, 175.0),
                action_ids.full: (400, 1.00, 4.0, 250.0),
            }
            for action_id in action_ids.all:
                for seed in (0, 1):
                    parameters, wall_time, flops, memory = costs[action_id]
                    loss = losses[action_id] + seed * 0.001
                    rows.append(
                        UtilityRecord(
                            episode_id=f"{dataset}-e{episode_index}",
                            dataset=dataset,
                            dataset_family="test",
                            horizon=8,
                            action_id=action_id,
                            seed=seed,
                            frozen_loss=1.2,
                            adapted_loss=loss,
                            normalized_gain=(1.2 - loss) / 1.2,
                            trainable_parameters=parameters,
                            stored_adapter_parameters=parameters,
                            total_parameters=10_000,
                            profiled_flops=flops,
                            peak_memory_mb=memory,
                            wall_time_s=wall_time,
                            evidence=evidence,
                            config_hash="test",
                            model_revision=f"tiny-{dataset_index}",
                            preprocessing_hash="fixed",
                            frozen_mae=1.0,
                            adapted_mae=math.sqrt(loss),
                            evidence_wall_time_s=0.01,
                        )
                    )
    return rows


def _accuracy_records(
    losses_by_dataset: dict[str, dict[str, tuple[float, ...]]],
) -> list[UtilityRecord]:
    """Build complete arms whose source marginal labels always route to ``L``."""

    rows: list[UtilityRecord] = []
    evidence = {
        "residual_mean_abs_autocorrelation": 0.0,
        "residual_lag1_abs_autocorrelation": 0.0,
        "residual_mean_abs_channel_correlation": 0.0,
        "residual_correlation_nonstationarity": 0.0,
        "channels": 3.0,
        "horizon": 8.0,
    }
    for dataset, action_losses in losses_by_dataset.items():
        seeds = range(len(action_losses["L"]))
        assert set(action_losses) == {"L", "LF", "LC", "LFC"}
        assert all(len(values) == len(action_losses["L"]) for values in action_losses.values())
        for action_id, losses in action_losses.items():
            for seed, loss in zip(seeds, losses, strict=True):
                rows.append(
                    UtilityRecord(
                        episode_id=f"{dataset}-e0",
                        dataset=dataset,
                        dataset_family="test",
                        horizon=8,
                        action_id=action_id,
                        seed=seed,
                        frozen_loss=400.0,
                        adapted_loss=loss,
                        normalized_gain=0.0,
                        trainable_parameters={"L": 100, "LF": 200, "LC": 300, "LFC": 400}[
                            action_id
                        ],
                        stored_adapter_parameters={
                            "L": 100,
                            "LF": 200,
                            "LC": 300,
                            "LFC": 400,
                        }[action_id],
                        total_parameters=10_000,
                        profiled_flops={"L": 1.0, "LF": 2.0, "LC": 3.0, "LFC": 4.0}[
                            action_id
                        ],
                        peak_memory_mb={"L": 100.0, "LF": 150.0, "LC": 175.0, "LFC": 250.0}[
                            action_id
                        ],
                        wall_time_s={"L": 0.4, "LF": 0.65, "LC": 0.75, "LFC": 1.0}[
                            action_id
                        ],
                        evidence=evidence,
                        config_hash="accuracy-estimand",
                        model_revision="tiny",
                        preprocessing_hash="fixed",
                        frozen_mae=20.0,
                        adapted_mae=math.sqrt(loss),
                        evidence_wall_time_s=0.01,
                    )
                )
    return rows


def test_separate_logistic_gates_route_all_four_action_types() -> None:
    records = _records()
    router = train_correlation_router(records, heldout_dataset="d2")
    assert router.training_datasets == ("d0", "d1")

    expected = {
        (-2.0, -2.0): "L",
        (-1.0, 2.0): "LC",
        (2.0, -1.0): "LF",
        (3.0, 3.0): "LFC",
    }
    for (frequency, channel), action in expected.items():
        evidence = {
            "residual_mean_abs_autocorrelation": frequency,
            "residual_lag1_abs_autocorrelation": frequency,
            "residual_mean_abs_channel_correlation": channel,
            "residual_correlation_nonstationarity": channel,
        }
        assert router.select(evidence) == action


def test_one_class_fallback_is_deterministic() -> None:
    router = train_correlation_router(_records(one_class=True), heldout_dataset="d2")
    assert router.frequency_model.estimator is None
    assert router.channel_model.estimator is None
    assert router.frequency_model.constant_probability == 1.0
    assert router.channel_model.constant_probability == 1.0
    assert router.select({}) == "LFC"


def test_minimum_benefit_changes_binary_labels() -> None:
    router = train_correlation_router(
        _records(),
        heldout_dataset="d2",
        config=CorrelationRouterConfig(min_relative_benefit=0.5),
    )
    assert router.frequency_model.positive_examples == 0
    assert router.channel_model.positive_examples == 0
    assert router.select({}) == "L"


def test_router_config_rejects_negative_noninferiority_margin() -> None:
    with pytest.raises(ValueError, match="noninferiority_margin"):
        CorrelationRouterConfig(noninferiority_margin=-0.001)


def test_router_rejects_query_derived_features() -> None:
    router = train_correlation_router(_records(), heldout_dataset="d2")
    with pytest.raises(ValueError, match="query-derived"):
        router.select({"query_loss": 0.1})


def test_lodo_report_pairs_accuracy_and_end_to_end_costs() -> None:
    report = evaluate_correlation_lodo(_records())
    assert len(report.folds) == 3
    assert report.episodes == 12
    assert set(report.route_counts) == {"L", "LF", "LC", "LFC"}
    assert report.router.runs == report.baseline.runs == 24
    assert report.router.mse <= report.baseline.mse
    assert report.router.evidence_wall_time_s == pytest.approx(0.01)
    assert report.baseline.evidence_wall_time_s == 0.0
    assert report.comparison.end_to_end_time_reduction_fraction > 0
    assert report.comparison.trainable_parameter_reduction_fraction > 0
    assert report.comparison.noninferior_within_margin
    assert report.comparison.noninferiority_margin == pytest.approx(0.01)
    assert report.to_dict()["action_ids"]["full"] == "LFC"
    for fold in report.folds:
        assert fold.heldout_dataset not in fold.training_datasets
        assert fold.episodes == 4


def test_relative_mse_seed_averages_inside_each_unit_before_comparison() -> None:
    records = _accuracy_records(
        {
            "d0": {
                "L": (2.0, 90.0),
                "LF": (4.0, 110.0),
                "LC": (4.0, 110.0),
                "LFC": (1.0, 100.0),
            },
            "d1": {
                "L": (4.0, 180.0),
                "LF": (8.0, 220.0),
                "LC": (8.0, 220.0),
                "LFC": (2.0, 200.0),
            },
        }
    )

    report = evaluate_correlation_lodo(records)
    d0 = next(fold for fold in report.folds if fold.heldout_dataset == "d0")
    ratio_after_seed_mean = ((2.0 + 90.0) / 2 - (1.0 + 100.0) / 2) / (
        (1.0 + 100.0) / 2
    )
    mean_of_seed_ratios = ((2.0 - 1.0) / 1.0 + (90.0 - 100.0) / 100.0) / 2

    assert d0.route_counts == {"L": 1}
    assert d0.comparison.relative_mse_difference == pytest.approx(ratio_after_seed_mean)
    assert d0.comparison.relative_mse_difference != pytest.approx(mean_of_seed_ratios)


def test_unit_relative_mse_prevents_heterogeneous_scale_reversal() -> None:
    records = _accuracy_records(
        {
            "small": {
                "L": (2.0, 6.0),
                "LF": (4.0, 12.0),
                "LC": (4.0, 12.0),
                "LFC": (1.0, 3.0),
            },
            "large": {
                "L": (90.0, 270.0),
                "LF": (110.0, 330.0),
                "LC": (110.0, 330.0),
                "LFC": (100.0, 300.0),
            },
        }
    )

    report = evaluate_correlation_lodo(
        records,
        config=CorrelationRouterConfig(noninferiority_margin=0.50),
    )

    assert report.route_counts == {"L": 2}
    assert report.router.mse == pytest.approx(92.0)
    assert report.baseline.mse == pytest.approx(101.0)
    assert (report.router.mse - report.baseline.mse) / report.baseline.mse < 0
    assert report.comparison.relative_mse_difference == pytest.approx((1.0 - 0.1) / 2)
    assert report.comparison.relative_mse_difference > 0
    assert report.comparison.noninferiority_margin == pytest.approx(0.50)
    assert report.comparison.noninferior_within_margin
    assert report.noninferiority_margin == pytest.approx(0.50)


def test_mvp_action_aliases_are_inferred() -> None:
    records = _records(aliases=True)
    action_ids = infer_router_action_ids(records)
    assert action_ids == RouterActionIds(base="A2", frequency="A3", channel="A4", full="A5")
    report = evaluate_correlation_lodo(records)
    assert report.action_ids.full == "A5"


def test_paper_parameter_formula_and_expected_savings() -> None:
    result = paper_time_peft_parameter_savings(
        8,
        3,
        frequency_activation_rate=0.5,
        channel_activation_rate=0.25,
        fixed_trainable_parameters=10,
    )
    assert result.frequency_adapter_parameters == 64
    assert result.channel_adapter_parameters == 160
    assert result.always_on_trainable_parameters == 234
    assert result.expected_router_trainable_parameters == pytest.approx(82.0)
    assert result.expected_saved_parameters == pytest.approx(152.0)
    assert result.expected_reduction_fraction == pytest.approx(152 / 234)
    assert result.active_parameters_by_route == {
        "L": 10,
        "LF": 74,
        "LC": 170,
        "LFC": 234,
    }


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("frequency_activation_rate", -0.1), ("channel_activation_rate", 1.1)),
)
def test_parameter_formula_validates_activation_rates(keyword, value) -> None:
    arguments = {"frequency_activation_rate": 0.5, "channel_activation_rate": 0.5}
    arguments[keyword] = value
    with pytest.raises(ValueError, match="activation_rate"):
        paper_time_peft_parameter_savings(8, 2, **arguments)
