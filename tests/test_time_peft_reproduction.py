from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import utility_peft.time_peft_reproduction as reproduction
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.data.datasets import DatasetSeries, DatasetSplit
from utility_peft.model import AdaptableForecaster
from utility_peft.time_peft_reproduction import (
    MethodRecord,
    SmokeCaps,
    SplitWindows,
    TimePEFTReproductionConfig,
    aggregate_method_records,
    build_split_windows,
    fit_train_standardizer,
    load_tuning_result,
    prepare_reproduction_windows,
    run_time_peft_reproduction,
    save_tuning_result,
    test_time_peft,
    tune_time_peft,
)


def _series(values: torch.Tensor | None = None) -> DatasetSeries:
    if values is None:
        time = torch.linspace(0, 8, 72)
        values = torch.stack((torch.sin(time), torch.cos(time * 0.7)))
    return DatasetSeries(
        name="paper-test",
        family="synthetic",
        values=values,
        splits=(
            DatasetSplit("train", 0, 36),
            DatasetSplit("validation", 36, 54),
            DatasetSplit("test", 54, 72),
        ),
        sha256="a" * 64,
    )


def _template() -> AdaptableForecaster:
    torch.manual_seed(11)
    return AdaptableForecaster(
        TinyBackbone(d_model=8, patch_len=4, depth=1, heads=2, max_horizon=8),
        channels=2,
        adapter_implementation="paper",
    )


def _smoke_config() -> TimePEFTReproductionConfig:
    return TimePEFTReproductionConfig(
        lookback=4,
        horizon=2,
        learning_rates=(1e-3, 1e-4),
        batch_size=4,
        max_epochs=3,
        early_stopping_patience=2,
        seeds=(3,),
        device="cpu",
        smoke_caps=SmokeCaps(
            train_windows=8,
            validation_windows=4,
            test_windows=4,
            batches_per_epoch=1,
            evaluation_batches=1,
            epochs=2,
        ),
    )


def test_split_targets_stay_inside_named_partition_and_context_can_cross_boundary() -> None:
    series = _series()
    scaler = fit_train_standardizer(series)
    train = build_split_windows(
        series,
        "train",
        lookback=4,
        horizon=3,
        standardizer=scaler,
    )
    validation = build_split_windows(
        series,
        "validation",
        lookback=4,
        horizon=3,
        standardizer=scaler,
    )
    test = build_split_windows(
        series,
        "test",
        lookback=4,
        horizon=3,
        standardizer=scaler,
    )

    assert int(train.context_starts.min()) == series.split("train").start
    assert int(train.target_starts.min()) == series.split("train").start + 4
    assert int(validation.target_starts.min()) == series.split("validation").start
    assert int(validation.context_starts.min()) == series.split("validation").start - 4
    assert int(test.target_starts.min()) == series.split("test").start
    assert int(test.context_starts.min()) == series.split("test").start - 4
    assert int(train.target_ends.max()) <= series.split("train").end
    assert int(validation.target_ends.max()) <= series.split("validation").end
    assert int(test.target_ends.max()) <= series.split("test").end


def test_train_scaler_uses_only_train_and_missing_context_gets_a_mask() -> None:
    values = torch.arange(144, dtype=torch.float32).reshape(2, 72)
    values[:, 36:] += 10_000
    values[0, 34] = torch.nan
    series = _series(values)
    scaler = fit_train_standardizer(series)
    expected = values[:, :36]
    torch.testing.assert_close(scaler.mean, torch.nanmean(expected, dim=1, keepdim=True))

    validation = build_split_windows(
        series,
        "validation",
        lookback=4,
        horizon=2,
        standardizer=scaler,
        max_windows=1,
    )
    x, y, mask = validation.batch([0], device="cpu")
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert not bool(mask[0, 2])


def test_nonfinite_target_is_rejected_before_training() -> None:
    values = _series().values.clone()
    values[1, 36] = torch.nan
    series = _series(values)
    with pytest.raises(ValueError, match="non-finite validation target.*timestamp 36"):
        prepare_reproduction_windows(series, lookback=4, horizon=2)


def test_config_preserves_paper_defaults_and_rejects_overlong_run() -> None:
    config = TimePEFTReproductionConfig()
    assert config.batch_size == 128
    assert config.max_epochs == 100
    assert config.learning_rates == (1e-3, 1e-4, 1e-5)
    with pytest.raises(ValueError, match="paper maximum of 100"):
        replace(config, max_epochs=101)
    with pytest.raises(ValueError, match="positive"):
        SmokeCaps(test_windows=0)


def test_staged_reproduction_never_touches_test_while_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_splits: list[str] = []
    original_batch = SplitWindows.batch

    def tracked_batch(self, indices, *, device):
        observed_splits.append(self.split_name)
        return original_batch(self, indices, device=device)

    monkeypatch.setattr(SplitWindows, "batch", tracked_batch)
    template = _template()
    series = _series()
    tuning = tune_time_peft(template, series, _smoke_config())
    assert "test" not in observed_splits
    json.dumps(tuning.metadata(), allow_nan=False)

    artifact = tmp_path / "tuning.pt"
    save_tuning_result(tuning, artifact)
    loaded = load_tuning_result(artifact)
    observed_splits.clear()
    result = test_time_peft(template, series, loaded)
    assert set(observed_splits) == {"test"}

    assert [record.method_id for record in result.records] == ["L", "LFC"]
    assert len(result.trial_records) == 4
    assert all(record.test_evaluations == 1 for record in result.records)
    assert all(record.smoke_caps_active for record in result.records)
    assert all(len(record.trials) == 2 for record in result.records)
    assert all(record.selected_learning_rate in {1e-3, 1e-4} for record in result.records)
    fingerprints = {
        trial.initialization_fingerprint for trial in result.trial_records
    }
    assert len(fingerprints) == 1
    assert {trial.batch_order_seed for trial in result.trial_records} == {3}
    assert all(torch.isfinite(torch.tensor(record.test_mse)) for record in result.records)
    assert all(trial.status == "ok" for trial in result.trial_records)
    assert all(trial.epoch_metrics for trial in result.trial_records)
    assert all(trial.elapsed_time_s > 0 for trial in result.trial_records)
    assert all(trial.peak_cuda_memory_mb == 0 for trial in result.trial_records)
    assert result.split_policy == "complex-70/10/20-override"
    assert result.split_ranges == (
        ("train", 0, 50),
        ("validation", 50, 58),
        ("test", 58, 72),
    )


def test_reproduction_is_deterministic_and_aggregate_is_structured() -> None:
    config = replace(_smoke_config(), learning_rates=(1e-3,))
    first = run_time_peft_reproduction(_template(), _series(), config)
    second = run_time_peft_reproduction(_template(), _series(), config)

    assert first.records == second.records
    aggregate = aggregate_method_records((*first.records, *second.records))
    assert [record.method_id for record in aggregate] == ["L", "LFC"]
    assert all(record.runs == 2 for record in aggregate)
    assert all(record.std_test_mse == 0.0 for record in aggregate)


def test_aggregate_rejects_nonfinite_metrics() -> None:
    result = run_time_peft_reproduction(
        _template(),
        _series(),
        replace(_smoke_config(), learning_rates=(1e-3,)),
    )
    invalid: MethodRecord = replace(result.records[0], test_mse=float("nan"))
    with pytest.raises(ValueError, match="non-finite metrics"):
        aggregate_method_records((invalid,))


def test_trial_cache_resumes_and_rejects_a_different_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "trial-cache"
    config = replace(_smoke_config(), learning_rates=(1e-3,))
    first = tune_time_peft(_template(), _series(), config, trial_cache=cache)
    entries = sorted(cache.glob("trial-*.pt"))
    assert len(entries) == 2
    assert not list(cache.glob("*.tmp"))
    assert all(trial.status == "ok" for trial in first.trial_records)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("a completed cached trial was retrained")

    monkeypatch.setattr(reproduction, "_fit_trial", unexpected_fit)
    resumed = tune_time_peft(_template(), _series(), config, trial_cache=cache)
    assert resumed.metadata() == first.metadata()

    changed = _template()
    with torch.no_grad():
        next(changed.parameters()).add_(1.0)
    with pytest.raises(AssertionError, match="cached trial was retrained"):
        tune_time_peft(changed, _series(), config, trial_cache=cache)
    with pytest.raises(AssertionError, match="cached trial was retrained"):
        tune_time_peft(
            _template(),
            _series(),
            replace(config, weight_decay=0.02),
            trial_cache=cache,
        )


def test_failed_trial_is_cached_and_lr_requires_success_for_every_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = replace(
        _smoke_config().smoke_caps,
        train_windows=2,
        validation_windows=2,
        test_windows=2,
        batches_per_epoch=1,
        evaluation_batches=1,
        epochs=1,
    )
    config = replace(
        _smoke_config(),
        seeds=(1, 2),
        max_epochs=1,
        early_stopping_patience=1,
        smoke_caps=smoke,
    )

    def factory(seed: int) -> AdaptableForecaster:
        torch.manual_seed(seed)
        return AdaptableForecaster(
            TinyBackbone(d_model=8, patch_len=4, depth=1, heads=2, max_horizon=8),
            channels=2,
            adapter_implementation="paper",
        )

    original_evaluate = reproduction._evaluate
    validation_calls = 0

    def fail_second_validation(*args, **kwargs):
        nonlocal validation_calls
        windows = args[1]
        if windows.split_name == "validation":
            validation_calls += 1
            if validation_calls == 2:
                raise FloatingPointError("simulated divergence")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(reproduction, "_evaluate", fail_second_validation)
    cache = tmp_path / "failure-cache"
    tuning = tune_time_peft(factory, _series(), config, trial_cache=cache)
    json.dumps(tuning.metadata(), allow_nan=False)
    failed = [trial for trial in tuning.trial_records if trial.status == "failed"]
    assert len(failed) == 1
    assert failed[0].method_id == "L"
    assert failed[0].seed == 2
    assert failed[0].learning_rate == 1e-3
    assert "simulated divergence" in (failed[0].error or "")
    l_tuning = next(method for method in tuning.methods if method.method_id == "L")
    assert l_tuning.selected_learning_rate == 1e-4
    assert {checkpoint.record.seed for checkpoint in l_tuning.checkpoints} == {1, 2}
    assert len(list(cache.glob("trial-*.pt"))) == 8

    result = test_time_peft(factory, _series(), tuning)
    assert len(result.records) == 4
    assert all(record.test_evaluations == 1 for record in result.records)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("cached success or failure was retrained")

    monkeypatch.setattr(reproduction, "_fit_trial", unexpected_fit)
    resumed = tune_time_peft(factory, _series(), config, trial_cache=cache)
    assert resumed.metadata() == tuning.metadata()
