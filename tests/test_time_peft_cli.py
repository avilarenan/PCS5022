from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from click import BadParameter
from omegaconf import OmegaConf

import utility_peft.cli as cli_module
from utility_peft.cli import (
    _build_source_head_template,
    _build_template,
    _load_time_peft_reproduction_series,
    _require_time_peft_protocol_lock,
    _resolve_time_peft_test_role,
    _sha256_file,
    _synthetic_dataset_loader_options,
    _time_peft_protocol_lock_payload,
    _time_peft_reproduction_config,
    _time_peft_reproduction_hash,
    _tune_time_peft_with_optional_cache,
    _validate_time_peft_confirmatory_profile,
    _validate_time_peft_result_artifact,
    _validate_time_peft_tuning_artifact,
    _write_time_peft_reproduction_report,
)
from utility_peft.config import load_config


def test_cli_reproduction_config_keeps_smoke_caps_explicit() -> None:
    config = load_config("time_peft_reproduction_smoke")
    training = _time_peft_reproduction_config(config, 96, (0,))
    assert training.batch_size == 128
    assert training.gradient_clip is None
    assert training.precision == "fp32"
    assert training.smoke_caps is not None
    assert training.smoke_caps.active
    assert training.smoke_caps.train_windows == 512


def test_reproduction_hash_includes_common_lr_seed_set() -> None:
    one_seed = load_config("time_peft_reproduction", ["experiment.seeds=[0]"])
    three_seeds = load_config("time_peft_reproduction")
    assert _time_peft_reproduction_hash(one_seed) != _time_peft_reproduction_hash(three_seeds)

    manifest = [
        {
            "name": "Lorenz",
            "family": "synthetic",
            "sha256": "a" * 64,
            "channels": 3,
            "timestamps": 12000,
        }
    ]
    changed = [dict(manifest[0], sha256="b" * 64)]
    assert _time_peft_reproduction_hash(
        one_seed, dataset_manifest=manifest
    ) != _time_peft_reproduction_hash(one_seed, dataset_manifest=changed)


def test_reproduction_test_role_distinguishes_smoke_and_development() -> None:
    smoke = load_config("time_peft_reproduction_smoke")
    uncapped = load_config("time_peft_reproduction")

    assert _resolve_time_peft_test_role(smoke, "auto") == "plumbing-smoke"
    assert _resolve_time_peft_test_role(uncapped, "auto") == "development-parity"
    assert _resolve_time_peft_test_role(uncapped, "confirmatory") == "confirmatory"
    with pytest.raises(BadParameter, match="Capped reproduction"):
        _resolve_time_peft_test_role(smoke, "confirmatory")
    with pytest.raises(BadParameter, match="requires at least one"):
        _resolve_time_peft_test_role(uncapped, "plumbing-smoke")


def test_confirmatory_role_requires_the_full_preregistered_profile() -> None:
    config = load_config("time_peft_reproduction")
    datasets = tuple(str(value) for value in config.experiment.datasets)
    horizons = tuple(int(value) for value in config.experiment.horizons)
    seeds = tuple(int(value) for value in config.experiment.seeds)
    _validate_time_peft_confirmatory_profile(
        config,
        datasets=datasets,
        horizons=horizons,
        seeds=seeds,
        test_role="confirmatory",
    )

    with pytest.raises(BadParameter, match="preregistered full protocol.*datasets"):
        _validate_time_peft_confirmatory_profile(
            config,
            datasets=("ECGCA515",),
            horizons=horizons,
            seeds=seeds,
            test_role="confirmatory",
        )
    proxy = load_config(
        "time_peft_reproduction",
        ["experiment.preprocessing.synthetic_generator=compatible"],
    )
    with pytest.raises(BadParameter, match="synthetic_generator"):
        _validate_time_peft_confirmatory_profile(
            proxy,
            datasets=datasets,
            horizons=horizons,
            seeds=seeds,
            test_role="confirmatory",
        )


def test_protocol_lock_is_complete_atomic_and_immutable(tmp_path: Path) -> None:
    config = load_config("time_peft_reproduction", ["experiment.seeds=[0,2]"])
    dataset_manifest = [
        {
            "name": "ECGCA515",
            "family": "medical",
            "sha256": "a" * 64,
            "channels": 6,
            "timestamps": 820000,
        },
        {
            "name": "Lorenz",
            "family": "synthetic",
            "sha256": "b" * 64,
            "channels": 3,
            "timestamps": 12000,
        },
    ]
    payload = _time_peft_protocol_lock_payload(
        config,
        run_hash="run-hash",
        model_revision="model-revision",
        datasets=("ECGCA515", "Lorenz"),
        dataset_manifest=dataset_manifest,
        horizons=(96, 192),
        seeds=(0, 2),
        test_role="development-parity",
    )
    lock_path = tmp_path / "protocol-lock.json"

    _require_time_peft_protocol_lock(lock_path, payload, create=True)
    _require_time_peft_protocol_lock(lock_path, payload, create=False)

    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert persisted["resolved_config"]["experiment"]["seeds"] == [0, 2]
    assert persisted["implementation_hash"]
    assert persisted["model_revision"] == "model-revision"
    assert persisted["test_role"] == "development-parity"
    assert persisted["synthetic_generation"]["synthetic_generator"] == "dysts"
    assert persisted["synthetic_generation"]["synthetic_length"] == 12000
    assert persisted["synthetic_generation"]["runtime_provenance"]["dysts_version"] == "0.96"
    assert persisted["dataset_manifest"] == dataset_manifest
    assert len(persisted["dataset_horizon_seed_cartesian"]) == 8

    conflicting = dict(payload)
    conflicting["test_role"] = "confirmatory"
    with pytest.raises(BadParameter, match="protocol lock mismatch.*test_role"):
        _require_time_peft_protocol_lock(lock_path, conflicting, create=True)
    assert json.loads(lock_path.read_text(encoding="utf-8")) == payload


def test_test_and_report_require_existing_protocol_lock(tmp_path: Path) -> None:
    with pytest.raises(BadParameter, match="Run --stage tune first"):
        _require_time_peft_protocol_lock(
            tmp_path / "missing.json",
            {"schema_version": 1},
            create=False,
        )


def _write_valid_tuning_sidecar(
    tuning_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    tuning_path.write_bytes(b"tuning-checkpoint")
    metadata: dict[str, object] = {
        "schema_version": 2,
        "config_hash": "config-hash",
        "model_revision": "model-revision",
        "protocol_lock_sha256": "1" * 64,
        "test_role": "development-parity",
        "tuning_artifact_sha256": _sha256_file(tuning_path),
        "tuning": {
            "dataset": "ECGCA515",
            "dataset_sha256": "a" * 64,
            "config": {"horizon": 96},
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _validate_test_tuning_artifact(tuning_path: Path, metadata_path: Path) -> None:
    _validate_time_peft_tuning_artifact(
        tuning_path,
        metadata_path,
        config_hash="config-hash",
        model_revision="model-revision",
        protocol_lock_sha256="1" * 64,
        test_role="development-parity",
        dataset="ECGCA515",
        horizon=96,
        dataset_sha256="a" * 64,
    )


def test_tuning_sidecar_binds_protocol_revision_cell_and_artifact_bytes(
    tmp_path: Path,
) -> None:
    tuning_path = tmp_path / "tuning.pt"
    metadata_path = tmp_path / "tuning.json"
    metadata = _write_valid_tuning_sidecar(tuning_path, metadata_path)

    _validate_test_tuning_artifact(tuning_path, metadata_path)

    metadata["config_hash"] = "different-config"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BadParameter, match="config_hash"):
        _validate_test_tuning_artifact(tuning_path, metadata_path)

    metadata["config_hash"] = "config-hash"
    metadata["model_revision"] = "different-revision"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BadParameter, match="model_revision"):
        _validate_test_tuning_artifact(tuning_path, metadata_path)

    metadata["model_revision"] = "model-revision"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    tuning_path.write_bytes(b"tampered-checkpoint")
    with pytest.raises(BadParameter, match="SHA-256 mismatch"):
        _validate_test_tuning_artifact(tuning_path, metadata_path)


def test_incomplete_tuning_artifact_is_never_treated_as_resumable(tmp_path: Path) -> None:
    tuning_path = tmp_path / "tuning.pt"
    metadata_path = tmp_path / "tuning.json"
    tuning_path.write_bytes(b"orphan")
    with pytest.raises(BadParameter, match="Incomplete tune artifact"):
        _validate_test_tuning_artifact(tuning_path, metadata_path)


def test_result_artifact_is_bound_to_protocol_lock_and_test_role(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {
        "schema_version": 2,
        "config_hash": "config-hash",
        "model_revision": "model-revision",
        "protocol_lock_sha256": "1" * 64,
        "test_role": "development-parity",
        "result": {
            "dataset": "ECGCA515",
            "dataset_sha256": "a" * 64,
            "horizon": 96,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        _validate_time_peft_result_artifact(
            path,
            config_hash="config-hash",
            model_revision="model-revision",
            protocol_lock_sha256="1" * 64,
            test_role="development-parity",
            dataset="ECGCA515",
            horizon=96,
            dataset_sha256="a" * 64,
        )
        == payload
    )
    with pytest.raises(BadParameter, match="test_role"):
        _validate_time_peft_result_artifact(
            path,
            config_hash="config-hash",
            model_revision="model-revision",
            protocol_lock_sha256="1" * 64,
            test_role="confirmatory",
            dataset="ECGCA515",
            horizon=96,
            dataset_sha256="a" * 64,
        )
    with pytest.raises(BadParameter, match="different cell"):
        _validate_time_peft_result_artifact(
            path,
            config_hash="config-hash",
            model_revision="model-revision",
            protocol_lock_sha256="1" * 64,
            test_role="development-parity",
            dataset="ECGCA515",
            horizon=96,
            dataset_sha256="b" * 64,
        )


def test_cli_routes_trial_cache_when_core_supports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | None] = []
    sentinel = object()

    def tune_with_cache(
        template: object,
        series: object,
        config: object,
        *,
        trial_cache: Path | None = None,
    ) -> object:
        del template, series, config
        calls.append(trial_cache)
        return sentinel

    monkeypatch.setattr(cli_module, "tune_time_peft", tune_with_cache)
    cache_dir = tmp_path / "trials"
    result = _tune_time_peft_with_optional_cache(
        object(),
        object(),
        object(),  # type: ignore[arg-type]
        trial_cache_dir=cache_dir,
    )
    assert result is sentinel
    assert calls == [cache_dir]


def test_reproduction_loader_routes_pinned_dysts_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def load_series(name: str, root: object, **kwargs: object) -> object:
        captured.update({"name": name, "root": root, **kwargs})
        return sentinel

    monkeypatch.setattr(cli_module, "load_dataset_series", load_series)
    config = load_config("time_peft_reproduction")
    result = _load_time_peft_reproduction_series(
        config,
        "Lorenz",
        download=False,
    )
    assert result is sentinel
    assert captured["synthetic_generator"] == "dysts"
    assert captured["synthetic_length"] == 12000
    assert captured["synthetic_random_seed"] == 0
    assert captured["synthetic_pts_per_period"] == 100


def test_build_template_passes_explicit_head_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMomentBackbone(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            captured.update(kwargs)
            self.d_model = 8

    monkeypatch.setattr(cli_module, "MomentBackbone", FakeMomentBackbone)
    config = load_config("time_peft_reproduction")
    OmegaConf.update(config, "model.head_dropout", 0.25, force_add=True)
    _build_template(config, 96, channels=1)
    assert captured["head_dropout"] == 0.25

    captured.clear()
    _build_source_head_template(config, 96)
    assert captured["head_dropout"] == 0.25


def test_prepare_data_synthetic_loader_options_are_explicit_and_legacy_safe() -> None:
    reproduction = load_config("time_peft_budget24")
    assert _synthetic_dataset_loader_options(reproduction) == {
        "synthetic_generator": "dysts",
        "synthetic_length": 12_000,
        "synthetic_random_seed": 0,
        "synthetic_pts_per_period": 100,
    }
    assert _synthetic_dataset_loader_options(load_config("correlation_pilot")) == {}


def test_reproduction_report_keeps_paper_targets_out_of_selection(tmp_path) -> None:
    config = load_config("time_peft_reproduction_smoke")

    def record(method: str, mse: float, mae: float, learning_rate: float) -> dict[str, object]:
        return {
            "method_id": method,
            "seed": 0,
            "test_mse": mse,
            "test_mae": mae,
            "selected_learning_rate": learning_rate,
            "smoke_caps_active": True,
        }

    payloads = [
        {
            "config_hash": "test-hash",
            "result": {
                "dataset": "ECGCA515",
                "dataset_family": "medical",
                "dataset_sha256": "a" * 64,
                "horizon": 96,
                "records": [
                    record("L", 0.22, 0.25, 1e-3),
                    record("LFC", 0.18, 0.22, 1e-4),
                ],
            },
        }
    ]
    _write_time_peft_reproduction_report(tmp_path, payloads, config, "test-hash")

    report = json.loads((tmp_path / "time_peft_reproduction.json").read_text())
    assert report["smoke_caps_active"] is True
    assert report["time_peft_better_cells"] == 1
    assert report["cells"][0]["paper_l_mse"] == 0.199
    assert report["cells"][0]["paper_lfc_mse"] == 0.125
    assert report["cells"][0]["selected_l_lr"] == 1e-3
    assert report["cells"][0]["selected_lfc_lr"] == 1e-4
    assert report["assumptions"]["head_dropout"] == 0.1
    assert report["assumptions"]["synthetic_generation"]["runtime_provenance"][
        "dysts_version"
    ] == "0.96"
    assert report["dataset_provenance"][0]["sha256"] == "a" * 64
    markdown = (tmp_path / "time_peft_reproduction.md").read_text()
    assert "Smoke only" in markdown
    assert "does not run or compose the correlation router" in markdown
