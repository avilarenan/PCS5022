"""Command-line workflows for the forecasting MVP."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import typer
from omegaconf import DictConfig, OmegaConf

from utility_peft.actions import (
    REFERENCE_ACTION,
    resolve_actions,
    resolve_time_peft_actions,
)
from utility_peft.artifacts import ArtifactLayout
from utility_peft.backbones.moment import MomentBackbone
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.config import load_config, resolved_dict
from utility_peft.controller import (
    ControllerTrainingConfig,
    train_controller,
    train_controller_nested,
)
from utility_peft.correlation import extract_correlation_evidence
from utility_peft.correlation_benchmark import (
    CorrelationRouterConfig,
    evaluate_correlation_lodo,
    paper_time_peft_parameter_savings,
)
from utility_peft.data.datasets import (
    DatasetSeries,
    dysts_reproduction_provenance,
    load_dataset_series,
)
from utility_peft.episodes import (
    EpisodeRepository,
    build_episode,
    chronological_starts,
    chronological_starts_in_range,
)
from utility_peft.evaluator import TrainingConfig, evaluate_action
from utility_peft.evidence import extract_evidence
from utility_peft.lodo import evaluate_leave_one_dataset_out
from utility_peft.model import AdaptableForecaster
from utility_peft.oracle import (
    evaluate_oracle_gate,
    require_oracle_gate,
    write_oracle_gate,
)
from utility_peft.parity import (
    TimePeftParityManifest,
    baseline_label,
    write_parity_manifest,
)
from utility_peft.reporting import build_report as render_report
from utility_peft.source_head import (
    SourceHeadTrainingConfig,
    train_source_head,
    validate_source_head_provenance,
)
from utility_peft.store import UtilityStore
from utility_peft.time_peft_reproduction import (
    PAPER_METHOD_IDS,
    SmokeCaps,
    TimePEFTReproductionConfig,
    load_tuning_result,
    save_tuning_result,
    test_time_peft,
    tune_time_peft,
)
from utility_peft.types import SupportView
from utility_peft.utils import (
    atomic_write_json,
    environment_metadata,
    implementation_hash,
    seed_everything,
    stable_hash,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Counterfactual utility prediction for time-series PEFT actions.",
)


@app.command("train-source-head")
def train_source_head_command(
    config_name: str = typer.Option("pilot", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    download: bool = typer.Option(False, help="Download the pinned source dataset."),
) -> None:
    """Train leakage-auditable horizon-specific forecasting heads."""

    config = load_config(config_name, override)
    layout = _layout(config)
    os.environ.setdefault("HF_HOME", str(Path(config.paths.cache).resolve()))
    source_name = str(config.source_head.dataset)
    evaluation_datasets = tuple(str(name) for name in config.experiment.datasets)
    if source_name in evaluation_datasets:
        raise typer.BadParameter(
            f"Source-head dataset {source_name} must be excluded from evaluation datasets"
        )
    dataset = load_dataset_series(source_name, config.paths.data, download=download)
    training_values = OmegaConf.to_container(config.source_head.training, resolve=True)
    if not isinstance(training_values, dict):
        raise TypeError("source_head.training must be a mapping")
    training = SourceHeadTrainingConfig(**training_values)
    pattern = str(config.source_head.output_pattern)
    for horizon_value in config.experiment.horizons:
        horizon = int(horizon_value)
        seed_everything(training.seed)
        template = _build_source_head_template(config, horizon)
        checkpoint = layout.checkpoints / pattern.format(horizon=horizon)
        metrics = train_source_head(
            template,
            dataset,
            horizon=horizon,
            lookback=int(config.experiment.lookback),
            checkpoint_path=checkpoint,
            evaluation_datasets=evaluation_datasets,
            config=training,
            device=str(config.device),
        )
        typer.echo(
            f"h={horizon}: best source validation MSE={metrics.best_validation_mse:.6g} "
            f"at update {metrics.best_update}"
        )
    _record_run(
        layout,
        "train-source-head",
        config,
        extra={"source_dataset_hash": dataset.sha256},
    )


@app.command("reproduce-time-peft")
def reproduce_time_peft(
    config_name: str = typer.Option("pilot", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    protocol: str = typer.Option("matched", help="matched or paper"),
) -> None:
    """Run the legacy MVP parity subset with an explicit claim guard.

    This command preserves the historical A2--A5 experiment.  The current
    paper-specified Time-PEFT comparator is the LFC arm generated by
    ``run-correlation-benchmark``.
    """

    if protocol not in {"matched", "paper"}:
        raise typer.BadParameter("protocol must be 'matched' or 'paper'")
    config = load_config(config_name, override)
    verified = bool(config.parity.verified)
    if protocol == "paper" and not verified:
        raise typer.BadParameter(
            "Paper-protocol reproduction is blocked until official parity is verified"
        )
    if protocol == "paper":
        raise typer.BadParameter(
            "No official Time-PEFT runner is configured; matched-budget results remain style-only"
        )

    layout = _layout(config)
    repository = EpisodeRepository(layout.episodes)
    configured_datasets = tuple(str(value) for value in config.parity.datasets)
    configured_horizons = tuple(int(value) for value in config.parity.horizons)
    configured_seeds = _configured_seed_tuple(config.parity.seeds, field="parity.seeds")
    manifests = [
        manifest
        for manifest in repository.manifests()
        if manifest.dataset in configured_datasets and manifest.horizon in configured_horizons
    ]
    if not manifests:
        raise typer.BadParameter("No parity episodes found; run prepare-data with the pilot config")
    selected_manifests = _middle_manifests(manifests)
    store = UtilityStore(layout.root / "parity" / "utilities")
    run_hash = _experiment_hash(config)
    revision = _model_revision(config)
    existing = {record.key for record in store.records()}
    actions = resolve_actions(["A2", "A3", "A4", "A5"])
    templates: dict[int, AdaptableForecaster] = {}
    generated = 0
    for manifest in selected_manifests:
        episode = repository.load(manifest.episode_id)
        if manifest.horizon not in templates:
            templates[manifest.horizon] = _build_template(config, manifest.horizon)
        template = templates[manifest.horizon]
        evidence, evidence_time = _extract_evidence_timed(
            episode.support, template, device=str(config.device)
        )
        for action in actions:
            matched_action = replace(action, update_steps=100)
            for seed in configured_seeds:
                key = (
                    manifest.dataset,
                    manifest.horizon,
                    manifest.episode_id,
                    matched_action.action_id,
                    seed,
                    run_hash,
                    revision,
                    manifest.preprocessing_hash,
                )
                if key in existing:
                    continue
                record = evaluate_action(
                    template,
                    episode,
                    matched_action,
                    evidence,
                    seed=seed,
                    config=_training_config(config),
                    config_hash=run_hash,
                    model_revision=revision,
                    device=str(config.device),
                    evidence_wall_time_s=evidence_time,
                )
                store.append(record)
                existing.add(record.key)
                generated += 1
    label = baseline_label(
        verified=verified,
        configured_label=str(config.parity.implementation_label),
    )
    parity_manifest = TimePeftParityManifest(
        paper_id=str(config.parity.paper_id),
        protocol=protocol,
        implementation_label=label,
        verified=verified,
        verification_note=str(config.parity.verification_note),
        model_revision=revision,
        datasets=configured_datasets,
        horizons=configured_horizons,
        actions=tuple(action.action_id for action in actions),
        seeds=configured_seeds,
        generated_records=generated,
    )
    write_parity_manifest(layout.reports / "time_peft_parity.json", parity_manifest)
    _record_run(layout, "reproduce-time-peft", config, extra={"protocol": protocol})
    typer.echo(f"Generated {generated} matched records against {label}")


@app.command("run-time-peft-reproduction")
def run_time_peft_reproduction_command(
    config_name: str = typer.Option("time_peft_reproduction", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    stage: str = typer.Option(
        "all",
        help="tune, test, report, or all. Test requires a completed tune artifact.",
    ),
    download: bool = typer.Option(False, help="Download pinned datasets that are missing."),
    test_role: str = typer.Option(
        "auto",
        "--test-role",
        help=(
            "auto, plumbing-smoke, development-parity, or confirmatory. The resolved role "
            "is immutable for this run hash."
        ),
    ),
) -> None:
    """Reproduce the paper's MOMENT-base Time-PEFT versus LoRA comparison."""

    if stage not in {"tune", "test", "report", "all"}:
        raise typer.BadParameter("stage must be tune, test, report, or all")
    config = load_config(config_name, override)
    configured_actions = tuple(str(value) for value in config.experiment.actions)
    if configured_actions != PAPER_METHOD_IDS:
        raise typer.BadParameter(
            f"Paper reproduction requires actions {PAPER_METHOD_IDS}, got {configured_actions}"
        )
    if str(config.model.adapter_implementation) not in {
        "paper",
        "paper_count_inferred",
    }:
        raise typer.BadParameter(
            "Paper reproduction requires model.adapter_implementation=paper or "
            "paper_count_inferred"
        )
    if config.model.get("source_head_checkpoint") is not None:
        raise typer.BadParameter(
            "Paper reproduction must initialize a fresh target forecasting head; "
            "model.source_head_checkpoint must be null"
        )
    if not bool(config.model.allow_random_head):
        raise typer.BadParameter(
            "Paper reproduction requires model.allow_random_head=true for the fresh target head"
        )

    layout = _layout(config)
    os.environ.setdefault("HF_HOME", str(Path(config.paths.cache).resolve()))
    datasets = tuple(str(value) for value in config.experiment.datasets)
    horizons = tuple(int(value) for value in config.experiment.horizons)
    seeds = _configured_seed_tuple(config.experiment.seeds)
    if not datasets or not horizons:
        raise typer.BadParameter("experiment.datasets and experiment.horizons cannot be empty")
    revision = _model_revision(config) + f":{config.model.adapter_implementation}"
    root = layout.root / "paper-reproduction"
    resolved_test_role = _resolve_time_peft_test_role(config, test_role)
    _validate_time_peft_confirmatory_profile(
        config,
        datasets=datasets,
        horizons=horizons,
        seeds=seeds,
        test_role=resolved_test_role,
    )
    loaded_series = tuple(
        _load_time_peft_reproduction_series(config, dataset, download=download)
        for dataset in datasets
    )
    dataset_manifest = _time_peft_dataset_manifest(loaded_series)
    run_hash = _time_peft_reproduction_hash(
        config,
        dataset_manifest=dataset_manifest,
    )
    protocol_lock = _time_peft_protocol_lock_payload(
        config,
        run_hash=run_hash,
        model_revision=revision,
        datasets=datasets,
        dataset_manifest=dataset_manifest,
        horizons=horizons,
        seeds=seeds,
        test_role=resolved_test_role,
    )
    protocol_lock_path = _time_peft_protocol_lock_path(root, run_hash)
    _require_time_peft_protocol_lock(
        protocol_lock_path,
        protocol_lock,
        create=stage in {"tune", "all"},
    )
    protocol_lock_sha256 = _sha256_file(protocol_lock_path)

    if stage in {"tune", "all"}:
        for series in loaded_series:
            for horizon in horizons:
                training = _time_peft_reproduction_config(config, horizon, seeds)
                tuning_path = _time_peft_tuning_path(root, series.name, horizon, run_hash)
                metadata_path = tuning_path.with_suffix(".json")
                if tuning_path.exists() or metadata_path.exists():
                    _validate_time_peft_tuning_artifact(
                        tuning_path,
                        metadata_path,
                        config_hash=run_hash,
                        model_revision=revision,
                        protocol_lock_sha256=protocol_lock_sha256,
                        test_role=resolved_test_role,
                        dataset=series.name,
                        horizon=horizon,
                        dataset_sha256=series.sha256,
                    )
                    typer.echo(f"Tune exists: {series.name}/h{horizon}")
                    continue
                factory = _time_peft_template_factory(
                    config,
                    horizon=horizon,
                    channels=int(series.values.shape[0]),
                )
                trial_cache_dir = _time_peft_trial_cache_dir(
                    root,
                    series.name,
                    horizon,
                    run_hash,
                )
                tuning = _tune_time_peft_with_optional_cache(
                    factory,
                    series,
                    training,
                    trial_cache_dir=trial_cache_dir,
                )
                temporary = tuning_path.with_suffix(".tmp")
                save_tuning_result(tuning, temporary)
                temporary.replace(tuning_path)
                atomic_write_json(
                    metadata_path,
                    {
                        "schema_version": 2,
                        "config_hash": run_hash,
                        "model_revision": revision,
                        "protocol_lock_sha256": protocol_lock_sha256,
                        "tuning_artifact_sha256": _sha256_file(tuning_path),
                        "adapter_implementation": str(config.model.adapter_implementation),
                        "implementation_label": str(config.claim.implementation_label),
                        "official_code_verified": bool(config.claim.official_code_verified),
                        "test_role": resolved_test_role,
                        "tuning": tuning.metadata(),
                    },
                )
                typer.echo(f"Tuned {series.name}/h{horizon}: {tuning_path}")
                del tuning
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if stage in {"test", "all"}:
        for series in loaded_series:
            for horizon in horizons:
                result_path = _time_peft_result_path(root, series.name, horizon, run_hash)
                if result_path.exists():
                    _validate_time_peft_result_artifact(
                        result_path,
                        config_hash=run_hash,
                        model_revision=revision,
                        protocol_lock_sha256=protocol_lock_sha256,
                        test_role=resolved_test_role,
                        dataset=series.name,
                        horizon=horizon,
                        dataset_sha256=series.sha256,
                    )
                    typer.echo(f"Test exists: {series.name}/h{horizon}")
                    continue
                tuning_path = _time_peft_tuning_path(root, series.name, horizon, run_hash)
                metadata_path = tuning_path.with_suffix(".json")
                _validate_time_peft_tuning_artifact(
                    tuning_path,
                    metadata_path,
                    config_hash=run_hash,
                    model_revision=revision,
                    protocol_lock_sha256=protocol_lock_sha256,
                    test_role=resolved_test_role,
                    dataset=series.name,
                    horizon=horizon,
                    dataset_sha256=series.sha256,
                )
                tuning = load_tuning_result(tuning_path)
                expected_training = _time_peft_reproduction_config(config, horizon, seeds)
                if tuning.config != expected_training:
                    raise typer.BadParameter(
                        f"Tune artifact for {series.name}/h{horizon} has a different protocol"
                    )
                factory = _time_peft_template_factory(
                    config,
                    horizon=horizon,
                    channels=int(series.values.shape[0]),
                )
                result = test_time_peft(factory, series, tuning, device=str(config.device))
                atomic_write_json(
                    result_path,
                    {
                        "schema_version": 2,
                        "config_hash": run_hash,
                        "model_revision": revision,
                        "protocol_lock_sha256": protocol_lock_sha256,
                        "adapter_implementation": str(config.model.adapter_implementation),
                        "implementation_label": str(config.claim.implementation_label),
                        "official_code_verified": bool(config.claim.official_code_verified),
                        "test_role": resolved_test_role,
                        "result": asdict(result),
                    },
                )
                typer.echo(f"Tested {series.name}/h{horizon}: {result_path}")
                del result, tuning
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if stage in {"report", "all"}:
        result_payloads = []
        missing = []
        for series in loaded_series:
            for horizon in horizons:
                result_path = _time_peft_result_path(root, series.name, horizon, run_hash)
                if not result_path.exists():
                    missing.append(f"{series.name}/h{horizon}")
                    continue
                payload = _validate_time_peft_result_artifact(
                    result_path,
                    config_hash=run_hash,
                    model_revision=revision,
                    protocol_lock_sha256=protocol_lock_sha256,
                    test_role=resolved_test_role,
                    dataset=series.name,
                    horizon=horizon,
                    dataset_sha256=series.sha256,
                )
                result_payloads.append(payload)
        if missing:
            raise typer.BadParameter(
                "Cannot report an incomplete Cartesian result; missing: " + ", ".join(missing)
            )
        report_dir = root / "reports"
        _write_time_peft_reproduction_report(report_dir, result_payloads, config, run_hash)
        typer.echo(f"Report: {report_dir / 'time_peft_reproduction.md'}")

    _record_run(
        layout,
        "run-time-peft-reproduction",
        config,
        extra={
            "workflow": "time-peft-paper-v1",
            "stage": stage,
            "reproduction_seeds": seeds,
            "test_role": resolved_test_role,
        },
    )


@app.command("run-correlation-benchmark")
def run_correlation_benchmark(
    config_name: str = typer.Option("correlation_pilot", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    analyze_only: bool = typer.Option(
        False,
        help="Reuse a complete utility store and rebuild only the LODO report.",
    ),
) -> None:
    """Compare residual-correlation routing with always-on L+F+C Time-PEFT."""

    config = load_config(config_name, override)
    adapter_implementation = str(
        config.model.get("adapter_implementation", "mvp")
    )
    if adapter_implementation not in {"paper", "paper_count_inferred"}:
        raise typer.BadParameter(
            "The correlation benchmark requires model.adapter_implementation=paper or "
            "paper_count_inferred"
        )
    layout = _layout(config)
    os.environ.setdefault("HF_HOME", str(Path(config.paths.cache).resolve()))
    repository = EpisodeRepository(layout.episodes)
    datasets = {str(name) for name in config.experiment.datasets}
    horizons = {int(value) for value in config.experiment.horizons}
    configured_seeds = _configured_seed_tuple(config.experiment.seeds)
    manifests = [
        manifest
        for manifest in repository.manifests()
        if manifest.dataset in datasets
        and manifest.horizon in horizons
        and manifest.partition == str(config.experiment.episode_partition)
        and manifest.lookback == int(config.experiment.lookback)
        and manifest.support_size == int(config.experiment.support_size)
        and manifest.query_size == int(config.experiment.query_size)
    ]
    episodes_per_cell = int(config.experiment.episodes_per_dataset_horizon)
    try:
        _validate_episode_cartesian_coverage(
            manifests,
            datasets=datasets,
            horizons=horizons,
            episodes_per_cell=episodes_per_cell,
        )
    except ValueError as error:
        raise typer.BadParameter(
            f"{error}; "
            f"run `utility-peft prepare-data --config {config_name} --download` first"
        ) from error

    correlation_root = layout.root / "correlation"
    store = UtilityStore(correlation_root / "utilities")
    run_hash = _experiment_hash(config, extra={"workflow": "residual-correlation-v1"})
    revision = _model_revision(config) + f":{adapter_implementation}"
    actions = resolve_time_peft_actions(
        tuple(str(value) for value in config.experiment.actions),
        update_steps=int(config.experiment.update_steps),
    )
    generated = 0
    templates: dict[tuple[int, int], AdaptableForecaster] = {}
    existing = {record.key for record in store.records()}
    training = _training_config(config)
    if not analyze_only:
        for manifest in manifests:
            episode = repository.load(manifest.episode_id)
            channels = int(episode.support.x.shape[1])
            template_key = (manifest.horizon, channels)
            if template_key not in templates:
                seed_everything(0)
                templates[template_key] = _build_template(
                    config,
                    manifest.horizon,
                    channels=channels,
                )
            template = templates[template_key]
            pending: list[tuple[Any, int]] = []
            for action in actions:
                for seed in configured_seeds:
                    key = (
                        manifest.dataset,
                        manifest.horizon,
                        manifest.episode_id,
                        action.action_id,
                        seed,
                        run_hash,
                        revision,
                        manifest.preprocessing_hash,
                    )
                    if key not in existing:
                        pending.append((action, seed))
            if not pending:
                continue
            evidence, evidence_time = _extract_correlation_evidence_timed(
                episode.support,
                template,
                device=str(config.device),
                max_lag=int(config.correlation.max_lag),
            )
            for action, seed in pending:
                record = evaluate_action(
                    template,
                    episode,
                    action,
                    evidence,
                    seed=seed,
                    config=training,
                    config_hash=run_hash,
                    model_revision=revision,
                    device=str(config.device),
                    evidence_wall_time_s=evidence_time,
                )
                store.append(record)
                existing.add(record.key)
                generated += 1

    episode_ids = {manifest.episode_id for manifest in manifests}
    configured_seed_set = set(configured_seeds)
    records = store.records(
        episode_ids=episode_ids,
        action_ids={action.action_id for action in actions},
        seeds=configured_seed_set,
        statuses={"ok"},
        config_hash=run_hash,
        model_revision=revision,
    )
    expected_records = len(manifests) * len(actions) * len(configured_seeds)
    if len(records) != expected_records:
        raise typer.BadParameter(
            f"Expected {expected_records} successful matched records, found {len(records)}. "
            "Resume the benchmark without --analyze-only; failed actions remain explicit."
        )
    router_config = _correlation_router_config(config)
    report = evaluate_correlation_lodo(
        records,
        config=router_config,
        expected_seeds=configured_seeds,
    )
    report_dir = correlation_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_correlation_report(report_dir, report, records, config)
    _record_run(
        layout,
        "run-correlation-benchmark",
        config,
        extra={"workflow": "residual-correlation-v1"},
    )
    typer.echo(
        f"Generated {generated} records; mean per-unit relative MSE "
        f"{report.comparison.relative_mse_difference:+.2%} "
        f"(95% CI {report.comparison.relative_mse_ci_low:+.2%}, "
        f"{report.comparison.relative_mse_ci_high:+.2%}); end-to-end time reduction "
        f"{report.comparison.end_to_end_time_reduction_fraction:+.2%}. "
        f"Report: {report_dir / 'correlation_benchmark.md'}"
    )


def _correlation_router_config(config: DictConfig) -> CorrelationRouterConfig:
    """Resolve every optional router/statistics knob at one auditable boundary."""

    router_defaults = CorrelationRouterConfig()
    return CorrelationRouterConfig(
        probability_threshold=float(config.correlation.probability_threshold),
        min_relative_benefit=float(config.correlation.minimum_relative_benefit),
        noninferiority_margin=float(config.correlation.get("noninferiority_margin", 0.01)),
        random_state=int(config.correlation.get("random_state", 17)),
        bootstrap_samples=int(config.correlation.get("bootstrap_samples", 10_000)),
        bootstrap_seed=int(
            config.correlation.get(
                "bootstrap_seed",
                config.correlation.get("seed", config.correlation.get("random_state", 17)),
            )
        ),
        bootstrap_confidence_level=float(
            config.correlation.get("bootstrap_confidence_level", 0.95)
        ),
        random_control_repeats=int(
            config.correlation.get("random_control_repeats", 1_000)
        ),
        regularization_c=float(
            config.correlation.get("regularization_c", router_defaults.regularization_c)
        ),
        max_iter=int(config.correlation.get("max_iter", router_defaults.max_iter)),
        frequency_features=tuple(
            str(value)
            for value in config.correlation.get(
                "frequency_features",
                router_defaults.frequency_features,
            )
        ),
        channel_features=tuple(
            str(value)
            for value in config.correlation.get(
                "channel_features",
                router_defaults.channel_features,
            )
        ),
    )


@app.command("prepare-data")
def prepare_data(
    config_name: str = typer.Option("config", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    download: bool = typer.Option(False, help="Download pinned public CSV sources."),
    allow_missing: bool = typer.Option(
        False, help="Skip manually acquired datasets that are not present."
    ),
) -> None:
    """Prepare chronological manifests and local support/query tensors."""

    config = load_config(config_name, override)
    layout = _layout(config)
    repository = EpisodeRepository(layout.episodes)
    prepared = 0
    partition_name = str(config.experiment.get("episode_partition", "test"))
    for dataset_name in config.experiment.datasets:
        try:
            loader_options = _synthetic_dataset_loader_options(config)
            dataset = load_dataset_series(
                str(dataset_name),
                config.paths.data,
                download=download,
                **loader_options,
            )
        except FileNotFoundError as error:
            if allow_missing:
                typer.echo(f"Skipping {dataset_name}: {error}")
                continue
            raise
        for horizon_value in config.experiment.horizons:
            horizon = int(horizon_value)
            partition = dataset.split(partition_name)
            starts = chronological_starts_in_range(
                partition.start,
                partition.end,
                lookback=int(config.experiment.lookback),
                horizon=horizon,
                support_size=int(config.experiment.support_size),
                query_size=int(config.experiment.query_size),
                episodes=int(config.experiment.episodes_per_dataset_horizon),
            )
            for episode_index, start in enumerate(starts):
                episode = build_episode(
                    dataset.values,
                    dataset=dataset.name,
                    dataset_family=dataset.family,
                    lookback=int(config.experiment.lookback),
                    horizon=horizon,
                    support_size=int(config.experiment.support_size),
                    query_size=int(config.experiment.query_size),
                    start=start,
                    seed=episode_index,
                    source_hash=dataset.sha256,
                    partition=partition.name,
                )
                repository.save(episode)
                prepared += 1
    _record_run(layout, "prepare-data", config)
    typer.echo(f"Prepared {prepared} episodes under {layout.episodes}")


@app.command("reproduce")
def reproduce(
    updates: int = typer.Option(2, min=1, help="CPU smoke-test updates per trainable action."),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
) -> None:
    """Run a deterministic synthetic CPU path through every subsystem."""

    config = load_config("reproduce", override)
    layout = _layout(config)
    repository = EpisodeRepository(layout.episodes)
    dataset = load_dataset_series("Lorenz", config.paths.data, lorenz_length=2_000)
    starts = chronological_starts(
        dataset.values.shape[1],
        lookback=int(config.experiment.lookback),
        horizon=int(config.experiment.horizons[0]),
        support_size=int(config.experiment.support_size),
        query_size=int(config.experiment.query_size),
        episodes=int(config.experiment.episodes_per_dataset_horizon),
    )
    episodes = []
    for index, start in enumerate(starts):
        episode = build_episode(
            dataset.values,
            dataset=dataset.name,
            dataset_family=dataset.family,
            lookback=int(config.experiment.lookback),
            horizon=int(config.experiment.horizons[0]),
            support_size=int(config.experiment.support_size),
            query_size=int(config.experiment.query_size),
            start=start,
            seed=index,
            source_hash=dataset.sha256,
        )
        repository.save(episode)
        episodes.append(episode)

    seed_everything(0)
    template = _build_template(config, int(config.experiment.horizons[0]))
    training = _training_config(config)
    store = UtilityStore(layout.utilities)
    run_hash = _experiment_hash(config, extra={"smoke_updates": updates})
    configured_seeds = _configured_seed_tuple(config.experiment.seeds)
    actions = [
        replace(action, update_steps=0 if action.action_id == "A0" else updates)
        for action in resolve_actions(list(config.experiment.actions))
    ]
    existing = {record.key for record in store.records()}
    revision = _model_revision(config)
    for episode in episodes:
        evidence, evidence_time = _extract_evidence_timed(episode.support, template, device="cpu")
        for action in actions:
            for seed in configured_seeds:
                key = (
                    episode.support.manifest.dataset,
                    episode.support.manifest.horizon,
                    episode.support.manifest.episode_id,
                    action.action_id,
                    seed,
                    run_hash,
                    revision,
                    episode.support.manifest.preprocessing_hash,
                )
                if key in existing:
                    continue
                record = evaluate_action(
                    template,
                    episode,
                    action,
                    evidence,
                    seed=seed,
                    config=training,
                    config_hash=run_hash,
                    model_revision=revision,
                    device="cpu",
                    evidence_wall_time_s=evidence_time,
                )
                store.append(record)
                existing.add(record.key)
    configured_seed_set = set(configured_seeds)
    records = store.records(
        seeds=configured_seed_set,
        config_hash=run_hash,
        model_revision=revision,
    )
    gate = evaluate_oracle_gate(
        records,
        bootstrap_samples=1_000,
        config_hash=run_hash,
        model_revision=revision,
        expected_seeds=configured_seeds,
    )
    write_oracle_gate(layout.oracle_gate, gate)
    controller_config = _controller_config(config)
    train_controller(
        records,
        layout.checkpoints / "synthetic-controller.pt",
        config=controller_config,
    )
    render_report(
        store,
        layout.reports,
        oracle_gate_path=layout.oracle_gate,
        seeds=configured_seed_set,
        config_hash=run_hash,
        model_revision=revision,
    )
    _record_run(layout, "reproduce", config, extra={"smoke_updates": updates})
    typer.echo(f"Synthetic reproduction completed under {layout.root}")


@app.command("generate-utilities")
def generate_utilities(
    config_name: str = typer.Option("config", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
) -> None:
    """Evaluate configured actions with exact resume-safe utility records."""

    config = load_config(config_name, override)
    layout = _layout(config)
    os.environ.setdefault("HF_HOME", str(Path(config.paths.cache).resolve()))
    repository = EpisodeRepository(layout.episodes)
    configured_datasets = {str(name) for name in config.experiment.datasets}
    configured_horizons = {int(value) for value in config.experiment.horizons}
    configured_partition = str(config.experiment.get("episode_partition", "test"))
    configured_seeds = _configured_seed_tuple(config.experiment.seeds)
    manifests = [
        manifest
        for manifest in repository.manifests()
        if manifest.dataset in configured_datasets
        and manifest.horizon in configured_horizons
        and manifest.partition == configured_partition
        and manifest.lookback == int(config.experiment.lookback)
        and manifest.support_size == int(config.experiment.support_size)
        and manifest.query_size == int(config.experiment.query_size)
    ]
    if not manifests:
        raise typer.BadParameter("No episode manifests found; run prepare-data first")
    try:
        _validate_episode_cartesian_coverage(
            manifests,
            datasets=configured_datasets,
            horizons=configured_horizons,
            episodes_per_cell=int(config.experiment.episodes_per_dataset_horizon),
        )
    except ValueError as error:
        raise typer.BadParameter(
            f"{error}; use a clean artifact root or regenerate the pilot manifests"
        ) from error
    store = UtilityStore(layout.utilities)
    run_hash = _experiment_hash(config)
    revision = _model_revision(config)
    existing = {record.key for record in store.records()}
    actions = resolve_actions(list(config.experiment.actions))
    if bool(config.experiment.include_reference):
        actions.append(REFERENCE_ACTION)
    reference_episodes = _reference_episode_ids(manifests)
    active_episode_ids = {manifest.episode_id for manifest in manifests}
    templates: dict[int, AdaptableForecaster] = {}
    generated = 0
    training = _training_config(config)
    for manifest in manifests:
        episode = repository.load(manifest.episode_id)
        horizon = manifest.horizon
        if horizon not in templates:
            seed_everything(0)
            templates[horizon] = _build_template(config, horizon)
        template = templates[horizon]
        pending = []
        for action in actions:
            if action.action_id == "A7" and manifest.episode_id not in reference_episodes:
                continue
            seed_values = (
                [configured_seeds[0]]
                if action.action_id == "A7"
                else configured_seeds
            )
            for seed_value in seed_values:
                seed = int(seed_value)
                key = (
                    manifest.dataset,
                    manifest.horizon,
                    manifest.episode_id,
                    action.action_id,
                    seed,
                    run_hash,
                    revision,
                    manifest.preprocessing_hash,
                )
                if key in existing:
                    continue
                pending.append((action, seed))
        if not pending:
            continue
        evidence, evidence_time = _extract_evidence_timed(
            episode.support, template, device=str(config.device)
        )
        for action, seed in pending:
            record = evaluate_action(
                template,
                episode,
                action,
                evidence,
                seed=seed,
                config=training,
                config_hash=run_hash,
                model_revision=revision,
                device=str(config.device),
                evidence_wall_time_s=evidence_time,
            )
            store.append(record)
            existing.add(record.key)
            generated += 1
    try:
        expected_rows = {
            (manifest.episode_id, action.action_id, int(seed_value))
            for manifest in manifests
            for action in actions
            if action.action_id != "A7" or manifest.episode_id in reference_episodes
            for seed_value in (
                [configured_seeds[0]]
                if action.action_id == "A7"
                else configured_seeds
            )
        }
        current_records = [
            record
            for record in store.records(
                episode_ids=active_episode_ids,
                action_ids={action.action_id for action in actions},
                seeds=set(configured_seeds),
                config_hash=run_hash,
                model_revision=revision,
            )
            if (record.episode_id, record.action_id, record.seed) in expected_rows
        ]
        if len(current_records) != len(expected_rows):
            raise RuntimeError(
                f"Expected {len(expected_rows)} utility records after generation, found "
                f"{len(current_records)}"
            )
        gate = evaluate_oracle_gate(
            current_records,
            config_hash=run_hash,
            model_revision=revision,
            expected_seeds=configured_seeds,
        )
        write_oracle_gate(layout.oracle_gate, gate)
        typer.echo(
            f"Oracle screen passed: {gate.screen_passed}; "
            f"confirmation ready: {gate.confirmation_ready}; final gate passed: {gate.passed}"
        )
    except ValueError as error:
        typer.echo(f"Oracle gate not evaluated: {error}")
    _record_run(layout, "generate-utilities", config)
    typer.echo(f"Generated {generated} new utility records under {layout.utilities}")


@app.command("train-controller")
def train_controller_command(
    config_name: str = typer.Option("config", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    exclude_dataset: str | None = typer.Option(
        None, help="Exclude one target dataset from controller fitting."
    ),
    feature_set: str = typer.Option(
        "full", help="complexity, structure, structure_mismatch, or full"
    ),
) -> None:
    """Train the controller only after the persisted oracle gate passes."""

    config = load_config(config_name, override)
    layout = _layout(config)
    run_hash = _experiment_hash(config)
    revision = _model_revision(config)
    require_oracle_gate(layout.oracle_gate, config_hash=run_hash, model_revision=revision)
    episode_ids = _oracle_episode_ids(layout.oracle_gate, run_hash, revision)
    configured_seeds = set(_configured_seed_tuple(config.experiment.seeds))
    records = UtilityStore(layout.utilities).records(
        episode_ids=episode_ids,
        statuses={"ok"},
        action_ids={f"A{index}" for index in range(7)},
        seeds=configured_seeds,
        config_hash=run_hash,
        model_revision=revision,
    )
    if exclude_dataset:
        records = [record for record in records if record.dataset != exclude_dataset]
    stem = f"controller-without-{exclude_dataset}" if exclude_dataset else "controller"
    output_name = f"{stem}.pt" if feature_set == "full" else f"{stem}-{feature_set}.pt"
    _, metrics = train_controller_nested(
        records,
        layout.checkpoints / output_name,
        config=_controller_config(config),
        feature_set=feature_set,
    )
    _record_run(layout, "train-controller", config)
    typer.echo(
        f"Controller selected epoch {metrics.best_epoch}; "
        f"source-held-out NDCG={metrics.validation_ndcg:.6f}"
    )


@app.command("evaluate-heldout")
def evaluate_heldout(
    config_name: str = typer.Option("config", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
) -> None:
    """Run leave-one-dataset-out routing over the completed utility table."""

    config = load_config(config_name, override)
    layout = _layout(config)
    run_hash = _experiment_hash(config)
    revision = _model_revision(config)
    require_oracle_gate(layout.oracle_gate, config_hash=run_hash, model_revision=revision)
    episode_ids = _oracle_episode_ids(layout.oracle_gate, run_hash, revision)
    configured_seeds = _configured_seed_tuple(config.experiment.seeds)
    configured_seed_set = set(configured_seeds)
    records = UtilityStore(layout.utilities).records(
        episode_ids=episode_ids,
        statuses={"ok"},
        action_ids={f"A{index}" for index in range(7)},
        seeds=configured_seed_set,
        config_hash=run_hash,
        model_revision=revision,
    )
    result = evaluate_leave_one_dataset_out(
        records,
        layout.root / "lodo",
        config=_controller_config(config),
        cost_weights=OmegaConf.to_container(config.cost_weights, resolve=True),
        expected_seeds=configured_seeds,
    )
    _record_run(layout, "evaluate-heldout", config)
    typer.echo(
        f"LODO NDCG={result.mean_controller_ndcg:.6f}; "
        f"oracle regret={result.mean_controller_oracle_regret:.6f}"
    )


@app.command("build-report")
def build_report_command(
    config_name: str = typer.Option("config", "--config"),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
    output: Path = typer.Option(Path("reports"), help="Versionable report directory."),
) -> None:
    """Build versionable tables and Markdown from local artifacts."""

    config = load_config(config_name, override)
    layout = _layout(config)
    run_hash = _experiment_hash(config)
    revision = _model_revision(config)
    episode_ids = _oracle_episode_ids(layout.oracle_gate, run_hash, revision)
    configured_seeds = set(_configured_seed_tuple(config.experiment.seeds))
    report = render_report(
        UtilityStore(layout.utilities),
        output,
        oracle_gate_path=layout.oracle_gate,
        lodo_metrics_path=layout.root / "lodo" / "lodo_metrics.json",
        parity_manifest_path=layout.reports / "time_peft_parity.json",
        episode_ids=episode_ids,
        seeds=configured_seeds,
        config_hash=run_hash,
        model_revision=revision,
    )
    typer.echo(str(report))


def _layout(config: DictConfig) -> ArtifactLayout:
    layout = ArtifactLayout.from_root(str(config.paths.artifacts))
    layout.create()
    return layout


def _build_template(
    config: DictConfig,
    horizon: int,
    *,
    channels: int | None = None,
    force_mvp_adapters: bool = False,
) -> AdaptableForecaster:
    if str(config.model.kind) == "tiny":
        backbone = TinyBackbone(
            d_model=int(config.model.d_model),
            patch_len=int(config.model.patch_len),
            depth=int(config.model.depth),
            heads=int(config.model.heads),
            max_horizon=max(horizon, 336),
        )
    elif str(config.model.kind) == "moment":
        checkpoint = _source_head_checkpoint(config, horizon)
        backbone = MomentBackbone(
            lookback=int(config.experiment.lookback),
            horizon=horizon,
            model_id=str(config.model.model_id),
            revision=str(config.model.revision),
            source_head_checkpoint=checkpoint,
            allow_random_head=bool(config.model.allow_random_head),
            head_dropout=float(config.model.get("head_dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unknown model kind: {config.model.kind}")
    implementation = (
        "mvp" if force_mvp_adapters else str(config.model.get("adapter_implementation", "mvp"))
    )
    return AdaptableForecaster(
        backbone,
        channels=channels,
        adapter_implementation=implementation,
        frequency_top_k=int(config.model.get("frequency_top_k", 3)),
        adapter_dropout=float(config.model.get("adapter_dropout", 0.0)),
    )


def _build_source_head_template(config: DictConfig, horizon: int) -> AdaptableForecaster:
    if str(config.model.kind) == "tiny":
        return _build_template(config, horizon, force_mvp_adapters=True)
    if str(config.model.kind) != "moment":
        raise ValueError(f"Unknown model kind: {config.model.kind}")
    backbone = MomentBackbone(
        lookback=int(config.experiment.lookback),
        horizon=horizon,
        model_id=str(config.model.model_id),
        revision=str(config.model.revision),
        source_head_checkpoint=None,
        allow_random_head=True,
        head_dropout=float(config.model.get("head_dropout", 0.0)),
    )
    return AdaptableForecaster(backbone)


def _source_head_checkpoint(config: DictConfig, horizon: int) -> str | None:
    value = config.model.get("source_head_checkpoint")
    if value is None:
        return None
    if isinstance(value, DictConfig):
        selected = value.get(str(horizon))
        checkpoint = str(selected) if selected else None
    else:
        checkpoint = str(value).format(horizon=horizon)
    source_config = config.get("source_head")
    if checkpoint and source_config and bool(source_config.get("require_provenance", False)):
        validate_source_head_provenance(
            checkpoint,
            horizon=horizon,
            evaluation_datasets=tuple(str(name) for name in config.experiment.datasets),
        )
    return checkpoint


def _training_config(config: DictConfig) -> TrainingConfig:
    values = OmegaConf.to_container(config.experiment.training, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("experiment.training must be a mapping")
    return TrainingConfig(**values)


def _synthetic_dataset_loader_options(config: DictConfig) -> dict[str, object]:
    """Forward an explicitly configured synthetic protocol without changing legacy defaults."""

    preprocessing = config.experiment.get("preprocessing")
    if preprocessing is None:
        return {}
    return {
        "synthetic_generator": str(
            preprocessing.get("synthetic_generator", "compatible")
        ),
        "synthetic_length": int(preprocessing.get("synthetic_length", 12_000)),
        "synthetic_random_seed": int(
            preprocessing.get("synthetic_random_seed", 0)
        ),
        "synthetic_pts_per_period": int(
            preprocessing.get("synthetic_pts_per_period", 100)
        ),
    }


def _extract_evidence_timed(
    support: Any,
    template: AdaptableForecaster,
    *,
    device: str,
) -> tuple[Any, float]:
    target = torch.device(device)
    with _inputs_placed_for_evidence(support, template, target) as placed_support:
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        evidence = extract_evidence(placed_support, template, device=target)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - started
    return evidence, elapsed


def _extract_correlation_evidence_timed(
    support: Any,
    template: AdaptableForecaster,
    *,
    device: str,
    max_lag: int,
) -> tuple[Any, float]:
    target = torch.device(device)
    with _inputs_placed_for_evidence(support, template, target) as placed_support:
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        evidence = extract_correlation_evidence(
            placed_support,
            template,
            device=target,
            max_lag=max_lag,
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - started
    return evidence, elapsed


@contextmanager
def _inputs_placed_for_evidence(
    support: Any,
    template: AdaptableForecaster,
    target: torch.device,
) -> Iterator[Any]:
    """Place immutable support tensors and a template outside the timed region."""

    first_parameter = next(template.parameters(), None)
    original_device = first_parameter.device if first_parameter is not None else target
    was_training = template.training
    original_flags = tuple(parameter.requires_grad for parameter in template.parameters())
    try:
        template.to(target)
        placed_support = (
            replace(
                support,
                x=support.x.to(target),
                y=support.y.to(target),
                mask=support.mask.to(target),
            )
            if type(support) is SupportView
            else support
        )
        yield placed_support
    finally:
        try:
            template.to(original_device)
        finally:
            template.train(was_training)
            for parameter, requires_grad in zip(
                template.parameters(), original_flags, strict=True
            ):
                parameter.requires_grad_(requires_grad)


def _write_correlation_report(
    output: Path,
    report: Any,
    records: list[Any],
    config: DictConfig,
) -> None:
    """Write machine-readable and human-readable matched benchmark reports."""

    route_counts = dict(report.route_counts)
    source_fixed_control = report.controls.source_fixed
    random_control = report.controls.random_histogram_matched
    oracle_control = report.controls.oracle
    hidden_size = int(config.model.get("d_model", 768))
    analytical: dict[str, object] = {}
    folds_by_dataset = {fold.heldout_dataset: fold for fold in report.folds}
    datasets = sorted({record.dataset for record in records})
    for dataset in datasets:
        dataset_records = [record for record in records if record.dataset == dataset]
        base_records = [record for record in dataset_records if record.action_id == "L"]
        fold = folds_by_dataset.get(dataset)
        if not base_records or fold is None:
            continue
        dataset_routes = dict(fold.route_counts)
        routed_episodes = max(int(fold.episodes), 1)
        frequency_rate = (
            dataset_routes.get("LF", 0) + dataset_routes.get("LFC", 0)
        ) / routed_episodes
        channel_rate = (
            dataset_routes.get("LC", 0) + dataset_routes.get("LFC", 0)
        ) / routed_episodes
        optional_rate = 1.0 - dataset_routes.get("L", 0) / routed_episodes
        channels = int(round(base_records[0].evidence.get("channels", 1.0)))
        fixed = int(
            round(sum(row.trainable_parameters for row in base_records) / len(base_records))
        )
        savings = paper_time_peft_parameter_savings(
            hidden_size,
            channels,
            frequency_activation_rate=frequency_rate,
            channel_activation_rate=channel_rate,
            fixed_trainable_parameters=fixed,
            count_inferred=(
                str(config.model.adapter_implementation) == "paper_count_inferred"
            ),
            optional_activation_rate=optional_rate,
        )
        analytical[dataset] = asdict(savings)

    payload = report.to_dict()
    implementation_label = str(config.claim.implementation_label)
    official_code_verified = bool(config.claim.official_code_verified)
    payload["implementation_label"] = implementation_label
    payload["official_code_verified"] = official_code_verified
    payload["baseline_identity"] = {
        "method": implementation_label,
        "action_id": report.action_ids.full,
        "always_on": True,
        "trainable_modules": [
            "forecast_head",
            "qkv_lora",
            "frequency_adapter",
            "channel_adapter",
        ],
        "official_code_verified": official_code_verified,
    }
    payload["evaluation_protocol"] = "episodic-fixed-update"
    payload["model_assumptions"] = {
        "adapter_implementation": str(config.model.adapter_implementation),
        "adapter_dropout": float(config.model.get("adapter_dropout", 0.0)),
        "head_dropout": float(config.model.get("head_dropout", 0.0)),
    }
    payload["analytical_parameter_savings"] = analytical
    atomic_write_json(output / "correlation_benchmark.json", payload)

    unit_fields = [
        "dataset",
        "horizon",
        "episode_id",
        "seeds",
        "frequency_probability",
        "channel_probability",
        "routed_action",
        "source_fixed_action",
        "oracle_action",
        *[f"{action_id}_seed_mean_mse" for action_id in report.action_ids.all],
        "router_seed_mean_mse",
        "lfc_seed_mean_mse",
        "relative_mse_difference_vs_lfc",
    ]
    with (output / "correlation_units.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=unit_fields)
        writer.writeheader()
        for unit in report.unit_table:
            writer.writerow(
                {
                    "dataset": unit.dataset,
                    "horizon": unit.horizon,
                    "episode_id": unit.episode_id,
                    "seeds": ",".join(str(seed) for seed in unit.seeds),
                    "frequency_probability": unit.frequency_probability,
                    "channel_probability": unit.channel_probability,
                    "routed_action": unit.routed_action,
                    "source_fixed_action": unit.source_fixed_action,
                    "oracle_action": unit.oracle_action,
                    **{
                        f"{action_id}_seed_mean_mse": unit.arm_seed_mean_mse[action_id]
                        for action_id in report.action_ids.all
                    },
                    "router_seed_mean_mse": unit.router_seed_mean_mse,
                    "lfc_seed_mean_mse": unit.lfc_seed_mean_mse,
                    "relative_mse_difference_vs_lfc": (
                        unit.relative_mse_difference_vs_lfc
                    ),
                }
            )

    fold_lines = []
    for fold in report.folds:
        fold_lines.append(
            "| "
            + " | ".join(
                (
                    fold.heldout_dataset,
                    str(fold.episodes),
                    f"{fold.router.mse:.6g}",
                    f"{fold.baseline.mse:.6g}",
                    f"{fold.comparison.relative_mse_difference:+.2%}",
                    f"[{fold.comparison.relative_mse_ci_low:+.2%}, "
                    f"{fold.comparison.relative_mse_ci_high:+.2%}]",
                    f"{fold.comparison.end_to_end_time_reduction_fraction:+.2%}",
                    f"{fold.comparison.trainable_parameter_reduction_fraction:+.2%}",
                    ", ".join(f"{key}:{value}" for key, value in fold.route_counts.items()),
                )
            )
            + " |"
        )
    unit_lines = [
        "| "
        + " | ".join(
            (
                unit.dataset,
                str(unit.horizon),
                unit.episode_id,
                f"{unit.frequency_probability:.3f}",
                f"{unit.channel_probability:.3f}",
                unit.routed_action,
                unit.source_fixed_action,
                unit.oracle_action,
                f"{unit.arm_seed_mean_mse[report.action_ids.base]:.6g}",
                f"{unit.arm_seed_mean_mse[report.action_ids.frequency]:.6g}",
                f"{unit.arm_seed_mean_mse[report.action_ids.channel]:.6g}",
                f"{unit.arm_seed_mean_mse[report.action_ids.full]:.6g}",
                f"{unit.relative_mse_difference_vs_lfc:+.2%}",
            )
        )
        + " |"
        for unit in report.unit_table
    ]
    markdown = "\n".join(
        (
            "# Residual-correlation router versus Time-PEFT",
            "",
            f"Baseline: **{implementation_label}** (`{report.action_ids.full}`).",
            "",
            "Here, *always-on* describes Time-PEFT itself: its forecast head, Q/K/V "
            "LoRA, frequency adapter, and channel adapter are trained together. It is "
            "not a separate generic baseline.",
            "",
            "> This is a paper-specified reimplementation, not an official Time-PEFT "
            "reproduction. The public paper omits code and several run-level hyperparameters.",
            "",
            "## Overall matched result",
            "",
            "| Routed MSE | Paper-specified Time-PEFT (`LFC`) MSE | Mean per-unit relative MSE | "
            "Paired 95% CI | "
            "End-to-end time reduction | "
            "Trainable-parameter reduction |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {report.router.mse:.6g} | {report.baseline.mse:.6g} | "
            f"{report.comparison.relative_mse_difference:+.2%} | "
            f"[{report.comparison.relative_mse_ci_low:+.2%}, "
            f"{report.comparison.relative_mse_ci_high:+.2%}] | "
            f"{report.comparison.end_to_end_time_reduction_fraction:+.2%} | "
            f"{report.comparison.trainable_parameter_reduction_fraction:+.2%} |",
            "",
            f"Routes across {report.episodes} held-out episodes: "
            + ", ".join(f"{key}={value}" for key, value in route_counts.items())
            + ".",
            "",
            "The routed time includes one frozen support forecast, correlation extraction, "
            "routing, and adaptation. The always-on baseline includes adaptation only. "
            "Offline exhaustive sweeps used to fit and evaluate the LODO router are research "
            "cost and are retained separately in the utility store.",
            "",
            "Routed and baseline MSE are descriptive means over matched seed runs. The "
            "relative-MSE estimand first averages seeds within each "
            "dataset/horizon/episode, computes a paired relative difference for that unit, "
            "and then gives every unit equal weight.",
            "",
            f"The paired {report.comparison.bootstrap_confidence_level:.0%} interval resamples "
            "datasets and then units within datasets after averaging seeds. Its upper bound "
            f"is {'below' if report.comparison.noninferior_within_margin else 'not below'} the "
            f"{report.noninferiority_margin:.2%} noninferiority margin; superiority is "
            f"{'supported' if report.comparison.accuracy_superior else 'not supported'} by "
            "this interval.",
            "",
            "## Source-fixed, random, and oracle controls",
            "",
            "The source-fixed arm is chosen from `L`, `LF`, `LC`, and `LFC` separately "
            "in every outer fold using source "
            "datasets only. The random control globally permutes held-out assignments while "
            "exactly preserving the router's overall route histogram. The oracle uses "
            "held-out query losses "
            "and is therefore a non-deployable accuracy ceiling.",
            "",
            f"- Source-fixed relative MSE versus `LFC`: "
            f"{source_fixed_control.relative_mse_difference_vs_lfc:+.2%}; router "
            f"versus source-fixed: "
            f"{source_fixed_control.router_relative_mse_difference_vs_control:+.2%}. "
            f"Paired 95% CI "
            f"[{source_fixed_control.router_relative_mse_difference_vs_control_ci_low:+.2%}, "
            f"{source_fixed_control.router_relative_mse_difference_vs_control_ci_high:+.2%}]. "
            "Fold choices: "
            + ", ".join(
                f"{dataset}={action}"
                for dataset, action in source_fixed_control.actions_by_heldout_dataset.items()
            )
            + ".",
            f"- Globally histogram-matched random ({random_control.repeats} "
            "repetitions; descriptive only) mean relative MSE versus `LFC`: "
            f"{random_control.relative_mse_difference_vs_lfc_mean:+.2%} "
            "(randomization interval "
            f"[{random_control.relative_mse_difference_vs_lfc_randomization_low:+.2%}, "
            f"{random_control.relative_mse_difference_vs_lfc_randomization_high:+.2%}]); "
            "router versus random: "
            f"{random_control.router_relative_mse_difference_vs_control_mean:+.2%}. "
            f"Observed {random_control.distinct_assignments_observed} distinct assignments "
            f"of {random_control.total_possible_assignments} possible.",
            f"- Oracle relative MSE versus `LFC`: "
            f"{oracle_control.relative_mse_difference_vs_lfc:+.2%}; router regret "
            f"versus oracle: {oracle_control.router_relative_mse_regret:+.2%}.",
            "",
            "One-class gate fallbacks: "
            + (", ".join(report.one_class_folds) if report.one_class_folds else "none")
            + ".",
            "",
            "## Held-out unit audit table",
            "",
            "| Dataset | Horizon | Episode | p(F) | p(C) | Route | Source-fixed | Oracle | "
            "L MSE | LF MSE | LC MSE | LFC MSE | d_u vs LFC |",
            "| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | "
            "---: | ---: |",
            *unit_lines,
            "",
            "The same seed-averaged unit table is stored as `correlation_units.csv` and in "
            "the JSON report.",
            "",
            "## Leave-one-dataset-out folds",
            "",
            "| Held-out dataset | Episodes | Routed MSE | Time-PEFT (`LFC`) MSE | Mean per-unit "
            "relative MSE | Paired 95% CI | "
            "Time reduction | Parameter reduction | Routes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *fold_lines,
            "",
            "## Interpretation guard",
            "",
            f"The CI-based noninferiority diagnostic uses the configured "
            f"{report.noninferiority_margin:.2%} relative-MSE margin. A CPU smoke run "
            "validates plumbing only; a research claim additionally requires the paired "
            "uncertainty analysis described in docs/EXPERIMENT.md and GPU results with a "
            "provenance-checked source head and all preregistered seeds.",
            "",
            "Analytical adapter-parameter calculations are in `correlation_benchmark.json`.",
            "",
        )
    )
    (output / "correlation_benchmark.md").write_text(markdown, encoding="utf-8")


_PAPER_COMPLEX_MSE: dict[str, dict[int, tuple[float, float]]] = {
    "Lorenz": {96: (0.423, 0.134), 192: (0.616, 0.408), 336: (0.742, 0.614)},
    "CellCycle": {96: (0.514, 0.249), 192: (0.703, 0.484), 336: (0.855, 0.675)},
    "DoublePendulum": {
        96: (0.907, 0.817),
        192: (0.984, 0.938),
        336: (1.023, 0.994),
    },
    "Hopfield": {96: (0.499, 0.244), 192: (0.676, 0.440), 336: (0.786, 0.599)},
    "LorenzCoupled": {
        96: (0.619, 0.228),
        192: (0.829, 0.532),
        336: (0.943, 0.723),
    },
    "ECGCA115": {96: (0.573, 0.462), 192: (0.788, 0.683), 336: (1.003, 0.900)},
    "ECGCA515": {96: (0.199, 0.125), 192: (0.255, 0.181), 336: (0.323, 0.250)},
}


def _time_peft_reproduction_config(
    config: DictConfig,
    horizon: int,
    seeds: tuple[int, ...],
) -> TimePEFTReproductionConfig:
    training = config.experiment.training
    if str(training.optimizer).lower() != "adamw":
        raise typer.BadParameter("Paper reproduction currently requires optimizer=AdamW")
    precision = str(training.precision).lower()
    smoke_values = OmegaConf.to_container(config.experiment.smoke_caps, resolve=True)
    if not isinstance(smoke_values, dict):
        raise typer.BadParameter("experiment.smoke_caps must be a mapping")
    smoke_caps = SmokeCaps(**smoke_values)
    if not smoke_caps.active:
        smoke_caps = None
    split = tuple(float(value) for value in config.experiment.preprocessing.complex_split)
    if split != (0.7, 0.1, 0.2):
        raise typer.BadParameter(
            "The current paper reconstruction requires preprocessing.complex_split=[0.7,0.1,0.2]"
        )
    if int(config.experiment.preprocessing.window_stride) != 1:
        raise typer.BadParameter("Paper reproduction requires stride-one windows")
    if not bool(config.experiment.preprocessing.validation_test_prior_context):
        raise typer.BadParameter(
            "Paper reproduction requires validation/test lookback context from prior history"
        )
    gradient_clip_value = training.get("gradient_clip")
    gradient_clip = float(gradient_clip_value) if gradient_clip_value is not None else None
    return TimePEFTReproductionConfig(
        lookback=int(config.experiment.lookback),
        horizon=horizon,
        learning_rates=tuple(float(value) for value in training.learning_rates),
        batch_size=int(training.batch_size),
        max_epochs=int(training.max_epochs),
        early_stopping_patience=int(training.early_stopping_patience),
        early_stopping_min_delta=float(training.early_stopping_min_delta),
        weight_decay=float(training.weight_decay),
        gradient_clip=gradient_clip,
        seeds=seeds,
        device=str(config.device),
        precision=precision,
        complex_split_override_70_10_20=True,
        smoke_caps=smoke_caps,
    )


def _time_peft_reproduction_hash(
    config: DictConfig,
    *,
    dataset_manifest: list[dict[str, object]] | None = None,
) -> str:
    """Hash the complete protocol, including seeds used for common-LR selection."""

    return stable_hash(
        {
            "config": resolved_dict(config),
            "synthetic_generation": _time_peft_synthetic_generation_settings(config),
            "dataset_manifest": dataset_manifest,
            "workflow": "time-peft-paper-v1",
            "implementation_hash": implementation_hash(),
        }
    )


def _time_peft_synthetic_generation_settings(config: DictConfig) -> dict[str, object]:
    preprocessing = config.experiment.preprocessing
    settings: dict[str, object] = {
        "synthetic_generator": str(preprocessing.get("synthetic_generator", "dysts")),
        "synthetic_length": int(preprocessing.get("synthetic_length", 12_000)),
        "synthetic_random_seed": int(preprocessing.get("synthetic_random_seed", 0)),
        "synthetic_pts_per_period": int(
            preprocessing.get("synthetic_pts_per_period", 100)
        ),
    }
    if settings["synthetic_generator"] == "dysts":
        settings["runtime_provenance"] = dysts_reproduction_provenance()
    return settings


def _time_peft_dataset_manifest(
    series_collection: tuple[DatasetSeries, ...],
) -> list[dict[str, object]]:
    """Bind every Cartesian dataset axis to exact bytes and dimensions."""

    return [
        {
            "name": series.name,
            "family": series.family,
            "sha256": series.sha256,
            "channels": int(series.values.shape[0]),
            "timestamps": int(series.values.shape[1]),
        }
        for series in series_collection
    ]


def _validate_time_peft_confirmatory_profile(
    config: DictConfig,
    *,
    datasets: tuple[str, ...],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
    test_role: str,
) -> None:
    """Reserve the confirmatory label for the preregistered full matrix."""

    if test_role != "confirmatory":
        return
    training = config.experiment.training
    preprocessing = config.experiment.preprocessing
    observed: dict[str, object] = {
        "datasets": datasets,
        "horizons": horizons,
        "seeds": seeds,
        "lookback": int(config.experiment.lookback),
        "optimizer": str(training.optimizer),
        "learning_rates": tuple(float(value) for value in training.learning_rates),
        "batch_size": int(training.batch_size),
        "max_epochs": int(training.max_epochs),
        "early_stopping_patience": int(training.early_stopping_patience),
        "early_stopping_min_delta": float(training.early_stopping_min_delta),
        "weight_decay": float(training.weight_decay),
        "gradient_clip": training.gradient_clip,
        "precision": str(training.precision),
        "scaler": str(preprocessing.scaler),
        "window_stride": int(preprocessing.window_stride),
        "complex_split": tuple(float(value) for value in preprocessing.complex_split),
        "validation_test_prior_context": bool(
            preprocessing.validation_test_prior_context
        ),
        "synthetic_generator": str(preprocessing.synthetic_generator),
        "synthetic_length": int(preprocessing.synthetic_length),
        "synthetic_random_seed": int(preprocessing.synthetic_random_seed),
        "synthetic_pts_per_period": int(preprocessing.synthetic_pts_per_period),
        "model_kind": str(config.model.kind),
        "model_id": str(config.model.model_id),
        "model_revision": str(config.model.revision),
        "frequency_top_k": int(config.model.frequency_top_k),
        "claim_test_role": str(config.claim.test_role),
    }
    expected: dict[str, object] = {
        "datasets": tuple(_PAPER_COMPLEX_MSE),
        "horizons": (96, 192, 336),
        "seeds": (0, 1, 2),
        "lookback": 96,
        "optimizer": "AdamW",
        "learning_rates": (1e-3, 1e-4, 1e-5),
        "batch_size": 128,
        "max_epochs": 100,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "weight_decay": 0.01,
        "gradient_clip": None,
        "precision": "fp32",
        "scaler": "target-train-channel-standard",
        "window_stride": 1,
        "complex_split": (0.7, 0.1, 0.2),
        "validation_test_prior_context": True,
        "synthetic_generator": "dysts",
        "synthetic_length": 12_000,
        "synthetic_random_seed": 0,
        "synthetic_pts_per_period": 100,
        "model_kind": "moment",
        "model_id": "AutonLab/MOMENT-1-base",
        "model_revision": "5e44b0ea26376a176360f87831124e018f876d96",
        "frequency_top_k": 3,
        "claim_test_role": "confirmatory",
    }
    mismatched = [key for key, value in expected.items() if observed[key] != value]
    if mismatched:
        raise typer.BadParameter(
            "test-role=confirmatory requires the preregistered full protocol; mismatched: "
            + ", ".join(mismatched)
        )


def _load_time_peft_reproduction_series(
    config: DictConfig,
    dataset: str,
    *,
    download: bool,
) -> Any:
    """Load reproduction data with its explicit paper-facing synthetic generator."""

    synthetic = _time_peft_synthetic_generation_settings(config)
    return load_dataset_series(
        dataset,
        config.paths.data,
        download=download,
        synthetic_generator=str(synthetic["synthetic_generator"]),
        synthetic_length=int(synthetic["synthetic_length"]),
        synthetic_random_seed=int(synthetic["synthetic_random_seed"]),
        synthetic_pts_per_period=int(synthetic["synthetic_pts_per_period"]),
    )


_TIME_PEFT_TEST_ROLES = frozenset(
    {"plumbing-smoke", "development-parity", "confirmatory"}
)


def _resolve_time_peft_test_role(config: DictConfig, requested: str) -> str:
    """Resolve and validate how the locked test outcomes may be interpreted."""

    smoke_values = OmegaConf.to_container(config.experiment.smoke_caps, resolve=True)
    if not isinstance(smoke_values, dict):
        raise typer.BadParameter("experiment.smoke_caps must be a mapping")
    smoke_active = any(value is not None for value in smoke_values.values())
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "plumbing-smoke" if smoke_active else "development-parity"
    if normalized not in _TIME_PEFT_TEST_ROLES:
        choices = ", ".join(sorted(_TIME_PEFT_TEST_ROLES))
        raise typer.BadParameter(f"test-role must be auto or one of: {choices}")
    if smoke_active and normalized != "plumbing-smoke":
        raise typer.BadParameter(
            "Capped reproduction runs must use test-role=plumbing-smoke"
        )
    if not smoke_active and normalized == "plumbing-smoke":
        raise typer.BadParameter(
            "test-role=plumbing-smoke requires at least one explicit smoke cap"
        )
    return normalized


def _time_peft_protocol_lock_payload(
    config: DictConfig,
    *,
    run_hash: str,
    model_revision: str,
    datasets: tuple[str, ...],
    dataset_manifest: list[dict[str, object]],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
    test_role: str,
) -> dict[str, Any]:
    """Describe the immutable reproduction protocol before test access."""

    duplicate_datasets = sorted(
        value for value, count in Counter(datasets).items() if count > 1
    )
    duplicate_horizons = sorted(
        value for value, count in Counter(horizons).items() if count > 1
    )
    if duplicate_datasets or duplicate_horizons:
        details: list[str] = []
        if duplicate_datasets:
            details.append("datasets=" + ",".join(duplicate_datasets))
        if duplicate_horizons:
            details.append(
                "horizons=" + ",".join(str(value) for value in duplicate_horizons)
            )
        raise typer.BadParameter(
            "Time-PEFT Cartesian axes must contain unique values; duplicates: "
            + "; ".join(details)
        )
    if test_role not in _TIME_PEFT_TEST_ROLES:
        raise typer.BadParameter(f"Unresolved Time-PEFT test role: {test_role!r}")
    manifest_names = tuple(str(entry.get("name")) for entry in dataset_manifest)
    if manifest_names != datasets:
        raise typer.BadParameter(
            "Time-PEFT dataset manifest does not match the configured dataset axis"
        )
    cartesian = [
        {"dataset": dataset, "horizon": horizon, "seed": seed}
        for dataset in datasets
        for horizon in horizons
        for seed in seeds
    ]
    return {
        "schema_version": 2,
        "workflow": "time-peft-paper-v1",
        "run_hash": run_hash,
        "implementation_hash": implementation_hash(),
        "model_revision": model_revision,
        "test_role": test_role,
        "test_access_policy": "validation-selection-before-single-test-evaluation",
        "resolved_config": resolved_dict(config),
        "synthetic_generation": _time_peft_synthetic_generation_settings(config),
        "dataset_manifest": dataset_manifest,
        "dataset_horizon_seed_cartesian": cartesian,
    }


def _time_peft_protocol_lock_path(root: Path, run_hash: str) -> Path:
    return root / "protocol-locks" / f"{run_hash}.json"


def _require_time_peft_protocol_lock(
    path: Path,
    expected: dict[str, Any],
    *,
    create: bool,
) -> None:
    """Atomically establish a lock or require its complete exact match."""

    if create and not path.exists():
        _atomic_create_json(path, expected)
    if not path.is_file():
        raise typer.BadParameter(
            f"Missing Time-PEFT protocol lock: {path}. Run --stage tune first."
        )
    observed = _read_json_object(path, label="Time-PEFT protocol lock")
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise typer.BadParameter(
            f"Time-PEFT protocol lock mismatch at {path}; differing fields: "
            + ", ".join(differing)
        )


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete JSON file exactly once without replacing an existing lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # If another process wins the race, the caller checks its lock for an exact match.
        with suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{label} must contain a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise typer.BadParameter(f"Cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _validate_time_peft_tuning_artifact(
    tuning_path: Path,
    metadata_path: Path,
    *,
    config_hash: str,
    model_revision: str,
    protocol_lock_sha256: str,
    test_role: str,
    dataset: str,
    horizon: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Verify tune provenance and bytes before an artifact is skipped or loaded."""

    missing = [
        str(path)
        for path in (tuning_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise typer.BadParameter(
            "Incomplete tune artifact; missing: "
            + ", ".join(missing)
            + ". Run --stage tune with a clean artifact path."
        )
    metadata = _read_json_object(metadata_path, label="Time-PEFT tune sidecar")
    expected_fields: dict[str, object] = {
        "schema_version": 2,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "protocol_lock_sha256": protocol_lock_sha256,
        "test_role": test_role,
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise typer.BadParameter(
                f"Tune sidecar {metadata_path} has incompatible {field}"
            )
    observed_sha256 = metadata.get("tuning_artifact_sha256")
    actual_sha256 = _sha256_file(tuning_path)
    if observed_sha256 != actual_sha256:
        raise typer.BadParameter(
            f"Tune artifact SHA-256 mismatch for {tuning_path}; refusing reuse"
        )
    tuning_metadata = metadata.get("tuning")
    if not isinstance(tuning_metadata, dict):
        raise typer.BadParameter(f"Malformed tuning metadata in {metadata_path}")
    config_metadata = tuning_metadata.get("config")
    if not isinstance(config_metadata, dict):
        raise typer.BadParameter(f"Malformed tuning config metadata in {metadata_path}")
    cell_fields = {
        "dataset": (tuning_metadata.get("dataset"), dataset),
        "dataset_sha256": (tuning_metadata.get("dataset_sha256"), dataset_sha256),
        "horizon": (config_metadata.get("horizon"), horizon),
    }
    mismatched = [
        field for field, (observed, expected) in cell_fields.items() if observed != expected
    ]
    if mismatched:
        raise typer.BadParameter(
            f"Tune sidecar {metadata_path} belongs to a different cell: "
            + ", ".join(mismatched)
        )
    return metadata


def _validate_time_peft_result_artifact(
    path: Path,
    *,
    config_hash: str,
    model_revision: str,
    protocol_lock_sha256: str,
    test_role: str,
    dataset: str,
    horizon: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Require result provenance to match the immutable protocol lock."""

    payload = _read_json_object(path, label="Time-PEFT result artifact")
    expected_fields: dict[str, object] = {
        "schema_version": 2,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "protocol_lock_sha256": protocol_lock_sha256,
        "test_role": test_role,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise typer.BadParameter(f"Result artifact {path} has incompatible {field}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise typer.BadParameter(f"Malformed Time-PEFT result artifact: {path}")
    if (
        result.get("dataset") != dataset
        or result.get("horizon") != horizon
        or result.get("dataset_sha256") != dataset_sha256
    ):
        raise typer.BadParameter(f"Result artifact {path} belongs to a different cell")
    return payload


def _time_peft_template_factory(
    config: DictConfig,
    *,
    horizon: int,
    channels: int,
) -> Any:
    def build(seed: int) -> AdaptableForecaster:
        seed_everything(seed)
        return _build_template(config, horizon, channels=channels)

    return build


def _time_peft_tuning_path(root: Path, dataset: str, horizon: int, run_hash: str) -> Path:
    return root / "tuning" / dataset / f"h{horizon}-{run_hash}.pt"


def _time_peft_trial_cache_dir(
    root: Path,
    dataset: str,
    horizon: int,
    run_hash: str,
) -> Path:
    return root / "trial-cache" / dataset / f"h{horizon}-{run_hash}"


def _tune_time_peft_with_optional_cache(
    template: Any,
    series: Any,
    config: TimePEFTReproductionConfig,
    *,
    trial_cache_dir: Path,
) -> Any:
    """Use resumable core trials when the installed core exposes the optional hook."""

    parameters = inspect.signature(tune_time_peft).parameters
    if "trial_cache" in parameters:
        return tune_time_peft(
            template,
            series,
            config,
            trial_cache=trial_cache_dir,
        )
    if "trial_cache_dir" in parameters:
        return tune_time_peft(
            template,
            series,
            config,
            trial_cache_dir=trial_cache_dir,
        )
    return tune_time_peft(template, series, config)


def _time_peft_result_path(root: Path, dataset: str, horizon: int, run_hash: str) -> Path:
    return root / "results" / dataset / f"h{horizon}-{run_hash}.json"


def _write_time_peft_reproduction_report(
    output: Path,
    payloads: list[dict[str, Any]],
    config: DictConfig,
    run_hash: str,
) -> None:
    """Write paired seed means in the same dataset/horizon layout as the paper."""

    expected_seeds = set(_configured_seed_tuple(config.experiment.seeds))
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        result = payload.get("result")
        if not isinstance(result, dict):
            raise typer.BadParameter("Malformed Time-PEFT result artifact")
        dataset = str(result["dataset"])
        horizon = int(result["horizon"])
        records = result.get("records")
        if not isinstance(records, list):
            raise typer.BadParameter(f"Malformed records for {dataset}/h{horizon}")
        by_method: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise typer.BadParameter(f"Malformed method record for {dataset}/h{horizon}")
            by_method.setdefault(str(record["method_id"]), []).append(record)
        if set(by_method) != set(PAPER_METHOD_IDS):
            raise typer.BadParameter(f"{dataset}/h{horizon} does not contain exact L/LFC results")
        for method_id, method_records in by_method.items():
            found_seeds = {int(record["seed"]) for record in method_records}
            if found_seeds != expected_seeds or len(method_records) != len(expected_seeds):
                raise typer.BadParameter(
                    f"{dataset}/h{horizon}/{method_id} seed coverage is invalid"
                )
        l_records = by_method["L"]
        lfc_records = by_method["LFC"]
        l_mse = _finite_record_mean(l_records, "test_mse", dataset, horizon, "L")
        lfc_mse = _finite_record_mean(lfc_records, "test_mse", dataset, horizon, "LFC")
        l_mae = _finite_record_mean(l_records, "test_mae", dataset, horizon, "L")
        lfc_mae = _finite_record_mean(lfc_records, "test_mae", dataset, horizon, "LFC")
        paper = _PAPER_COMPLEX_MSE.get(dataset, {}).get(horizon)
        rows.append(
            {
                "dataset": dataset,
                "dataset_family": str(result["dataset_family"]),
                "dataset_sha256": str(result["dataset_sha256"]),
                "horizon": horizon,
                "seeds": len(expected_seeds),
                "l_mse": l_mse,
                "lfc_mse": lfc_mse,
                "l_mae": l_mae,
                "lfc_mae": lfc_mae,
                "time_peft_relative_mse_improvement": (l_mse - lfc_mse) / l_mse,
                "selected_l_lr": _common_selected_lr(l_records, dataset, horizon, "L"),
                "selected_lfc_lr": _common_selected_lr(lfc_records, dataset, horizon, "LFC"),
                "paper_l_mse": paper[0] if paper else None,
                "paper_lfc_mse": paper[1] if paper else None,
                "smoke_caps_active": any(
                    bool(record["smoke_caps_active"]) for record in records
                ),
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["horizon"]))

    dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        l_mse = sum(float(row["l_mse"]) for row in selected) / len(selected)
        lfc_mse = sum(float(row["lfc_mse"]) for row in selected) / len(selected)
        dataset_rows.append(
            {
                "dataset": dataset,
                "horizons": [int(row["horizon"]) for row in selected],
                "l_mean_mse": l_mse,
                "lfc_mean_mse": lfc_mse,
                "time_peft_relative_mse_improvement": (l_mse - lfc_mse) / l_mse,
                "paper_l_mean_mse": _paper_horizon_mean(
                    dataset, 0, [int(row["horizon"]) for row in selected]
                ),
                "paper_lfc_mean_mse": _paper_horizon_mean(
                    dataset, 1, [int(row["horizon"]) for row in selected]
                ),
            }
        )

    smoke = any(bool(row["smoke_caps_active"]) for row in rows)
    synthetic_generation = _time_peft_synthetic_generation_settings(config)
    test_roles = {str(payload.get("test_role", "unknown")) for payload in payloads}
    if len(test_roles) != 1:
        raise typer.BadParameter("Result artifacts contain inconsistent test roles")
    dataset_provenance = [
        {
            "dataset": dataset,
            "family": next(
                str(row["dataset_family"]) for row in rows if row["dataset"] == dataset
            ),
            "sha256": next(
                str(row["dataset_sha256"]) for row in rows if row["dataset"] == dataset
            ),
        }
        for dataset in sorted({str(row["dataset"]) for row in rows})
    ]
    report = {
        "schema_version": 2,
        "config_hash": run_hash,
        "test_role": next(iter(test_roles)),
        "paper_id": str(config.claim.paper_id),
        "implementation_label": str(config.claim.implementation_label),
        "official_code_verified": bool(config.claim.official_code_verified),
        "adapter_implementation": str(config.model.adapter_implementation),
        "evaluation_protocol": "target-train-validation-test",
        "interpretation": (
            "confirmatory"
            if next(iter(test_roles)) == "confirmatory"
            else "preliminary-development-only"
        ),
        "smoke_caps_active": smoke,
        "paper_claim": "Time-PEFT improves over LoRA on complex datasets, up to 38%.",
        "paper_claim_scope": "MOMENT-base complex datasets; not universal standard datasets",
        "assumptions": {
            "learning_rate_grid": [
                float(value) for value in config.experiment.training.learning_rates
            ],
            "learning_rate_selection": (
                "per-dataset/horizon/method mean validation MSE across seeds"
            ),
            "max_epochs": int(config.experiment.training.max_epochs),
            "seeds": sorted(expected_seeds),
            "fresh_target_head": True,
            "complex_split": [0.7, 0.1, 0.2],
            "train_only_channel_scaler": True,
            "adapter_dropout": float(config.model.adapter_dropout),
            "head_dropout": float(config.model.get("head_dropout", 0.0)),
            "precision": str(config.experiment.training.precision),
            "synthetic_generation": synthetic_generation,
        },
        "dataset_provenance": dataset_provenance,
        "cells": rows,
        "dataset_horizon_averages": dataset_rows,
        "time_peft_better_cells": sum(float(row["lfc_mse"]) < float(row["l_mse"]) for row in rows),
        "total_cells": len(rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "time_peft_reproduction.json", report)

    test_role = next(iter(test_roles))
    if smoke:
        warning = (
            "> **Smoke only:** window, batch, or epoch caps are active; these values are not "
            "accuracy evidence."
        )
    elif test_role == "confirmatory":
        warning = (
            "> **Protocol-locked confirmatory run.** Numerical parity is still limited by "
            "omitted paper details and unavailable official code."
        )
    else:
        warning = (
            "> **Preliminary development result:** the data windows are uncapped, but the "
            "dataset/horizon/seed/epoch budget may be reduced. Do not interpret this as a "
            "full-paper reproduction or confirmatory result."
        )
    cell_lines = [
        "| "
        + " | ".join(
            (
                str(row["dataset"]),
                str(row["horizon"]),
                f"{float(row['l_mse']):.6g}",
                f"{float(row['lfc_mse']):.6g}",
                f"{float(row['time_peft_relative_mse_improvement']):+.2%}",
                (
                    f"{float(row['paper_l_mse']):.3f} / {float(row['paper_lfc_mse']):.3f}"
                    if row["paper_l_mse"] is not None
                    else "n/a"
                ),
                f"{float(row['selected_l_lr']):.0e} / {float(row['selected_lfc_lr']):.0e}",
            )
        )
        + " |"
        for row in rows
    ]
    average_lines = [
        "| "
        + " | ".join(
            (
                str(row["dataset"]),
                f"{float(row['l_mean_mse']):.6g}",
                f"{float(row['lfc_mean_mse']):.6g}",
                f"{float(row['time_peft_relative_mse_improvement']):+.2%}",
            )
        )
        + " |"
        for row in dataset_rows
    ]
    provenance_lines = [
        f"- `{entry['dataset']}`: `{entry['sha256']}` ({entry['family']})"
        for entry in dataset_provenance
    ]
    runtime = synthetic_generation.get("runtime_provenance")
    if isinstance(runtime, dict):
        numerical_stack = (
            f"dysts {runtime['dysts_version']}, NumPy {runtime['numpy_version']}, "
            f"SciPy {runtime['scipy_version']}, Numba "
            f"{runtime['numba_version'] or 'disabled/not installed'}"
        )
    else:
        numerical_stack = "not applicable"
    markdown = "\n".join(
        (
            "# Time-PEFT versus LoRA reproduction",
            "",
            warning,
            "",
            f"Implementation: **{config.claim.implementation_label}** "
            f"(`{config.model.adapter_implementation}`).",
            "",
            "This workflow trains on target train windows, selects one common LR per method "
            "from validation across seeds, and evaluates the selected checkpoints on test once. "
            "It does not run or compose the correlation router.",
            "",
            "## Dataset and horizon results",
            "",
            "| Dataset | Horizon | LoRA MSE | Time-PEFT MSE | Time-PEFT improvement | "
            "Paper LoRA / Time-PEFT MSE | Selected LR L / LFC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *cell_lines,
            "",
            "## Simple horizon averages",
            "",
            "| Dataset | LoRA MSE | Time-PEFT MSE | Time-PEFT improvement |",
            "| --- | ---: | ---: | ---: |",
            *average_lines,
            "",
            f"Time-PEFT has lower point-estimate MSE in "
            f"{report['time_peft_better_cells']}/{report['total_cells']} cells.",
            "",
            "Paper targets are reference values only and never participate in tuning.",
            "",
            "## Reproduction provenance",
            "",
            f"Test role: **{next(iter(test_roles))}**. Numerical stack: {numerical_stack}.",
            "",
            f"Synthetic trajectory assumption: generator "
            f"`{synthetic_generation['synthetic_generator']}`, length "
            f"{synthetic_generation['synthetic_length']}, seed "
            f"{synthetic_generation['synthetic_random_seed']}, and "
            f"{synthetic_generation['synthetic_pts_per_period']} points per period.",
            "",
            f"Adapter dropout: {float(config.model.adapter_dropout):g}; forecast-head "
            f"dropout: {float(config.model.get('head_dropout', 0.0)):g}.",
            "",
            "Dataset SHA-256 bindings:",
            "",
            *provenance_lines,
            "",
        )
    )
    (output / "time_peft_reproduction.md").write_text(markdown, encoding="utf-8")


def _finite_record_mean(
    records: list[dict[str, Any]],
    field: str,
    dataset: str,
    horizon: int,
    method: str,
) -> float:
    values = [float(record[field]) for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise typer.BadParameter(
            f"Non-finite or empty {field} for {dataset}/h{horizon}/{method}"
        )
    return sum(values) / len(values)


def _common_selected_lr(
    records: list[dict[str, Any]], dataset: str, horizon: int, method: str
) -> float:
    values = {float(record["selected_learning_rate"]) for record in records}
    if len(values) != 1:
        raise typer.BadParameter(
            f"{dataset}/h{horizon}/{method} does not use one validation-selected LR"
        )
    return values.pop()


def _paper_horizon_mean(dataset: str, index: int, horizons: list[int]) -> float | None:
    targets = _PAPER_COMPLEX_MSE.get(dataset)
    if not targets:
        return None
    selected = [targets[horizon][index] for horizon in horizons if horizon in targets]
    return sum(selected) / len(selected) if selected else None


def _controller_config(config: DictConfig) -> ControllerTrainingConfig:
    values = OmegaConf.to_container(config.experiment.controller, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("experiment.controller must be a mapping")
    values.setdefault("device", str(config.device))
    return ControllerTrainingConfig(**values)


def _model_revision(config: DictConfig) -> str:
    revision = str(config.model.revision)
    checkpoint = config.model.get("source_head_checkpoint")
    if checkpoint is None:
        suffix = "random-head" if str(config.model.kind) == "moment" else "base"
        return f"{revision}:{suffix}"
    paths: list[Path] = []
    if isinstance(checkpoint, DictConfig):
        paths = [Path(str(path)) for path in checkpoint.values() if path]
    else:
        paths = [
            Path(str(checkpoint).format(horizon=horizon)) for horizon in config.experiment.horizons
        ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"{revision}:head-{digest.hexdigest()[:16]}"


def _reference_episode_ids(manifests: list[Any]) -> set[str]:
    grouped: dict[tuple[str, int], list[Any]] = {}
    for manifest in manifests:
        grouped.setdefault((manifest.dataset, manifest.horizon), []).append(manifest)
    return {
        sorted(values, key=lambda item: item.support_start)[len(values) // 2].episode_id
        for values in grouped.values()
    }


def _configured_seed_tuple(values: Any, *, field: str = "experiment.seeds") -> tuple[int, ...]:
    try:
        seeds = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(f"{field} must contain integer seeds") from error
    if not seeds:
        raise typer.BadParameter(f"{field} must contain at least one seed")
    duplicates = sorted(seed for seed, count in Counter(seeds).items() if count > 1)
    if duplicates:
        raise typer.BadParameter(f"{field} must contain unique seeds; duplicates: {duplicates}")
    return seeds


def _validate_episode_cartesian_coverage(
    manifests: list[Any],
    *,
    datasets: set[str],
    horizons: set[int],
    episodes_per_cell: int,
) -> None:
    """Require exact, unique episode coverage for every dataset-horizon cell."""

    if not datasets or not horizons:
        raise ValueError("Configured datasets and horizons must be non-empty")
    if episodes_per_cell <= 0:
        raise ValueError("episodes_per_dataset_horizon must be positive")

    expected_cells = {(dataset, horizon) for dataset in datasets for horizon in horizons}
    counts = Counter((str(manifest.dataset), int(manifest.horizon)) for manifest in manifests)
    mismatched = [
        (dataset, horizon, counts[(dataset, horizon)])
        for dataset, horizon in sorted(expected_cells)
        if counts[(dataset, horizon)] != episodes_per_cell
    ]
    unexpected = sorted(set(counts) - expected_cells)
    episode_ids = Counter(str(manifest.episode_id) for manifest in manifests)
    duplicated_ids = sorted(episode_id for episode_id, count in episode_ids.items() if count > 1)
    if not mismatched and not unexpected and not duplicated_ids:
        return

    details: list[str] = []
    if mismatched:
        cell_counts = ", ".join(
            f"{dataset}/h{horizon}={count}"
            for dataset, horizon, count in mismatched
        )
        details.append(
            f"expected {episodes_per_cell} episode(s) per dataset-horizon cell; "
            f"mismatched counts: {cell_counts}"
        )
    if unexpected:
        details.append(
            "unexpected cells: "
            + ", ".join(f"{dataset}/h{horizon}" for dataset, horizon in unexpected)
        )
    if duplicated_ids:
        details.append("duplicate episode IDs: " + ", ".join(duplicated_ids))
    raise ValueError("Episode Cartesian coverage is invalid: " + "; ".join(details))


def _middle_manifests(manifests: list[Any]) -> list[Any]:
    grouped: dict[tuple[str, int], list[Any]] = {}
    for manifest in manifests:
        grouped.setdefault((manifest.dataset, manifest.horizon), []).append(manifest)
    return [
        sorted(values, key=lambda item: item.support_start)[len(values) // 2]
        for _, values in sorted(grouped.items())
    ]


def _oracle_episode_ids(path: str | Path, config_hash: str, model_revision: str) -> set[str] | None:
    target = Path(path)
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as handle:
        gate = json.load(handle)
    if gate.get("config_hash") != config_hash:
        raise RuntimeError("Oracle gate belongs to a different run configuration")
    if gate.get("model_revision") != model_revision:
        raise RuntimeError("Oracle gate belongs to a different model revision")
    episode_ids = set(gate.get("near_ties", {}))
    if not episode_ids:
        raise RuntimeError("Oracle gate does not identify any evaluated episodes")
    return episode_ids


def _record_run(
    layout: ArtifactLayout,
    command: str,
    config: DictConfig,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    payload = resolved_dict(config)
    if extra:
        payload["runtime_overrides"] = extra
    run_id = f"{command}-{_experiment_hash(config, extra=extra)}"
    root = layout.runs / run_id
    atomic_write_json(root / "config.json", payload)
    atomic_write_json(root / "environment.json", environment_metadata())


def _experiment_hash(config: DictConfig, *, extra: dict[str, object] | None = None) -> str:
    payload = resolved_dict(config)
    experiment = payload.get("experiment")
    if isinstance(experiment, dict):
        # Seeds and the optional A7 reference expand an immutable utility table;
        # they do not change any individual action evaluation.
        experiment.pop("seeds", None)
        experiment.pop("include_reference", None)
    return stable_hash(
        {
            "config": payload,
            "runtime_overrides": extra or {},
            "implementation_hash": implementation_hash(),
        }
    )


if __name__ == "__main__":
    app()
