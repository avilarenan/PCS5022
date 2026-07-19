"""Command-line workflows for the forecasting MVP."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import typer
from omegaconf import DictConfig, OmegaConf

from utility_peft.actions import REFERENCE_ACTION, resolve_actions
from utility_peft.artifacts import ArtifactLayout
from utility_peft.backbones.moment import MomentBackbone
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.config import load_config, resolved_dict
from utility_peft.controller import (
    ControllerTrainingConfig,
    train_controller,
    train_controller_nested,
)
from utility_peft.data.datasets import load_dataset_series
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
    """Run the matched-budget parity subset with an explicit claim guard."""

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
    configured_seeds = tuple(int(value) for value in config.parity.seeds)
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
            dataset = load_dataset_series(str(dataset_name), config.paths.data, download=download)
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
    actions = [
        replace(action, update_steps=0 if action.action_id == "A0" else updates)
        for action in resolve_actions(list(config.experiment.actions))
    ]
    existing = {record.key for record in store.records()}
    revision = _model_revision(config)
    for episode in episodes:
        evidence, evidence_time = _extract_evidence_timed(
            episode.support, template, device="cpu"
        )
        for action in actions:
            for seed_value in config.experiment.seeds:
                seed = int(seed_value)
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
    records = store.records(config_hash=run_hash, model_revision=revision)
    gate = evaluate_oracle_gate(
        records,
        bootstrap_samples=1_000,
        config_hash=run_hash,
        model_revision=revision,
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
    expected_manifests = (
        len(configured_datasets)
        * len(configured_horizons)
        * int(config.experiment.episodes_per_dataset_horizon)
    )
    if len(manifests) != expected_manifests:
        raise typer.BadParameter(
            f"Expected {expected_manifests} configured episode manifests, found "
            f"{len(manifests)}; use a clean artifact root or regenerate the pilot manifests"
        )
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
                [config.experiment.seeds[0]]
                if action.action_id == "A7"
                else config.experiment.seeds
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
                [config.experiment.seeds[0]]
                if action.action_id == "A7"
                else config.experiment.seeds
            )
        }
        current_records = [
            record
            for record in store.records(
                episode_ids=active_episode_ids,
                action_ids={action.action_id for action in actions},
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
    records = UtilityStore(layout.utilities).records(
        episode_ids=episode_ids,
        statuses={"ok"},
        action_ids={f"A{index}" for index in range(7)},
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
    result = evaluate_leave_one_dataset_out(
        UtilityStore(layout.utilities).records(
            episode_ids=episode_ids,
            statuses={"ok"},
            action_ids={f"A{index}" for index in range(7)},
            config_hash=run_hash,
            model_revision=revision,
        ),
        layout.root / "lodo",
        config=_controller_config(config),
        cost_weights=OmegaConf.to_container(config.cost_weights, resolve=True),
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
    report = render_report(
        UtilityStore(layout.utilities),
        output,
        oracle_gate_path=layout.oracle_gate,
        lodo_metrics_path=layout.root / "lodo" / "lodo_metrics.json",
        parity_manifest_path=layout.reports / "time_peft_parity.json",
        episode_ids=episode_ids,
        config_hash=run_hash,
        model_revision=revision,
    )
    typer.echo(str(report))


def _layout(config: DictConfig) -> ArtifactLayout:
    layout = ArtifactLayout.from_root(str(config.paths.artifacts))
    layout.create()
    return layout


def _build_template(config: DictConfig, horizon: int) -> AdaptableForecaster:
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
        )
    else:
        raise ValueError(f"Unknown model kind: {config.model.kind}")
    return AdaptableForecaster(backbone)


def _build_source_head_template(config: DictConfig, horizon: int) -> AdaptableForecaster:
    if str(config.model.kind) == "tiny":
        return _build_template(config, horizon)
    if str(config.model.kind) != "moment":
        raise ValueError(f"Unknown model kind: {config.model.kind}")
    backbone = MomentBackbone(
        lookback=int(config.experiment.lookback),
        horizon=horizon,
        model_id=str(config.model.model_id),
        revision=str(config.model.revision),
        source_head_checkpoint=None,
        allow_random_head=True,
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


def _extract_evidence_timed(
    support: Any,
    template: AdaptableForecaster,
    *,
    device: str,
) -> tuple[Any, float]:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = time.perf_counter()
    evidence = extract_evidence(support, template, device=target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return evidence, time.perf_counter() - started


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


def _middle_manifests(manifests: list[Any]) -> list[Any]:
    grouped: dict[tuple[str, int], list[Any]] = {}
    for manifest in manifests:
        grouped.setdefault((manifest.dataset, manifest.horizon), []).append(manifest)
    return [
        sorted(values, key=lambda item: item.support_start)[len(values) // 2]
        for _, values in sorted(grouped.items())
    ]


def _oracle_episode_ids(
    path: str | Path, config_hash: str, model_revision: str
) -> set[str] | None:
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
