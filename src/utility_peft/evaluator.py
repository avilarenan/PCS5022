"""Exact, fixed-budget action adaptation and query evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utility_peft.model import AdaptableForecaster, count_parameters, model_for_action
from utility_peft.types import ActionSpec, EvaluationEpisode, EvidenceBundle, UtilityRecord
from utility_peft.utils import seed_everything


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    head_lr: float = 1e-3
    adapter_lr: float = 1e-4
    full_lr: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    effective_batch_size: int = 32
    query_batch_size: int = 32
    bf16: bool = True
    profile_flops: bool = True


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    mse: float
    mae: float


def evaluate_action(
    template: AdaptableForecaster,
    episode: EvaluationEpisode,
    action: ActionSpec,
    evidence: EvidenceBundle,
    *,
    seed: int,
    config: TrainingConfig,
    config_hash: str,
    model_revision: str,
    device: str | torch.device,
    evidence_wall_time_s: float = 0.0,
) -> UtilityRecord:
    """Evaluate an action, retrying only a clean OOM/NaN failure once."""

    failure: BaseException | None = None
    for _attempt in range(2):
        try:
            return _evaluate_once(
                template,
                episode,
                action,
                evidence,
                seed=seed,
                config=config,
                config_hash=config_hash,
                model_revision=model_revision,
                device=torch.device(device),
                evidence_wall_time_s=evidence_wall_time_s,
            )
        except BaseException as error:
            if not _retryable(error):
                raise
            failure = error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _failed_record(
        template,
        episode,
        action,
        evidence,
        seed=seed,
        config_hash=config_hash,
        model_revision=model_revision,
        error=failure,
        evidence_wall_time_s=evidence_wall_time_s,
    )


def _evaluate_once(
    template: AdaptableForecaster,
    episode: EvaluationEpisode,
    action: ActionSpec,
    evidence: EvidenceBundle,
    *,
    seed: int,
    config: TrainingConfig,
    config_hash: str,
    model_revision: str,
    device: torch.device,
    evidence_wall_time_s: float,
) -> UtilityRecord:
    seed_everything(seed)
    model = model_for_action(template, action).to(device)
    trainable, total = count_parameters(model)
    support = episode.support
    horizon = support.manifest.horizon
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frozen = _query_metrics(model, episode, device, config.query_batch_size)
    if not math.isfinite(frozen.mse):
        raise FloatingPointError("Frozen query loss is NaN or infinite")

    wall_time = 0.0
    peak_memory_mb = 0.0
    flops = 0.0
    if trainable:
        optimizer = _optimizer(model, action, config)
        x = support.x.to(device)
        y = support.y.to(device)
        mask = support.mask.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        use_amp = config.bf16 and device.type == "cuda"
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            started = time.perf_counter()

        model.train()
        for _step in range(action.update_steps):
            indices = torch.randint(
                x.shape[0],
                (config.effective_batch_size,),
                generator=generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model.predict(x[indices], mask[indices], horizon)
                loss = F.mse_loss(prediction.float(), y[indices].float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Adaptation loss is NaN or infinite")
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.gradient_clip,
            )
            optimizer.step()

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(device)
            wall_time = start_event.elapsed_time(end_event) / 1000
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            wall_time = time.perf_counter() - started
        if config.profile_flops:
            flops = _profile_flops(
                model,
                x[: min(x.shape[0], config.effective_batch_size)],
                y[: min(y.shape[0], config.effective_batch_size)],
                mask[: min(mask.shape[0], config.effective_batch_size)],
                horizon,
                use_amp=use_amp,
            )
            # The profiler observes one optimization step; records represent
            # the complete fixed-budget adaptation run.
            flops *= action.update_steps

    adapted = _query_metrics(model, episode, device, config.query_batch_size)
    if device.type == "cuda":
        peak_memory_mb = max(
            peak_memory_mb,
            torch.cuda.max_memory_allocated(device) / 1024**2,
        )
    if not math.isfinite(adapted.mse):
        raise FloatingPointError("Adapted query loss is NaN or infinite")
    normalized_gain = (frozen.mse - adapted.mse) / max(abs(frozen.mse), 1e-12)
    manifest = support.manifest
    return UtilityRecord(
        episode_id=manifest.episode_id,
        dataset=manifest.dataset,
        dataset_family=manifest.dataset_family,
        horizon=manifest.horizon,
        action_id=action.action_id,
        seed=seed,
        frozen_loss=frozen.mse,
        adapted_loss=adapted.mse,
        normalized_gain=normalized_gain,
        trainable_parameters=trainable,
        stored_adapter_parameters=trainable,
        total_parameters=total,
        profiled_flops=flops,
        peak_memory_mb=peak_memory_mb,
        wall_time_s=wall_time,
        evidence=evidence.as_dict(),
        config_hash=config_hash,
        model_revision=model_revision,
        preprocessing_hash=manifest.preprocessing_hash,
        frozen_mae=frozen.mae,
        adapted_mae=adapted.mae,
        evidence_wall_time_s=evidence_wall_time_s,
    )


def _optimizer(
    model: AdaptableForecaster, action: ActionSpec, config: TrainingConfig
) -> torch.optim.Optimizer:
    if "full" in action.modules:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.full_lr,
            weight_decay=config.weight_decay,
        )
    head_ids = {id(parameter) for parameter in model.backbone.head_parameters()}
    head = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in head_ids
    ]
    adapters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups: list[dict[str, object]] = []
    if head:
        groups.append({"params": head, "lr": config.head_lr})
    if adapters:
        groups.append({"params": adapters, "lr": config.adapter_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


@torch.no_grad()
def _query_metrics(
    model: AdaptableForecaster,
    episode: EvaluationEpisode,
    device: torch.device,
    batch_size: int,
) -> QueryMetrics:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    horizon = episode.support.manifest.horizon
    for start in range(0, episode.query_x.shape[0], batch_size):
        stop = start + batch_size
        x = episode.query_x[start:stop].to(device)
        y = episode.query_y[start:stop].to(device)
        mask = episode.query_mask[start:stop].to(device)
        prediction = model.predict(x, mask, horizon)
        residual = prediction.float() - y.float()
        squared_error += float(residual.square().sum())
        absolute_error += float(residual.abs().sum())
        elements += y.numel()
    return QueryMetrics(
        mse=squared_error / max(elements, 1),
        mae=absolute_error / max(elements, 1),
    )


def _profile_flops(
    model: AdaptableForecaster,
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    horizon: int,
    *,
    use_amp: bool,
) -> float:
    model.zero_grad(set_to_none=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.mse_loss(model.predict(x, mask, horizon).float(), y.float())
        loss.backward()
    model.zero_grad(set_to_none=True)
    return float(sum(event.flops or 0 for event in profiler.key_averages()))


def _retryable(error: BaseException) -> bool:
    return isinstance(error, FloatingPointError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _failed_record(
    template: AdaptableForecaster,
    episode: EvaluationEpisode,
    action: ActionSpec,
    evidence: EvidenceBundle,
    *,
    seed: int,
    config_hash: str,
    model_revision: str,
    error: BaseException | None,
    evidence_wall_time_s: float,
) -> UtilityRecord:
    model = model_for_action(template, action)
    trainable, total = count_parameters(model)
    manifest = episode.support.manifest
    return UtilityRecord(
        episode_id=manifest.episode_id,
        dataset=manifest.dataset,
        dataset_family=manifest.dataset_family,
        horizon=manifest.horizon,
        action_id=action.action_id,
        seed=seed,
        frozen_loss=math.nan,
        adapted_loss=math.nan,
        normalized_gain=math.nan,
        trainable_parameters=trainable,
        stored_adapter_parameters=trainable,
        total_parameters=total,
        profiled_flops=math.nan,
        peak_memory_mb=math.nan,
        wall_time_s=math.nan,
        evidence=evidence.as_dict(),
        config_hash=config_hash,
        model_revision=model_revision,
        preprocessing_hash=manifest.preprocessing_hash,
        status="failed",
        error=repr(error),
        evidence_wall_time_s=evidence_wall_time_s,
    )
