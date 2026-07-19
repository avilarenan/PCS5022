"""Chronological, leakage-auditable forecasting episodes."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from utility_peft.types import EpisodeManifest, EvaluationEpisode, SupportView
from utility_peft.utils import atomic_write_json, preprocessing_hash, stable_hash

PREPROCESSING_SPEC = {
    "version": 1,
    "layout": "batch-channel-time",
    "scaler": "support-channel-standard",
    "missing": "zero-after-scaling",
    "window_stride": 1,
    "support_query_raw_overlap": False,
}


def chronological_starts(
    series_length: int,
    *,
    lookback: int,
    horizon: int,
    support_size: int,
    query_size: int,
    episodes: int,
) -> tuple[int, ...]:
    """Place episode starts in chronological order across the available series."""

    span = episode_span(
        lookback=lookback,
        horizon=horizon,
        support_size=support_size,
        query_size=query_size,
    )
    latest = series_length - span
    if latest < 0:
        raise ValueError(f"Series length {series_length} is shorter than episode span {span}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes == 1:
        return (latest // 2,)
    return tuple(int(value) for value in np.linspace(0, latest, episodes))


def chronological_starts_in_range(
    start: int,
    end: int,
    *,
    lookback: int,
    horizon: int,
    support_size: int,
    query_size: int,
    episodes: int,
) -> tuple[int, ...]:
    """Place episode starts inside one immutable raw-timestamp partition."""

    if start < 0 or end <= start:
        raise ValueError(f"Invalid chronological range [{start}, {end})")
    local = chronological_starts(
        end - start,
        lookback=lookback,
        horizon=horizon,
        support_size=support_size,
        query_size=query_size,
        episodes=episodes,
    )
    return tuple(start + value for value in local)


def episode_span(*, lookback: int, horizon: int, support_size: int, query_size: int) -> int:
    window = lookback + horizon
    return (support_size - 1) + window + (query_size - 1) + window


def build_episode(
    series: Tensor,
    *,
    dataset: str,
    dataset_family: str,
    lookback: int,
    horizon: int,
    support_size: int,
    query_size: int,
    start: int,
    seed: int,
    subject_id: str | None = None,
    session_id: str | None = None,
    source_hash: str | None = None,
    partition: str = "full",
) -> EvaluationEpisode:
    """Build support and query windows with disjoint raw timestamp intervals."""

    if series.ndim != 2:
        raise ValueError("series must have shape [channels, time]")
    if min(lookback, horizon, support_size, query_size) <= 0:
        raise ValueError("episode dimensions must be positive")
    window = lookback + horizon
    support_start = start
    support_end = support_start + support_size - 1 + window
    query_start = support_end
    query_end = query_start + query_size - 1 + window
    if query_end > series.shape[1]:
        raise ValueError("episode extends beyond the source series")

    fitted = series[:, support_start:support_end]
    mean = torch.nanmean(fitted, dim=1, keepdim=True)
    centered = fitted - mean
    std = torch.sqrt(torch.nanmean(centered.square(), dim=1, keepdim=True)).clamp_min(1e-6)
    normalized = (series - mean) / std
    digest = preprocessing_hash(PREPROCESSING_SPEC)
    if source_hash is None:
        source_hash = hashlib.sha256(
            series.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    identity = {
        "dataset": dataset,
        "dataset_family": dataset_family,
        "lookback": lookback,
        "horizon": horizon,
        "support_size": support_size,
        "query_size": query_size,
        "support_start": support_start,
        "support_end": support_end,
        "query_start": query_start,
        "query_end": query_end,
        "seed": seed,
        "preprocessing_hash": digest,
        "subject_id": subject_id,
        "session_id": session_id,
        "source_hash": source_hash,
        "partition": partition,
    }
    manifest = EpisodeManifest(episode_id=stable_hash(identity, length=24), **identity)
    support_x, support_y, support_mask = _windows(
        normalized,
        first=support_start,
        count=support_size,
        lookback=lookback,
        horizon=horizon,
    )
    query_x, query_y, query_mask = _windows(
        normalized,
        first=query_start,
        count=query_size,
        lookback=lookback,
        horizon=horizon,
    )
    support = SupportView(
        x=support_x,
        y=support_y,
        mask=support_mask,
        manifest=manifest,
    )
    return EvaluationEpisode(
        support=support,
        query_x=query_x,
        query_y=query_y,
        query_mask=query_mask,
    )


def assert_no_timestamp_overlap(manifest: EpisodeManifest) -> None:
    if manifest.support_end > manifest.query_start:
        raise AssertionError("Support and query reference overlapping raw timestamps")


class EpisodeRepository:
    """Local tensor storage with independently inspectable manifests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, episode: EvaluationEpisode) -> Path:
        episode_id = episode.support.manifest.episode_id
        target = self.root / episode.support.manifest.dataset / f"{episode_id}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "support_x": episode.support.x.cpu(),
            "support_y": episode.support.y.cpu(),
            "support_mask": episode.support.mask.cpu(),
            "query_x": episode.query_x.cpu(),
            "query_y": episode.query_y.cpu(),
            "query_mask": episode.query_mask.cpu(),
        }
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        temporary.replace(target)
        atomic_write_json(target.with_suffix(".json"), asdict(episode.support.manifest))
        return target

    def load(self, episode_id: str) -> EvaluationEpisode:
        matches = list(self.root.glob(f"*/{episode_id}.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one manifest for episode {episode_id}")
        manifest_path = matches[0]
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = EpisodeManifest(**json.load(handle))
        tensors = torch.load(
            manifest_path.with_suffix(".pt"), map_location="cpu", weights_only=True
        )
        support = SupportView(
            x=tensors["support_x"],
            y=tensors["support_y"],
            mask=tensors["support_mask"],
            manifest=manifest,
        )
        return EvaluationEpisode(
            support=support,
            query_x=tensors["query_x"],
            query_y=tensors["query_y"],
            query_mask=tensors["query_mask"],
        )

    def manifests(self, *, dataset: str | None = None) -> list[EpisodeManifest]:
        pattern = f"{dataset}/*.json" if dataset else "*/*.json"
        manifests: list[EpisodeManifest] = []
        for path in sorted(self.root.glob(pattern)):
            with path.open(encoding="utf-8") as handle:
                manifests.append(EpisodeManifest(**json.load(handle)))
        return manifests


def _windows(
    series: Tensor, *, first: int, count: int, lookback: int, horizon: int
) -> tuple[Tensor, Tensor, Tensor]:
    length = lookback + horizon
    selected = series[:, first : first + count - 1 + length].unfold(1, length, 1)
    selected = selected[:, :count].permute(1, 0, 2).contiguous().float()
    x = selected[..., :lookback]
    y = selected[..., lookback:]
    mask = torch.isfinite(x).all(dim=1)
    return torch.nan_to_num(x), torch.nan_to_num(y), mask.to(torch.bool)
