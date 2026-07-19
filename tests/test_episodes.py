from __future__ import annotations

import dataclasses

import pytest
import torch

from utility_peft.episodes import (
    EpisodeRepository,
    assert_no_timestamp_overlap,
    build_episode,
    chronological_starts,
)


def test_shapes_masks_and_raw_timestamp_chronology(episode) -> None:
    support = episode.support
    assert support.x.shape == (8, 3, 16)
    assert support.y.shape == (8, 3, 8)
    assert support.mask.shape == (8, 16)
    assert support.mask.dtype == torch.bool
    assert episode.query_x.shape == (8, 3, 16)
    assert support.manifest.support_end == support.manifest.query_start
    assert_no_timestamp_overlap(support.manifest)


def test_episode_id_and_tensors_are_deterministic(series: torch.Tensor) -> None:
    kwargs = {
        "dataset": "deterministic",
        "dataset_family": "test",
        "lookback": 16,
        "horizon": 8,
        "support_size": 8,
        "query_size": 8,
        "start": 4,
        "seed": 9,
    }
    first = build_episode(series, **kwargs)
    second = build_episode(series.clone(), **kwargs)
    assert first.support.manifest.episode_id == second.support.manifest.episode_id
    assert torch.equal(first.support.x, second.support.x)
    assert torch.equal(first.query_y, second.query_y)


def test_query_values_cannot_affect_support_preprocessing(series: torch.Tensor) -> None:
    kwargs = {
        "dataset": "leakage",
        "dataset_family": "test",
        "lookback": 16,
        "horizon": 8,
        "support_size": 8,
        "query_size": 8,
        "start": 0,
        "seed": 0,
    }
    original = build_episode(series, **kwargs)
    changed = series.clone()
    changed[:, original.support.manifest.query_start :] += 10_000
    modified = build_episode(changed, **kwargs)
    assert torch.equal(original.support.x, modified.support.x)
    assert torch.equal(original.support.y, modified.support.y)


def test_chronological_starts_validate_available_length() -> None:
    starts = chronological_starts(
        200,
        lookback=16,
        horizon=8,
        support_size=8,
        query_size=8,
        episodes=5,
    )
    assert starts == tuple(sorted(starts))
    assert len(set(starts)) == 5
    with pytest.raises(ValueError, match="shorter"):
        chronological_starts(
            20,
            lookback=16,
            horizon=8,
            support_size=8,
            query_size=8,
            episodes=1,
        )


def test_manifest_rejects_overlap(episode) -> None:
    manifest = episode.support.manifest
    with pytest.raises(ValueError, match="overlaps"):
        dataclasses.replace(manifest, query_start=manifest.support_end - 1)


def test_episode_repository_round_trip(tmp_path, episode) -> None:
    repository = EpisodeRepository(tmp_path)
    repository.save(episode)
    loaded = repository.load(episode.support.manifest.episode_id)
    assert loaded.support.manifest == episode.support.manifest
    assert torch.equal(loaded.support.x, episode.support.x)
    assert torch.equal(loaded.query_y, episode.query_y)
    assert repository.manifests() == [episode.support.manifest]
