"""Validation helpers for matched action/seed experiment records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

_Record = TypeVar("_Record")


def require_exact_seed_pairing(
    records_by_action: Mapping[str, Sequence[_Record]],
    required_actions: Sequence[str],
    *,
    episode_label: str,
    seed_getter: Callable[[_Record], int],
    expected_seeds: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Require one record per seed and the same seed set for every action.

    The returned seeds are sorted so callers can perform deterministic paired
    comparisons without taking an intersection that would hide missing runs.
    """

    missing_actions = [
        action_id for action_id in required_actions if action_id not in records_by_action
    ]
    if missing_actions:
        raise ValueError(
            f"Complete action coverage is required for episode {episode_label}; "
            f"missing actions: {missing_actions}"
        )

    seeds_by_action: dict[str, tuple[int, ...]] = {}
    for action_id in required_actions:
        seeds = [int(seed_getter(record)) for record in records_by_action[action_id]]
        duplicates = sorted(seed for seed, count in Counter(seeds).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"Duplicate seed rows for episode {episode_label}, action {action_id}: "
                f"{duplicates}"
            )
        seeds_by_action[action_id] = tuple(sorted(seeds))

    reference_action = required_actions[0]
    reference_seeds = seeds_by_action[reference_action]
    if not reference_seeds:
        raise ValueError(
            f"Exact seed pairing requires at least one seed for episode {episode_label}"
        )
    if any(seeds_by_action[action_id] != reference_seeds for action_id in required_actions[1:]):
        details = ", ".join(
            f"{action_id}={list(seeds_by_action[action_id])}" for action_id in required_actions
        )
        raise ValueError(
            f"Exact seed pairing is required for episode {episode_label}; "
            f"action seed sets differ: {details}"
        )
    if expected_seeds is not None:
        expected = tuple(sorted(int(seed) for seed in expected_seeds))
        if not expected:
            raise ValueError("Expected seeds must be non-empty")
        if len(expected) != len(set(expected)):
            raise ValueError(f"Expected seeds must be unique; received {list(expected)}")
        if reference_seeds != expected:
            raise ValueError(
                f"Configured seed coverage is required for episode {episode_label}; "
                f"expected {list(expected)}, observed {list(reference_seeds)}"
            )
    return reference_seeds
