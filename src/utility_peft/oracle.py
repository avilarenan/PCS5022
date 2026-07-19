"""Oracle-map analysis and the mandatory controller-development gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from utility_peft.actions import MVP_ACTIONS
from utility_peft.types import UtilityRecord
from utility_peft.utils import atomic_write_json

ACTION_FAMILY = {
    "A0": "frozen",
    "A1": "head",
    "A2": "lora",
    "A3": "frequency",
    "A4": "channel",
    "A5": "frequency+channel",
    "A6": "fourierft",
}


@dataclass(frozen=True, slots=True)
class OracleGateResult:
    passed: bool
    screen_passed: bool
    confirmation_ready: bool
    heterogeneous_adapter_winners: bool
    positive_fixed_action_regret: bool
    winning_families: tuple[str, ...]
    best_fixed_action: str
    mean_oracle_regret: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    episode_count: int
    seed_count: int
    near_ties: Mapping[str, tuple[str, ...]]
    config_hash: str = ""
    model_revision: str = ""


def evaluate_oracle_gate(
    records: list[UtilityRecord],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    config_hash: str = "",
    model_revision: str = "",
) -> OracleGateResult:
    mvp_ids = {action.action_id for action in MVP_ACTIONS}
    valid = [
        record
        for record in records
        if record.status == "ok"
        and record.action_id in mvp_ids
        and math.isfinite(record.normalized_gain)
    ]
    if not valid:
        raise ValueError("Oracle gate requires successful MVP utility records")
    grouped: dict[tuple[str, str], list[float]] = {}
    episode_strata: dict[str, tuple[str, int]] = {}
    episode_seeds: dict[str, set[int]] = {}
    for record in valid:
        grouped.setdefault((record.episode_id, record.action_id), []).append(
            record.normalized_gain
        )
        episode_strata.setdefault(record.episode_id, (record.dataset, record.horizon))
        episode_seeds.setdefault(record.episode_id, set()).add(record.seed)

    episode_actions: dict[str, dict[str, tuple[float, float]]] = {}
    for (episode_id, action_id), values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        standard_error = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
        episode_actions.setdefault(episode_id, {})[action_id] = (
            float(array.mean()),
            standard_error,
        )
    required_actions = mvp_ids
    incomplete = [
        episode_id
        for episode_id, values in episode_actions.items()
        if set(values) != required_actions
    ]
    if incomplete:
        raise ValueError(
            f"Oracle gate requires A0-A6 for every episode; incomplete: {incomplete[:5]}"
        )

    near_ties: dict[str, tuple[str, ...]] = {}
    primary_winners: dict[str, str] = {}
    for episode_id, action_values in episode_actions.items():
        winner = max(action_values, key=lambda action: action_values[action][0])
        winner_mean, winner_se = action_values[winner]
        primary_winners[episode_id] = winner
        tied = []
        for action_id, (mean, standard_error) in action_values.items():
            difference_se = math.sqrt(winner_se**2 + standard_error**2)
            if winner_mean - mean <= max(difference_se, 1e-12):
                tied.append(action_id)
        near_ties[episode_id] = tuple(sorted(tied))

    winning_families = tuple(
        sorted(
            {
                ACTION_FAMILY[action_id]
                for action_id in primary_winners.values()
                if ACTION_FAMILY[action_id] not in {"frozen", "head"}
            }
        )
    )
    heterogeneous = len(winning_families) >= 2
    episode_ids = sorted(episode_actions)
    fixed_means = {
        action_id: float(
            np.mean([episode_actions[episode_id][action_id][0] for episode_id in episode_ids])
        )
        for action_id in sorted(mvp_ids)
    }
    best_fixed = max(fixed_means, key=fixed_means.get)
    regrets = np.asarray(
        [
            max(value[0] for value in episode_actions[episode_id].values())
            - episode_actions[episode_id][best_fixed][0]
            for episode_id in episode_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    regret_by_episode = dict(zip(episode_ids, regrets, strict=True))
    strata: dict[tuple[str, int], list[str]] = {}
    for episode_id in episode_ids:
        strata.setdefault(episode_strata[episode_id], []).append(episode_id)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for sample in range(bootstrap_samples):
        stratum_means = []
        for episode_group in strata.values():
            selected = rng.choice(episode_group, size=len(episode_group), replace=True)
            stratum_means.append(
                float(np.mean([regret_by_episode[str(episode)] for episode in selected]))
            )
        means[sample] = float(np.mean(stratum_means))
    low, high = np.quantile(means, [0.025, 0.975])
    positive = float(low) > 0
    seed_count = min(len(values) for values in episode_seeds.values())
    screen_passed = heterogeneous and float(regrets.mean()) > 0
    confirmation_ready = seed_count >= 3
    return OracleGateResult(
        passed=screen_passed and positive and confirmation_ready,
        screen_passed=screen_passed,
        confirmation_ready=confirmation_ready,
        heterogeneous_adapter_winners=heterogeneous,
        positive_fixed_action_regret=positive,
        winning_families=winning_families,
        best_fixed_action=best_fixed,
        mean_oracle_regret=float(regrets.mean()),
        bootstrap_ci_low=float(low),
        bootstrap_ci_high=float(high),
        episode_count=len(episode_ids),
        seed_count=seed_count,
        near_ties=near_ties,
        config_hash=config_hash,
        model_revision=model_revision,
    )


def write_oracle_gate(path: str | Path, result: OracleGateResult) -> None:
    atomic_write_json(path, asdict(result))


def require_oracle_gate(
    path: str | Path,
    *,
    config_hash: str | None = None,
    model_revision: str | None = None,
) -> None:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not result.get("passed", False):
        raise RuntimeError("Oracle gate failed: controller development is intentionally blocked")
    if config_hash is not None and result.get("config_hash") != config_hash:
        raise RuntimeError("Oracle gate belongs to a different run configuration")
    if model_revision is not None and result.get("model_revision") != model_revision:
        raise RuntimeError("Oracle gate belongs to a different model revision")
