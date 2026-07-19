"""Known action costs, hard-budget filtering, and utility arithmetic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

from utility_peft.types import Budget, UtilityRecord


@dataclass(frozen=True, slots=True)
class ActionCost:
    trainable_parameters: int
    total_parameters: int
    flops: float
    peak_memory_mb: float
    wall_time_s: float

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / max(self.total_parameters, 1)


@dataclass(frozen=True, slots=True)
class CostTable:
    costs: Mapping[str, ActionCost]

    @classmethod
    def from_records(cls, records: list[UtilityRecord]) -> CostTable:
        grouped: dict[str, list[UtilityRecord]] = {}
        for record in records:
            if record.status == "ok":
                grouped.setdefault(record.action_id, []).append(record)
        costs: dict[str, ActionCost] = {}
        for action_id, rows in grouped.items():
            costs[action_id] = ActionCost(
                trainable_parameters=int(median(row.trainable_parameters for row in rows)),
                total_parameters=int(median(row.total_parameters for row in rows)),
                flops=median(row.profiled_flops for row in rows),
                peak_memory_mb=median(row.peak_memory_mb for row in rows),
                wall_time_s=median(row.wall_time_s for row in rows),
            )
        return cls(costs=costs)

    def normalized(self, action_id: str) -> dict[str, float]:
        cost = self.costs[action_id]
        maxima = {
            "parameters": max(
                (item.trainable_fraction for item in self.costs.values()), default=1.0
            ),
            "flops": max((item.flops for item in self.costs.values()), default=1.0),
            "memory": max((item.peak_memory_mb for item in self.costs.values()), default=1.0),
            "time": max((item.wall_time_s for item in self.costs.values()), default=1.0),
        }
        raw = {
            "parameters": cost.trainable_fraction,
            "flops": cost.flops,
            "memory": cost.peak_memory_mb,
            "time": cost.wall_time_s,
        }
        return {name: _safe_ratio(value, maxima[name]) for name, value in raw.items()}

    def feasible(self, action_id: str, budget: Budget) -> bool:
        if action_id not in self.costs:
            return False
        cost = self.costs[action_id]
        checks = (
            budget.max_trainable_parameters is None
            or cost.trainable_parameters <= budget.max_trainable_parameters,
            budget.max_trainable_fraction is None
            or cost.trainable_fraction <= budget.max_trainable_fraction,
            budget.max_peak_memory_mb is None or cost.peak_memory_mb <= budget.max_peak_memory_mb,
            budget.max_wall_time_s is None or cost.wall_time_s <= budget.max_wall_time_s,
        )
        return all(checks)

    def utility(
        self,
        action_id: str,
        predicted_gain: float,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        normalized = self.normalized(action_id)
        weights = weights or {}
        penalty = sum(float(weights.get(name, 0.0)) * value for name, value in normalized.items())
        return predicted_gain - penalty


def filter_feasible(action_ids: list[str], costs: CostTable, budget: Budget) -> list[str]:
    return [action_id for action_id in action_ids if costs.feasible(action_id, budget)]


def _safe_ratio(value: float, maximum: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return value / maximum if maximum > 0 and math.isfinite(maximum) else 0.0
