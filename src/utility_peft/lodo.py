"""Leakage-safe leave-one-dataset-out controller and superiority evaluation."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from utility_peft.actions import MVP_ACTIONS
from utility_peft.controller import ControllerTrainingConfig, train_controller_nested
from utility_peft.costs import CostTable
from utility_peft.types import EvidenceBundle, UtilityRecord
from utility_peft.utils import atomic_write_json

ABLATION_FEATURE_SETS = ("complexity", "structure", "structure_mismatch", "full")


@dataclass(frozen=True, slots=True)
class FeatureSetMetrics:
    ndcg: float
    oracle_regret: float
    top1_accuracy: float
    top2_accuracy: float


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    heldout_dataset: str
    episodes: int
    controller_ndcg: float
    random_ndcg: float
    complexity_ndcg: float
    controller_oracle_regret: float
    source_fixed_oracle_regret: float
    complexity_oracle_regret: float
    source_fixed_action: str
    time_peft_oracle_regret: float
    controller_top1_accuracy: float
    controller_top2_accuracy: float
    complexity_top1_accuracy: float
    complexity_top2_accuracy: float
    mean_relative_mse_vs_time_peft: float
    mean_controller_total_time_s: float
    mean_time_peft_time_s: float
    time_reduction_fraction: float
    median_controller_trainable_parameters: float
    median_time_peft_trainable_parameters: float
    controller_negative_adaptation_rate: float
    time_peft_negative_adaptation_rate: float
    selected_action_counts: Mapping[str, int]
    ablations: Mapping[str, FeatureSetMetrics]


@dataclass(frozen=True, slots=True)
class SuperiorityMetrics:
    mean_relative_mse_difference: float
    relative_mse_ci_low: float
    relative_mse_ci_high: float
    accuracy_superior: bool
    noninferior_within_one_percent: bool
    time_reduction_fraction: float
    fewer_active_parameters: bool
    pareto_superior: bool
    surpasses_matched_time_peft: bool


@dataclass(frozen=True, slots=True)
class EvidenceComparison:
    mean_ndcg_improvement: float
    ndcg_improvement_ci_low: float
    ndcg_improvement_ci_high: float
    mean_regret_reduction: float
    regret_reduction_ci_low: float
    regret_reduction_ci_high: float


@dataclass(frozen=True, slots=True)
class LodoMetrics:
    folds: tuple[FoldMetrics, ...]
    mean_controller_ndcg: float
    mean_random_ndcg: float
    mean_complexity_ndcg: float
    mean_controller_oracle_regret: float
    mean_source_fixed_oracle_regret: float
    controller_beats_both_rankers: bool
    controller_beats_source_fixed_regret: bool
    mean_time_peft_oracle_regret: float
    evidence_comparison: EvidenceComparison
    superiority: SuperiorityMetrics
    hypothesis_h2_passed: bool
    hypothesis_h3_passed: bool
    hypothesis_h4_passed: bool


@dataclass(frozen=True, slots=True)
class _BootstrapValue:
    stratum: tuple[str, int]
    episode_id: str
    value: float


def evaluate_leave_one_dataset_out(
    records: list[UtilityRecord],
    output_dir: str | Path,
    *,
    config: ControllerTrainingConfig | None = None,
    cost_weights: Mapping[str, float] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> LodoMetrics:
    config = config or ControllerTrainingConfig()
    mvp_ids = {action.action_id for action in MVP_ACTIONS}
    records = [
        record
        for record in records
        if record.status == "ok"
        and record.action_id in mvp_ids
        and math.isfinite(record.normalized_gain)
    ]
    datasets = sorted({record.dataset for record in records})
    if len(datasets) < 3:
        raise ValueError("Nested leave-one-dataset-out evaluation requires at least three datasets")
    if not any(record.action_id == "A5" for record in records):
        raise ValueError("Matched Time-PEFT comparison requires action A5")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds: list[FoldMetrics] = []
    relative_mse_rows: list[_BootstrapValue] = []
    ndcg_difference_rows: list[_BootstrapValue] = []
    regret_reduction_rows: list[_BootstrapValue] = []
    controller_times: list[float] = []
    time_peft_times: list[float] = []
    controller_parameters: list[float] = []
    time_peft_parameters: list[float] = []

    for heldout in datasets:
        source = [record for record in records if record.dataset != heldout]
        target = [record for record in records if record.dataset == heldout]
        bundles = {}
        for feature_set in ABLATION_FEATURE_SETS:
            bundle, _ = train_controller_nested(
                source,
                output / heldout / feature_set / "controller.pt",
                config=config,
                feature_set=feature_set,
            )
            bundles[feature_set] = bundle

        source_costs = CostTable.from_records(source)
        source_fixed = _best_fixed(source, source_costs, cost_weights)
        grouped = _episode_action_rows(target)
        fold_feature_values: dict[str, dict[str, list[float]]] = {
            feature_set: {"ndcg": [], "regret": [], "top1": [], "top2": []}
            for feature_set in ABLATION_FEATURE_SETS
        }
        random_ndcg: list[float] = []
        fixed_regret: list[float] = []
        time_peft_regret: list[float] = []
        relative_mse: list[float] = []
        selected_times: list[float] = []
        baseline_times: list[float] = []
        selected_parameters: list[float] = []
        baseline_parameters: list[float] = []
        selected_negative: list[float] = []
        baseline_negative: list[float] = []
        selected_counts: Counter[str] = Counter()

        for episode_index, (episode_id, actions) in enumerate(sorted(grouped.items())):
            action_ids = tuple(sorted(actions))
            if set(action_ids) != set(bundles["full"].action_ids):
                raise ValueError(f"Incomplete held-out action map for episode {episode_id}")
            first = next(iter(actions.values()))
            evidence = EvidenceBundle.from_mapping(episode_id, first["evidence"])
            actual = {
                action_id: source_costs.utility(action_id, values["gain"], cost_weights)
                for action_id, values in actions.items()
            }
            oracle_action = max(actual, key=actual.get)
            oracle = actual[oracle_action]
            actual_order = sorted(actual, key=actual.get, reverse=True)
            selected_by_feature: dict[str, str] = {}
            ndcg_by_feature: dict[str, float] = {}
            regret_by_feature: dict[str, float] = {}
            inference_time_by_feature: dict[str, float] = {}

            for feature_set, bundle in bundles.items():
                started = time.perf_counter()
                predictions = bundle.predict(evidence)
                inference_time_by_feature[feature_set] = time.perf_counter() - started
                scores = {
                    action_id: source_costs.utility(
                        action_id, predictions[action_id][0], cost_weights
                    )
                    for action_id in action_ids
                }
                selected = max(scores, key=scores.get)
                selected_by_feature[feature_set] = selected
                ndcg = _ndcg(actual, scores)
                regret = oracle - actual[selected]
                ndcg_by_feature[feature_set] = ndcg
                regret_by_feature[feature_set] = regret
                values = fold_feature_values[feature_set]
                values["ndcg"].append(ndcg)
                values["regret"].append(regret)
                values["top1"].append(float(selected == oracle_action))
                values["top2"].append(float(selected in actual_order[:2]))

            full_selected = selected_by_feature["full"]
            selected_counts[full_selected] += 1
            fixed_regret.append(oracle - actual[source_fixed])
            time_peft_regret.append(oracle - actual["A5"])
            random_ndcg.append(_random_ndcg(actual, seed=episode_index))
            stratum = (heldout, int(first["horizon"]))
            ndcg_difference_rows.append(
                _BootstrapValue(
                    stratum,
                    episode_id,
                    ndcg_by_feature["full"] - ndcg_by_feature["complexity"],
                )
            )
            regret_reduction_rows.append(
                _BootstrapValue(
                    stratum,
                    episode_id,
                    regret_by_feature["complexity"] - regret_by_feature["full"],
                )
            )

            selected_rows = actions[full_selected]["records"]
            baseline_rows = actions["A5"]["records"]
            selected_by_seed = {record.seed: record for record in selected_rows}
            baseline_by_seed = {record.seed: record for record in baseline_rows}
            common_seeds = sorted(selected_by_seed.keys() & baseline_by_seed.keys())
            if not common_seeds:
                raise ValueError(f"No paired A5 seeds for episode {episode_id}")
            episode_relative = []
            for seed in common_seeds:
                selected_record = selected_by_seed[seed]
                baseline_record = baseline_by_seed[seed]
                difference = (
                    selected_record.adapted_loss - baseline_record.adapted_loss
                ) / max(abs(baseline_record.adapted_loss), 1e-12)
                episode_relative.append(difference)
                selected_negative.append(float(selected_record.normalized_gain < 0))
                baseline_negative.append(float(baseline_record.normalized_gain < 0))
            relative_value = float(np.mean(episode_relative))
            relative_mse.append(relative_value)
            relative_mse_rows.append(
                _BootstrapValue(stratum, episode_id, relative_value)
            )

            evidence_time = float(first["evidence_wall_time_s"])
            selected_time = (
                float(actions[full_selected]["wall_time_s"])
                + evidence_time
                + inference_time_by_feature["full"]
            )
            baseline_time = float(actions["A5"]["wall_time_s"])
            selected_times.append(selected_time)
            baseline_times.append(baseline_time)
            selected_parameters.append(float(actions[full_selected]["trainable_parameters"]))
            baseline_parameters.append(float(actions["A5"]["trainable_parameters"]))

        ablations = {
            feature_set: FeatureSetMetrics(
                ndcg=float(np.mean(values["ndcg"])),
                oracle_regret=float(np.mean(values["regret"])),
                top1_accuracy=float(np.mean(values["top1"])),
                top2_accuracy=float(np.mean(values["top2"])),
            )
            for feature_set, values in fold_feature_values.items()
        }
        time_reduction = 1.0 - float(np.mean(selected_times)) / max(
            float(np.mean(baseline_times)), 1e-12
        )
        folds.append(
            FoldMetrics(
                heldout_dataset=heldout,
                episodes=len(grouped),
                controller_ndcg=ablations["full"].ndcg,
                random_ndcg=float(np.mean(random_ndcg)),
                complexity_ndcg=ablations["complexity"].ndcg,
                controller_oracle_regret=ablations["full"].oracle_regret,
                source_fixed_oracle_regret=float(np.mean(fixed_regret)),
                complexity_oracle_regret=ablations["complexity"].oracle_regret,
                source_fixed_action=source_fixed,
                time_peft_oracle_regret=float(np.mean(time_peft_regret)),
                controller_top1_accuracy=ablations["full"].top1_accuracy,
                controller_top2_accuracy=ablations["full"].top2_accuracy,
                complexity_top1_accuracy=ablations["complexity"].top1_accuracy,
                complexity_top2_accuracy=ablations["complexity"].top2_accuracy,
                mean_relative_mse_vs_time_peft=float(np.mean(relative_mse)),
                mean_controller_total_time_s=float(np.mean(selected_times)),
                mean_time_peft_time_s=float(np.mean(baseline_times)),
                time_reduction_fraction=time_reduction,
                median_controller_trainable_parameters=float(np.median(selected_parameters)),
                median_time_peft_trainable_parameters=float(np.median(baseline_parameters)),
                controller_negative_adaptation_rate=float(np.mean(selected_negative)),
                time_peft_negative_adaptation_rate=float(np.mean(baseline_negative)),
                selected_action_counts=dict(sorted(selected_counts.items())),
                ablations=ablations,
            )
        )
        controller_times.extend(selected_times)
        time_peft_times.extend(baseline_times)
        controller_parameters.extend(selected_parameters)
        time_peft_parameters.extend(baseline_parameters)

    relative_point, relative_low, relative_high = _stratified_bootstrap_interval(
        relative_mse_rows, samples=bootstrap_samples, seed=bootstrap_seed
    )
    ndcg_point, ndcg_low, ndcg_high = _stratified_bootstrap_interval(
        ndcg_difference_rows, samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    regret_point, regret_low, regret_high = _stratified_bootstrap_interval(
        regret_reduction_rows, samples=bootstrap_samples, seed=bootstrap_seed + 2
    )
    time_reduction = 1.0 - float(np.mean(controller_times)) / max(
        float(np.mean(time_peft_times)), 1e-12
    )
    fewer_parameters = float(np.median(controller_parameters)) < float(
        np.median(time_peft_parameters)
    )
    accuracy_superior = relative_high < 0
    noninferior = relative_high <= 0.01
    pareto_superior = noninferior and time_reduction >= 0.20 and fewer_parameters
    superiority = SuperiorityMetrics(
        mean_relative_mse_difference=relative_point,
        relative_mse_ci_low=relative_low,
        relative_mse_ci_high=relative_high,
        accuracy_superior=accuracy_superior,
        noninferior_within_one_percent=noninferior,
        time_reduction_fraction=time_reduction,
        fewer_active_parameters=fewer_parameters,
        pareto_superior=pareto_superior,
        surpasses_matched_time_peft=accuracy_superior or pareto_superior,
    )
    evidence_comparison = EvidenceComparison(
        mean_ndcg_improvement=ndcg_point,
        ndcg_improvement_ci_low=ndcg_low,
        ndcg_improvement_ci_high=ndcg_high,
        mean_regret_reduction=regret_point,
        regret_reduction_ci_low=regret_low,
        regret_reduction_ci_high=regret_high,
    )
    result = LodoMetrics(
        folds=tuple(folds),
        mean_controller_ndcg=float(np.mean([fold.controller_ndcg for fold in folds])),
        mean_random_ndcg=float(np.mean([fold.random_ndcg for fold in folds])),
        mean_complexity_ndcg=float(np.mean([fold.complexity_ndcg for fold in folds])),
        mean_controller_oracle_regret=float(
            np.mean([fold.controller_oracle_regret for fold in folds])
        ),
        mean_source_fixed_oracle_regret=float(
            np.mean([fold.source_fixed_oracle_regret for fold in folds])
        ),
        controller_beats_both_rankers=all(
            fold.controller_ndcg > max(fold.random_ndcg, fold.complexity_ndcg)
            for fold in folds
        ),
        controller_beats_source_fixed_regret=all(
            fold.controller_oracle_regret < fold.source_fixed_oracle_regret for fold in folds
        ),
        mean_time_peft_oracle_regret=float(
            np.mean([fold.time_peft_oracle_regret for fold in folds])
        ),
        evidence_comparison=evidence_comparison,
        superiority=superiority,
        hypothesis_h2_passed=ndcg_low > 0 and regret_low > 0,
        hypothesis_h3_passed=superiority.surpasses_matched_time_peft,
        hypothesis_h4_passed=all(
            fold.controller_ndcg > fold.random_ndcg
            and fold.controller_oracle_regret < fold.source_fixed_oracle_regret
            for fold in folds
        ),
    )
    atomic_write_json(output / "lodo_metrics.json", asdict(result))
    return result


def _episode_action_rows(
    records: list[UtilityRecord],
) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[tuple[str, str], list[UtilityRecord]] = {}
    for record in records:
        grouped.setdefault((record.episode_id, record.action_id), []).append(record)
    output: dict[str, dict[str, dict[str, object]]] = {}
    for (episode_id, action_id), values in grouped.items():
        output.setdefault(episode_id, {})[action_id] = {
            "gain": float(np.mean([record.normalized_gain for record in values])),
            "evidence": dict(values[0].evidence),
            "evidence_wall_time_s": float(
                np.median([record.evidence_wall_time_s for record in values])
            ),
            "wall_time_s": float(np.median([record.wall_time_s for record in values])),
            "trainable_parameters": float(
                np.median([record.trainable_parameters for record in values])
            ),
            "horizon": values[0].horizon,
            "records": tuple(values),
        }
    return output


def _best_fixed(
    records: list[UtilityRecord],
    costs: CostTable,
    weights: Mapping[str, float] | None,
) -> str:
    values: dict[str, list[float]] = {}
    for record in records:
        values.setdefault(record.action_id, []).append(
            costs.utility(record.action_id, record.normalized_gain, weights)
        )
    return max(values, key=lambda action_id: float(np.mean(values[action_id])))


def _ndcg(actual: Mapping[str, float], scores: Mapping[str, float]) -> float:
    actions = sorted(actual)
    relevance = np.asarray([actual[action] for action in actions], dtype=np.float64)
    relevance -= relevance.min()
    if np.allclose(relevance, 0):
        return 1.0
    predicted = np.asarray([scores.get(action, -math.inf) for action in actions])
    order = np.argsort(-predicted)
    ideal = np.argsort(-relevance)
    discounts = np.log2(np.arange(2, len(actions) + 2))
    dcg = np.sum((2**relevance[order] - 1) / discounts)
    idcg = np.sum((2**relevance[ideal] - 1) / discounts)
    return float(dcg / idcg)


def _random_ndcg(actual: Mapping[str, float], *, seed: int) -> float:
    rng = np.random.default_rng(seed)
    actions = sorted(actual)
    samples = []
    for _ in range(256):
        permutation = rng.permutation(len(actions))
        scores = {action: float(permutation[index]) for index, action in enumerate(actions)}
        samples.append(_ndcg(actual, scores))
    return float(np.mean(samples))


def _stratified_bootstrap_interval(
    rows: list[_BootstrapValue],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not rows:
        raise ValueError("Bootstrap requires at least one paired episode value")
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.stratum, {}).setdefault(row.episode_id, []).append(row.value)
    stratum_means = [
        float(np.mean([value for values in episodes.values() for value in values]))
        for episodes in grouped.values()
    ]
    point = float(np.mean(stratum_means))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    strata = sorted(grouped)
    for sample in range(samples):
        sampled_strata = []
        for stratum in strata:
            episodes = grouped[stratum]
            episode_ids = sorted(episodes)
            selected = rng.choice(episode_ids, size=len(episode_ids), replace=True)
            sampled_strata.append(
                float(np.mean([np.mean(episodes[str(episode)]) for episode in selected]))
            )
        estimates[sample] = float(np.mean(sampled_strata))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)
