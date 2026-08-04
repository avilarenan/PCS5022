"""Correlation router training, leave-one-dataset-out evaluation, and cost analysis."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utility_peft.correlation import (
    CHANNEL_ROUTING_FEATURES,
    FREQUENCY_ROUTING_FEATURES,
)
from utility_peft.matching import require_exact_seed_pairing
from utility_peft.types import EvidenceBundle, UtilityRecord


@dataclass(frozen=True, slots=True)
class RouterActionIds:
    """Matched actions used by the two independent module gates."""

    base: str = "L"
    frequency: str = "LF"
    channel: str = "LC"
    full: str = "LFC"

    def __post_init__(self) -> None:
        if len({self.base, self.frequency, self.channel, self.full}) != 4:
            raise ValueError("Correlation-routing action IDs must be distinct")

    @property
    def all(self) -> tuple[str, str, str, str]:
        return (self.base, self.frequency, self.channel, self.full)

    def route(self, *, frequency: bool, channel: bool) -> str:
        if frequency and channel:
            return self.full
        if frequency:
            return self.frequency
        if channel:
            return self.channel
        return self.base


@dataclass(frozen=True, slots=True)
class CorrelationRouterConfig:
    """Source-only training choices for the two binary logistic gates."""

    probability_threshold: float = 0.5
    min_relative_benefit: float = 0.0
    noninferiority_margin: float = 0.01
    regularization_c: float = 1.0
    max_iter: int = 1_000
    random_state: int = 0
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0
    bootstrap_confidence_level: float = 0.95
    random_control_repeats: int = 1_000
    frequency_features: tuple[str, ...] = FREQUENCY_ROUTING_FEATURES
    channel_features: tuple[str, ...] = CHANNEL_ROUTING_FEATURES

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1]")
        if self.min_relative_benefit < 0:
            raise ValueError("min_relative_benefit must be non-negative")
        if self.noninferiority_margin < 0:
            raise ValueError("noninferiority_margin must be non-negative")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("bootstrap_confidence_level must lie in (0, 1)")
        if self.random_control_repeats <= 0:
            raise ValueError("random_control_repeats must be positive")
        if not self.frequency_features or not self.channel_features:
            raise ValueError("Each module gate needs at least one feature")


@dataclass(frozen=True, slots=True)
class BinaryRoutingModel:
    """One logistic gate, including a deterministic one-class fallback."""

    feature_names: tuple[str, ...]
    estimator: Any | None
    constant_probability: float | None
    training_examples: int
    positive_examples: int

    def predict_probability(self, evidence: Mapping[str, float] | EvidenceBundle) -> float:
        values = _evidence_mapping(evidence)
        if self.constant_probability is not None:
            return self.constant_probability
        if self.estimator is None:
            raise RuntimeError("Binary routing model has not been fitted")
        row = np.asarray(
            [[_numeric_or_nan(values.get(name)) for name in self.feature_names]],
            dtype=np.float64,
        )
        probability = float(self.estimator.predict_proba(row)[0, 1])
        return min(max(probability, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    action_id: str
    frequency_active: bool
    channel_active: bool
    frequency_probability: float
    channel_probability: float


@dataclass(frozen=True, slots=True)
class CorrelationRouter:
    """Independent residual-autocorrelation and cross-channel gates."""

    frequency_model: BinaryRoutingModel
    channel_model: BinaryRoutingModel
    action_ids: RouterActionIds
    probability_threshold: float
    min_relative_benefit: float
    training_datasets: tuple[str, ...]

    def predict(self, evidence: Mapping[str, float] | EvidenceBundle) -> RoutingDecision:
        frequency_probability = self.frequency_model.predict_probability(evidence)
        channel_probability = self.channel_model.predict_probability(evidence)
        frequency_active = frequency_probability >= self.probability_threshold
        channel_active = channel_probability >= self.probability_threshold
        return RoutingDecision(
            action_id=self.action_ids.route(
                frequency=frequency_active,
                channel=channel_active,
            ),
            frequency_active=frequency_active,
            channel_active=channel_active,
            frequency_probability=frequency_probability,
            channel_probability=channel_probability,
        )

    def select(self, evidence: Mapping[str, float] | EvidenceBundle) -> str:
        return self.predict(evidence).action_id


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    """Mean matched-run accuracy and computation measurements."""

    runs: int
    mse: float
    mae: float | None
    adaptation_wall_time_s: float
    evidence_wall_time_s: float
    routing_wall_time_s: float
    end_to_end_wall_time_s: float
    trainable_parameters: float
    stored_adapter_parameters: float
    peak_memory_mb: float
    profiled_flops: float


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Router changes relative to the always-on LFC baseline.

    Accuracy differences are equal-weighted means of paired unit-level relative
    differences after seed averaging. Cost differences use the descriptive run
    aggregates because their units share a common physical scale.
    """

    relative_mse_difference: float
    relative_mse_ci_low: float
    relative_mse_ci_high: float
    relative_mae_difference: float | None
    end_to_end_time_reduction_fraction: float
    trainable_parameter_reduction_fraction: float
    stored_parameter_reduction_fraction: float
    peak_memory_reduction_fraction: float
    flops_reduction_fraction: float
    noninferiority_margin: float
    bootstrap_samples: int
    bootstrap_confidence_level: float
    accuracy_superior: bool
    point_noninferior_within_margin: bool
    noninferior_within_margin: bool


@dataclass(frozen=True, slots=True)
class SourceFixedControl:
    """Best fixed arm selected independently inside every source-only fold."""

    actions_by_heldout_dataset: Mapping[str, str]
    route_counts: Mapping[str, int]
    aggregate: BenchmarkAggregate
    relative_mse_difference_vs_lfc: float
    router_relative_mse_difference_vs_control: float
    router_relative_mse_difference_vs_control_ci_low: float
    router_relative_mse_difference_vs_control_ci_high: float


@dataclass(frozen=True, slots=True)
class OracleControl:
    """Target-query-informed accuracy ceiling; never a deployable selector."""

    route_counts: Mapping[str, int]
    aggregate: BenchmarkAggregate
    relative_mse_difference_vs_lfc: float
    router_relative_mse_regret: float


@dataclass(frozen=True, slots=True)
class RandomMatchedControl:
    """Random target assignments that exactly preserve each fold's route histogram."""

    repeats: int
    assignment_scope: str
    route_counts: Mapping[str, int]
    distinct_assignments_observed: int
    total_possible_assignments: int
    descriptive_only: bool
    relative_mse_difference_vs_lfc_mean: float
    relative_mse_difference_vs_lfc_randomization_low: float
    relative_mse_difference_vs_lfc_randomization_high: float
    router_relative_mse_difference_vs_control_mean: float
    router_relative_mse_difference_vs_control_randomization_low: float
    router_relative_mse_difference_vs_control_randomization_high: float


@dataclass(frozen=True, slots=True)
class CorrelationControlReport:
    source_fixed: SourceFixedControl
    random_histogram_matched: RandomMatchedControl
    oracle: OracleControl


@dataclass(frozen=True, slots=True)
class CorrelationUnitAudit:
    """One inspectable held-out unit after seed averaging."""

    dataset: str
    horizon: int
    episode_id: str
    seeds: tuple[int, ...]
    frequency_probability: float
    channel_probability: float
    routed_action: str
    source_fixed_action: str
    oracle_action: str
    arm_seed_mean_mse: Mapping[str, float]
    router_seed_mean_mse: float
    lfc_seed_mean_mse: float
    relative_mse_difference_vs_lfc: float


@dataclass(frozen=True, slots=True)
class CorrelationFoldReport:
    heldout_dataset: str
    training_datasets: tuple[str, ...]
    episodes: int
    route_counts: Mapping[str, int]
    router: BenchmarkAggregate
    baseline: BenchmarkAggregate
    comparison: BenchmarkComparison
    frequency_training_examples: int
    frequency_positive_examples: int
    channel_training_examples: int
    channel_positive_examples: int


@dataclass(frozen=True, slots=True)
class CorrelationBenchmarkReport:
    """Complete LODO result; each dataset contributes target episodes once."""

    folds: tuple[CorrelationFoldReport, ...]
    episodes: int
    route_counts: Mapping[str, int]
    router: BenchmarkAggregate
    baseline: BenchmarkAggregate
    comparison: BenchmarkComparison
    action_ids: RouterActionIds
    probability_threshold: float
    min_relative_benefit: float
    noninferiority_margin: float
    controls: CorrelationControlReport
    one_class_folds: tuple[str, ...]
    unit_table: tuple[CorrelationUnitAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperParameterSavings:
    """Paper-formula active trainable parameters for probabilistic routing."""

    hidden_size: int
    frequency_hidden_size: int
    channel_rank: int
    channels: int
    fixed_trainable_parameters: int
    frequency_adapter_parameters: int
    channel_adapter_parameters: int
    output_norm_parameters: int
    always_on_trainable_parameters: int
    expected_router_trainable_parameters: float
    expected_saved_parameters: float
    expected_reduction_fraction: float
    active_parameters_by_route: Mapping[str, int]


@dataclass(slots=True)
class _EpisodeActions:
    dataset: str
    episode_id: str
    horizon: int
    evidence: dict[str, float]
    seeds: tuple[int, ...]
    records: dict[str, tuple[Any, ...]]


@dataclass(slots=True)
class _MatchedObservation:
    unit_key: tuple[str, int, str]
    selected: Any
    baseline: Any
    evidence_wall_time_s: float
    routing_wall_time_s: float


def train_correlation_router(
    records: Sequence[UtilityRecord],
    *,
    heldout_dataset: str | None = None,
    config: CorrelationRouterConfig | None = None,
    action_ids: RouterActionIds | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> CorrelationRouter:
    """Fit both gates, excluding ``heldout_dataset`` before label construction."""

    config = config or CorrelationRouterConfig()
    action_ids = action_ids or infer_router_action_ids(records)
    episodes = _complete_episodes(records, action_ids, expected_seeds=expected_seeds)
    source = [episode for episode in episodes if episode.dataset != heldout_dataset]
    if not source:
        raise ValueError("No complete source episodes are available for router training")

    frequency_rows: list[dict[str, float]] = []
    frequency_labels: list[int] = []
    channel_rows: list[dict[str, float]] = []
    channel_labels: list[int] = []
    for episode in source:
        losses = {
            action_id: _mean_finite(
                [_record_float(record, "adapted_loss") for record in episode.records[action_id]]
            )
            for action_id in action_ids.all
        }
        frequency_benefit = float(
            np.mean(
                (
                    _relative_improvement(losses[action_ids.base], losses[action_ids.frequency]),
                    _relative_improvement(losses[action_ids.channel], losses[action_ids.full]),
                )
            )
        )
        channel_benefit = float(
            np.mean(
                (
                    _relative_improvement(losses[action_ids.base], losses[action_ids.channel]),
                    _relative_improvement(losses[action_ids.frequency], losses[action_ids.full]),
                )
            )
        )
        frequency_rows.append(episode.evidence)
        frequency_labels.append(int(frequency_benefit > config.min_relative_benefit))
        channel_rows.append(episode.evidence)
        channel_labels.append(int(channel_benefit > config.min_relative_benefit))

    return CorrelationRouter(
        frequency_model=_fit_binary_gate(
            frequency_rows,
            frequency_labels,
            config.frequency_features,
            config,
        ),
        channel_model=_fit_binary_gate(
            channel_rows,
            channel_labels,
            config.channel_features,
            config,
        ),
        action_ids=action_ids,
        probability_threshold=config.probability_threshold,
        min_relative_benefit=config.min_relative_benefit,
        training_datasets=tuple(sorted({episode.dataset for episode in source})),
    )


def evaluate_correlation_lodo(
    records: Sequence[UtilityRecord],
    *,
    config: CorrelationRouterConfig | None = None,
    action_ids: RouterActionIds | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> CorrelationBenchmarkReport:
    """Evaluate support-only correlation routing with leave-one-dataset-out fits."""

    config = config or CorrelationRouterConfig()
    action_ids = action_ids or infer_router_action_ids(records)
    episodes = _complete_episodes(records, action_ids, expected_seeds=expected_seeds)
    datasets = sorted({episode.dataset for episode in episodes})
    if len(datasets) < 2:
        raise ValueError("Correlation LODO evaluation requires at least two datasets")

    folds: list[CorrelationFoldReport] = []
    all_observations: list[_MatchedObservation] = []
    all_source_fixed_observations: list[_MatchedObservation] = []
    all_router_vs_source_fixed: list[_MatchedObservation] = []
    all_oracle_observations: list[_MatchedObservation] = []
    all_router_vs_oracle: list[_MatchedObservation] = []
    all_routes: Counter[str] = Counter()
    source_fixed_routes: Counter[str] = Counter()
    oracle_routes: Counter[str] = Counter()
    source_fixed_actions: dict[str, str] = {}
    routed_units: list[tuple[_EpisodeActions, str]] = []
    unit_table: list[CorrelationUnitAudit] = []
    one_class_folds: list[str] = []
    for fold_index, heldout_dataset in enumerate(datasets):
        router = train_correlation_router(
            records,
            heldout_dataset=heldout_dataset,
            config=config,
            action_ids=action_ids,
            expected_seeds=expected_seeds,
        )
        source = [episode for episode in episodes if episode.dataset != heldout_dataset]
        source_fixed_action = _best_source_fixed_action(source, action_ids)
        source_fixed_actions[heldout_dataset] = source_fixed_action
        if router.frequency_model.positive_examples in {
            0,
            router.frequency_model.training_examples,
        }:
            one_class_folds.append(f"{heldout_dataset}:frequency")
        if router.channel_model.positive_examples in {
            0,
            router.channel_model.training_examples,
        }:
            one_class_folds.append(f"{heldout_dataset}:channel")
        target = sorted(
            (episode for episode in episodes if episode.dataset == heldout_dataset),
            key=lambda episode: (episode.horizon, episode.episode_id),
        )
        observations: list[_MatchedObservation] = []
        route_counts: Counter[str] = Counter()
        for episode in target:
            started = time.perf_counter()
            decision = router.predict(episode.evidence)
            routing_time = time.perf_counter() - started
            selected_by_seed = {
                int(_record_value(record, "seed")): record
                for record in episode.records[decision.action_id]
            }
            baseline_by_seed = {
                int(_record_value(record, "seed")): record
                for record in episode.records[action_ids.full]
            }
            route_counts[decision.action_id] += 1
            evidence_time = _mean_finite(
                [
                    _record_float(record, "evidence_wall_time_s", default=0.0)
                    for record in episode.records[action_ids.base]
                ]
            )
            oracle_action = min(
                action_ids.all,
                key=lambda action_id: (
                    _episode_action_mean(episode, action_id, "adapted_loss"),
                    action_id,
                ),
            )
            arm_mean_mse = {
                action_id: _episode_action_mean(episode, action_id, "adapted_loss")
                for action_id in action_ids.all
            }
            routed_mse = arm_mean_mse[decision.action_id]
            lfc_mse = arm_mean_mse[action_ids.full]
            routed_units.append((episode, decision.action_id))
            unit_table.append(
                CorrelationUnitAudit(
                    dataset=episode.dataset,
                    horizon=episode.horizon,
                    episode_id=episode.episode_id,
                    seeds=episode.seeds,
                    frequency_probability=decision.frequency_probability,
                    channel_probability=decision.channel_probability,
                    routed_action=decision.action_id,
                    source_fixed_action=source_fixed_action,
                    oracle_action=oracle_action,
                    arm_seed_mean_mse=dict(sorted(arm_mean_mse.items())),
                    router_seed_mean_mse=routed_mse,
                    lfc_seed_mean_mse=lfc_mse,
                    relative_mse_difference_vs_lfc=_relative_difference(
                        routed_mse,
                        lfc_mse,
                    ),
                )
            )
            source_fixed_routes[source_fixed_action] += 1
            oracle_routes[oracle_action] += 1
            fixed_by_seed = _records_by_seed(episode, source_fixed_action)
            oracle_by_seed = _records_by_seed(episode, oracle_action)
            for seed in episode.seeds:
                unit_key = (episode.dataset, episode.horizon, episode.episode_id)
                routed = _MatchedObservation(
                    unit_key=unit_key,
                    selected=selected_by_seed[seed],
                    baseline=baseline_by_seed[seed],
                    evidence_wall_time_s=evidence_time,
                    routing_wall_time_s=routing_time,
                )
                observations.append(routed)
                all_source_fixed_observations.append(
                    _MatchedObservation(
                        unit_key=unit_key,
                        selected=fixed_by_seed[seed],
                        baseline=baseline_by_seed[seed],
                        evidence_wall_time_s=0.0,
                        routing_wall_time_s=0.0,
                    )
                )
                all_router_vs_source_fixed.append(
                    _MatchedObservation(
                        unit_key=unit_key,
                        selected=selected_by_seed[seed],
                        baseline=fixed_by_seed[seed],
                        evidence_wall_time_s=evidence_time,
                        routing_wall_time_s=routing_time,
                    )
                )
                all_oracle_observations.append(
                    _MatchedObservation(
                        unit_key=unit_key,
                        selected=oracle_by_seed[seed],
                        baseline=baseline_by_seed[seed],
                        evidence_wall_time_s=0.0,
                        routing_wall_time_s=0.0,
                    )
                )
                all_router_vs_oracle.append(
                    _MatchedObservation(
                        unit_key=unit_key,
                        selected=selected_by_seed[seed],
                        baseline=oracle_by_seed[seed],
                        evidence_wall_time_s=evidence_time,
                        routing_wall_time_s=routing_time,
                    )
                )

        if not observations:
            raise ValueError(f"No paired target runs are available for {heldout_dataset}")
        router_aggregate = _aggregate_observations(observations, selected=True)
        baseline_aggregate = _aggregate_observations(observations, selected=False)
        folds.append(
            CorrelationFoldReport(
                heldout_dataset=heldout_dataset,
                training_datasets=router.training_datasets,
                episodes=sum(route_counts.values()),
                route_counts=dict(sorted(route_counts.items())),
                router=router_aggregate,
                baseline=baseline_aggregate,
                comparison=_compare_observations(
                    observations,
                    router_aggregate,
                    baseline_aggregate,
                    noninferiority_margin=config.noninferiority_margin,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_confidence_level=config.bootstrap_confidence_level,
                    bootstrap_seed=config.bootstrap_seed + fold_index + 1,
                ),
                frequency_training_examples=router.frequency_model.training_examples,
                frequency_positive_examples=router.frequency_model.positive_examples,
                channel_training_examples=router.channel_model.training_examples,
                channel_positive_examples=router.channel_model.positive_examples,
            )
        )
        all_observations.extend(observations)
        all_routes.update(route_counts)

    router_aggregate = _aggregate_observations(all_observations, selected=True)
    baseline_aggregate = _aggregate_observations(all_observations, selected=False)
    fixed_aggregate = _aggregate_observations(all_source_fixed_observations, selected=True)
    oracle_aggregate = _aggregate_observations(all_oracle_observations, selected=True)

    # With only a few episodes per outer fold, fold-wise permutations are often
    # degenerate. Permute the complete held-out route multiset globally instead;
    # this preserves the exact overall action histogram while breaking every
    # episode/evidence association. This remains a descriptive randomization
    # control rather than a sampling-confidence interval.
    random_vs_lfc: list[list[float]] = [
        [] for _ in range(config.random_control_repeats)
    ]
    router_vs_random: list[list[float]] = [
        [] for _ in range(config.random_control_repeats)
    ]
    random_generator = np.random.default_rng(config.random_state + 10_000)
    route_pool = [route for _episode, route in routed_units]
    observed_assignments: set[tuple[str, ...]] = set()
    for repeat in range(config.random_control_repeats):
        permuted = tuple(str(value) for value in random_generator.permutation(route_pool))
        observed_assignments.add(permuted)
        for (episode, router_action), random_action in zip(
            routed_units,
            permuted,
            strict=True,
        ):
            random_mse = _episode_action_mean(episode, random_action, "adapted_loss")
            router_mse = _episode_action_mean(episode, router_action, "adapted_loss")
            lfc_mse = _episode_action_mean(episode, action_ids.full, "adapted_loss")
            random_vs_lfc[repeat].append(_relative_difference(random_mse, lfc_mse))
            router_vs_random[repeat].append(
                _relative_difference(router_mse, random_mse)
            )
    random_vs_lfc_estimates = np.asarray(
        [_mean_finite(values) for values in random_vs_lfc], dtype=np.float64
    )
    router_vs_random_estimates = np.asarray(
        [_mean_finite(values) for values in router_vs_random], dtype=np.float64
    )
    random_low, random_high = _central_interval(
        random_vs_lfc_estimates,
        config.bootstrap_confidence_level,
    )
    router_random_low, router_random_high = _central_interval(
        router_vs_random_estimates,
        config.bootstrap_confidence_level,
    )
    fixed_ci_low, fixed_ci_high = _paired_cluster_bootstrap_interval(
        all_router_vs_source_fixed,
        "adapted_loss",
        samples=config.bootstrap_samples,
        confidence_level=config.bootstrap_confidence_level,
        seed=config.bootstrap_seed + 1_000,
    )
    controls = CorrelationControlReport(
        source_fixed=SourceFixedControl(
            actions_by_heldout_dataset=dict(sorted(source_fixed_actions.items())),
            route_counts=dict(sorted(source_fixed_routes.items())),
            aggregate=fixed_aggregate,
            relative_mse_difference_vs_lfc=float(
                _mean_unit_relative_difference(
                    all_source_fixed_observations,
                    "adapted_loss",
                )
            ),
            router_relative_mse_difference_vs_control=float(
                _mean_unit_relative_difference(
                    all_router_vs_source_fixed,
                    "adapted_loss",
                )
            ),
            router_relative_mse_difference_vs_control_ci_low=fixed_ci_low,
            router_relative_mse_difference_vs_control_ci_high=fixed_ci_high,
        ),
        random_histogram_matched=RandomMatchedControl(
            repeats=config.random_control_repeats,
            assignment_scope="global-heldout-units",
            route_counts=dict(sorted(all_routes.items())),
            distinct_assignments_observed=len(observed_assignments),
            total_possible_assignments=_multiset_permutation_count(route_pool),
            descriptive_only=True,
            relative_mse_difference_vs_lfc_mean=float(random_vs_lfc_estimates.mean()),
            relative_mse_difference_vs_lfc_randomization_low=random_low,
            relative_mse_difference_vs_lfc_randomization_high=random_high,
            router_relative_mse_difference_vs_control_mean=float(
                router_vs_random_estimates.mean()
            ),
            router_relative_mse_difference_vs_control_randomization_low=router_random_low,
            router_relative_mse_difference_vs_control_randomization_high=router_random_high,
        ),
        oracle=OracleControl(
            route_counts=dict(sorted(oracle_routes.items())),
            aggregate=oracle_aggregate,
            relative_mse_difference_vs_lfc=float(
                _mean_unit_relative_difference(all_oracle_observations, "adapted_loss")
            ),
            router_relative_mse_regret=float(
                _mean_unit_relative_difference(all_router_vs_oracle, "adapted_loss")
            ),
        ),
    )
    return CorrelationBenchmarkReport(
        folds=tuple(folds),
        episodes=sum(all_routes.values()),
        route_counts=dict(sorted(all_routes.items())),
        router=router_aggregate,
        baseline=baseline_aggregate,
        comparison=_compare_observations(
            all_observations,
            router_aggregate,
            baseline_aggregate,
            noninferiority_margin=config.noninferiority_margin,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_confidence_level=config.bootstrap_confidence_level,
            bootstrap_seed=config.bootstrap_seed,
        ),
        action_ids=action_ids,
        probability_threshold=config.probability_threshold,
        min_relative_benefit=config.min_relative_benefit,
        noninferiority_margin=config.noninferiority_margin,
        controls=controls,
        one_class_folds=tuple(sorted(one_class_folds)),
        unit_table=tuple(unit_table),
    )


def infer_router_action_ids(records: Sequence[UtilityRecord]) -> RouterActionIds:
    """Recognize either descriptive IDs or the MVP's A2--A5 aliases."""

    available = {str(_record_value(record, "action_id")) for record in records}
    descriptive = RouterActionIds()
    if set(descriptive.all) <= available:
        return descriptive
    mvp = RouterActionIds(base="A2", frequency="A3", channel="A4", full="A5")
    if set(mvp.all) <= available:
        return mvp
    raise ValueError("Records must contain L/LF/LC/LFC (or the compatible A2/A3/A4/A5 aliases)")


def paper_time_peft_parameter_savings(
    hidden_size: int,
    channels: int,
    *,
    frequency_activation_rate: float,
    channel_activation_rate: float,
    fixed_trainable_parameters: int = 0,
    frequency_hidden_size: int | None = None,
    channel_rank: int | None = None,
    action_ids: RouterActionIds | None = None,
    count_inferred: bool = False,
    optional_activation_rate: float | None = None,
) -> PaperParameterSavings:
    """Apply the Time-PEFT paper adapter parameter formulas.

    ``P_F = h1*h2`` and ``P_C = r*(h1+h2) + C*r*h1``.  The default
    ``h2=h1`` and ``r=h1/2`` match the paper. ``count_inferred=True`` adds the
    projection biases and affine output LayerNorm used by the reproduction
    variant. Fixed parameters are the head and LoRA terms that remain active for
    every route. This computes *active trainable* parameters; storing all four
    route options can require more.
    """

    if hidden_size <= 0 or channels <= 0:
        raise ValueError("hidden_size and channels must be positive")
    if fixed_trainable_parameters < 0:
        raise ValueError("fixed_trainable_parameters must be non-negative")
    if not 0.0 <= frequency_activation_rate <= 1.0:
        raise ValueError("frequency_activation_rate must lie in [0, 1]")
    if not 0.0 <= channel_activation_rate <= 1.0:
        raise ValueError("channel_activation_rate must lie in [0, 1]")
    if optional_activation_rate is not None and not 0.0 <= optional_activation_rate <= 1.0:
        raise ValueError("optional_activation_rate must lie in [0, 1]")
    h2 = frequency_hidden_size if frequency_hidden_size is not None else hidden_size
    rank = channel_rank if channel_rank is not None else hidden_size // 2
    if h2 <= 0 or rank <= 0:
        raise ValueError("frequency_hidden_size and channel_rank must be positive")

    frequency_parameters = hidden_size * h2 + (h2 if count_inferred else 0)
    channel_parameters = (
        rank * (hidden_size + h2)
        + channels * rank * hidden_size
        + (rank + channels * hidden_size if count_inferred else 0)
    )
    output_norm_parameters = 2 * hidden_size if count_inferred else 0
    if optional_activation_rate is None:
        # It has no effect in the bias-free/non-affine variant. Count-inferred
        # callers should pass the observed union P(F or C), which is identifiable
        # from route counts but not from the two marginal activation rates alone.
        optional_activation_rate = 0.0
    always_on = (
        fixed_trainable_parameters
        + frequency_parameters
        + channel_parameters
        + output_norm_parameters
    )
    expected = (
        fixed_trainable_parameters
        + frequency_activation_rate * frequency_parameters
        + channel_activation_rate * channel_parameters
        + optional_activation_rate * output_norm_parameters
    )
    saved = always_on - expected
    routes = action_ids or RouterActionIds()
    active_by_route = {
        routes.base: fixed_trainable_parameters,
        routes.frequency: (
            fixed_trainable_parameters + frequency_parameters + output_norm_parameters
        ),
        routes.channel: (
            fixed_trainable_parameters + channel_parameters + output_norm_parameters
        ),
        routes.full: always_on,
    }
    return PaperParameterSavings(
        hidden_size=hidden_size,
        frequency_hidden_size=h2,
        channel_rank=rank,
        channels=channels,
        fixed_trainable_parameters=fixed_trainable_parameters,
        frequency_adapter_parameters=frequency_parameters,
        channel_adapter_parameters=channel_parameters,
        output_norm_parameters=output_norm_parameters,
        always_on_trainable_parameters=always_on,
        expected_router_trainable_parameters=expected,
        expected_saved_parameters=saved,
        expected_reduction_fraction=saved / always_on if always_on else 0.0,
        active_parameters_by_route=active_by_route,
    )


time_peft_parameter_savings = paper_time_peft_parameter_savings


def _fit_binary_gate(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: tuple[str, ...],
    config: CorrelationRouterConfig,
) -> BinaryRoutingModel:
    if not rows:
        raise ValueError("At least one source episode is required to fit a routing gate")
    target = np.asarray(labels, dtype=np.int64)
    positives = int(target.sum())
    unique = np.unique(target)
    if unique.size == 1:
        return BinaryRoutingModel(
            feature_names=feature_names,
            estimator=None,
            constant_probability=float(unique[0]),
            training_examples=len(rows),
            positive_examples=positives,
        )

    features = np.asarray(
        [[_numeric_or_nan(row.get(name)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    estimator = Pipeline(
        steps=(
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=config.regularization_c,
                    class_weight="balanced",
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    solver="liblinear",
                ),
            ),
        )
    )
    estimator.fit(features, target)
    return BinaryRoutingModel(
        feature_names=feature_names,
        estimator=estimator,
        constant_probability=None,
        training_examples=len(rows),
        positive_examples=positives,
    )


def _complete_episodes(
    records: Sequence[UtilityRecord],
    action_ids: RouterActionIds,
    *,
    expected_seeds: Sequence[int] | None = None,
) -> list[_EpisodeActions]:
    grouped: dict[tuple[Any, ...], dict[str, list[Any]]] = {}
    for record in records:
        action_id = str(_record_value(record, "action_id"))
        if action_id not in action_ids.all:
            continue
        key = (
            str(_record_value(record, "dataset")),
            str(_record_value(record, "episode_id")),
            int(_record_value(record, "horizon")),
            str(_record_value(record, "config_hash", default="")),
            str(_record_value(record, "model_revision", default="")),
            str(_record_value(record, "preprocessing_hash", default="")),
        )
        grouped.setdefault(key, {}).setdefault(action_id, []).append(record)

    output: list[_EpisodeActions] = []
    for key, raw_actions in sorted(grouped.items(), key=lambda item: item[0]):
        episode_label = _episode_label(key)
        require_exact_seed_pairing(
            raw_actions,
            action_ids.all,
            episode_label=episode_label,
            seed_getter=lambda record: int(_record_value(record, "seed")),
        )
        actions = {
            action_id: [
                record
                for record in raw_actions[action_id]
                if str(_record_value(record, "status", default="ok")) == "ok"
                and math.isfinite(_record_float(record, "adapted_loss"))
            ]
            for action_id in action_ids.all
        }
        seeds = require_exact_seed_pairing(
            actions,
            action_ids.all,
            episode_label=f"{episode_label} (successful finite records)",
            seed_getter=lambda record: int(_record_value(record, "seed")),
            expected_seeds=expected_seeds,
        )
        base_rows = actions[action_ids.base]
        evidence = _mean_evidence(base_rows)
        output.append(
            _EpisodeActions(
                dataset=str(key[0]),
                episode_id=str(key[1]),
                horizon=int(key[2]),
                evidence=evidence,
                seeds=seeds,
                records={
                    action_id: tuple(
                        sorted(
                            actions[action_id],
                            key=lambda record: int(_record_value(record, "seed")),
                        )
                    )
                    for action_id in action_ids.all
                },
            )
        )
    if not output:
        raise ValueError("No episodes contain a complete matched action set")
    return output


def _episode_label(key: tuple[Any, ...]) -> str:
    dataset, episode_id, horizon, config_hash, model_revision, preprocessing_hash = key
    return (
        f"{dataset}/{episode_id} (horizon={horizon}, config_hash={config_hash!r}, "
        f"model_revision={model_revision!r}, preprocessing_hash={preprocessing_hash!r})"
    )


def _mean_evidence(records: Sequence[Any]) -> dict[str, float]:
    mappings = [_evidence_mapping(_record_value(record, "evidence")) for record in records]
    names = sorted({name for mapping in mappings for name in mapping})
    output: dict[str, float] = {}
    for name in names:
        values = [_numeric_or_nan(mapping.get(name)) for mapping in mappings]
        finite = [value for value in values if math.isfinite(value)]
        output[name] = float(np.mean(finite)) if finite else 0.0
    return output


def _aggregate_observations(
    observations: Sequence[_MatchedObservation], *, selected: bool
) -> BenchmarkAggregate:
    records = [row.selected if selected else row.baseline for row in observations]
    evidence_times = [row.evidence_wall_time_s for row in observations] if selected else [0.0]
    routing_times = [row.routing_wall_time_s for row in observations] if selected else [0.0]
    adaptation_times = [_record_float(record, "wall_time_s", default=0.0) for record in records]
    mean_adaptation = _mean_finite(adaptation_times)
    mean_evidence = _mean_finite(evidence_times)
    mean_routing = _mean_finite(routing_times)
    maes = [_record_optional_float(record, "adapted_mae") for record in records]
    finite_maes = [value for value in maes if value is not None and math.isfinite(value)]
    return BenchmarkAggregate(
        runs=len(records),
        mse=_mean_finite([_record_float(record, "adapted_loss") for record in records]),
        mae=float(np.mean(finite_maes)) if finite_maes else None,
        adaptation_wall_time_s=mean_adaptation,
        evidence_wall_time_s=mean_evidence,
        routing_wall_time_s=mean_routing,
        end_to_end_wall_time_s=mean_adaptation + mean_evidence + mean_routing,
        trainable_parameters=_mean_finite(
            [_record_float(record, "trainable_parameters") for record in records]
        ),
        stored_adapter_parameters=_mean_finite(
            [_record_float(record, "stored_adapter_parameters") for record in records]
        ),
        peak_memory_mb=_mean_finite(
            [_record_float(record, "peak_memory_mb") for record in records]
        ),
        profiled_flops=_mean_finite(
            [_record_float(record, "profiled_flops") for record in records]
        ),
    )


def _compare_observations(
    observations: Sequence[_MatchedObservation],
    router: BenchmarkAggregate,
    baseline: BenchmarkAggregate,
    *,
    noninferiority_margin: float,
    bootstrap_samples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BenchmarkComparison:
    """Compare accuracy by matched unit and costs by descriptive run aggregates.

    Accuracy follows the preregistered estimand: average paired seeds inside each
    dataset/horizon/episode unit, compute that unit's relative difference, then
    give every unit equal weight.  The aggregate MSE/MAE fields remain useful for
    descriptive display but are deliberately not used for the accuracy decision.
    """

    relative_mse = _mean_unit_relative_difference(observations, "adapted_loss")
    if relative_mse is None:
        raise ValueError("MSE comparison unexpectedly produced no paired value")
    relative_mse_ci_low, relative_mse_ci_high = _paired_cluster_bootstrap_interval(
        observations,
        "adapted_loss",
        samples=bootstrap_samples,
        confidence_level=bootstrap_confidence_level,
        seed=bootstrap_seed,
    )
    relative_mae = _mean_unit_relative_difference(
        observations,
        "adapted_mae",
        optional=True,
    )
    return BenchmarkComparison(
        relative_mse_difference=relative_mse,
        relative_mse_ci_low=relative_mse_ci_low,
        relative_mse_ci_high=relative_mse_ci_high,
        relative_mae_difference=relative_mae,
        end_to_end_time_reduction_fraction=_reduction(
            router.end_to_end_wall_time_s,
            baseline.end_to_end_wall_time_s,
        ),
        trainable_parameter_reduction_fraction=_reduction(
            router.trainable_parameters,
            baseline.trainable_parameters,
        ),
        stored_parameter_reduction_fraction=_reduction(
            router.stored_adapter_parameters,
            baseline.stored_adapter_parameters,
        ),
        peak_memory_reduction_fraction=_reduction(
            router.peak_memory_mb,
            baseline.peak_memory_mb,
        ),
        flops_reduction_fraction=_reduction(router.profiled_flops, baseline.profiled_flops),
        noninferiority_margin=noninferiority_margin,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        accuracy_superior=relative_mse_ci_high < 0.0,
        point_noninferior_within_margin=relative_mse <= noninferiority_margin,
        noninferior_within_margin=relative_mse_ci_high < noninferiority_margin,
    )


def _mean_unit_relative_difference(
    observations: Sequence[_MatchedObservation],
    metric: str,
    *,
    optional: bool = False,
) -> float | None:
    unit_differences = _unit_relative_differences(
        observations,
        metric,
        optional=optional,
    )
    if not unit_differences:
        if optional:
            return None
        raise ValueError(f"No finite paired unit values are available for {metric}")
    return _mean_finite(list(unit_differences.values()))


def _unit_relative_differences(
    observations: Sequence[_MatchedObservation],
    metric: str,
    *,
    optional: bool = False,
) -> dict[tuple[str, int, str], float]:
    """Average seeds first and return one paired relative value per unit."""

    grouped: dict[tuple[str, int, str], list[_MatchedObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.unit_key, []).append(observation)

    unit_differences: dict[tuple[str, int, str], float] = {}
    for unit_key, unit_observations in grouped.items():
        selected_values: list[float] = []
        baseline_values: list[float] = []
        for observation in unit_observations:
            if optional:
                selected_value = _record_optional_float(observation.selected, metric)
                baseline_value = _record_optional_float(observation.baseline, metric)
                if selected_value is None or baseline_value is None:
                    continue
            else:
                selected_value = _record_float(observation.selected, metric)
                baseline_value = _record_float(observation.baseline, metric)
            if not math.isfinite(selected_value) or not math.isfinite(baseline_value):
                continue
            selected_values.append(selected_value)
            baseline_values.append(baseline_value)
        if selected_values:
            unit_differences[unit_key] = _relative_difference(
                _mean_finite(selected_values),
                _mean_finite(baseline_values),
            )
    return unit_differences


def _paired_cluster_bootstrap_interval(
    observations: Sequence[_MatchedObservation],
    metric: str,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Resample datasets, then matched units, after seed averaging.

    The target Cartesian protocol gives every dataset the same number of units.
    Resampling whole dataset clusters before their units preserves cross-unit
    dependence and avoids pretending that adaptation seeds are independent units.
    """

    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence_level must lie in (0, 1)")
    values = _unit_relative_differences(observations, metric)
    if not values:
        raise ValueError(f"No finite paired unit values are available for {metric}")
    by_dataset: dict[str, list[float]] = {}
    for (dataset, _horizon, _episode_id), value in values.items():
        by_dataset.setdefault(dataset, []).append(value)
    datasets = sorted(by_dataset)
    rng = np.random.default_rng(seed)
    clusters = [np.asarray(by_dataset[dataset], dtype=np.float64) for dataset in datasets]
    cluster_sizes = {cluster.size for cluster in clusters}
    selected_datasets = rng.integers(
        0,
        len(clusters),
        size=(samples, len(clusters)),
    )
    if len(cluster_sizes) == 1:
        # The configured Cartesian benchmark follows this fast path.
        width = next(iter(cluster_sizes))
        values_array = np.stack(clusters)
        selected_units = rng.integers(
            0,
            width,
            size=(samples, len(clusters), width),
        )
        estimates = values_array[
            selected_datasets[..., None],
            selected_units,
        ].mean(axis=(1, 2))
    else:
        # Preserve cluster sizes if an analysis-only legacy store is unbalanced.
        weighted_sums = np.zeros(samples, dtype=np.float64)
        sampled_counts = np.zeros(samples, dtype=np.int64)
        for position in range(len(clusters)):
            chosen = selected_datasets[:, position]
            for cluster_index, cluster in enumerate(clusters):
                rows = np.flatnonzero(chosen == cluster_index)
                if rows.size == 0:
                    continue
                indices = rng.integers(
                    0,
                    cluster.size,
                    size=(rows.size, cluster.size),
                )
                weighted_sums[rows] += cluster[indices].sum(axis=1)
                sampled_counts[rows] += cluster.size
        estimates = weighted_sums / sampled_counts
    return _central_interval(estimates, confidence_level)


def _central_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _multiset_permutation_count(values: Sequence[str]) -> int:
    counts = Counter(values)
    permutations = math.factorial(len(values))
    for count in counts.values():
        permutations //= math.factorial(count)
    return permutations


def _records_by_seed(episode: _EpisodeActions, action_id: str) -> dict[int, Any]:
    return {
        int(_record_value(record, "seed")): record
        for record in episode.records[action_id]
    }


def _episode_action_mean(
    episode: _EpisodeActions,
    action_id: str,
    metric: str,
) -> float:
    return _mean_finite(
        [_record_float(record, metric) for record in episode.records[action_id]]
    )


def _best_source_fixed_action(
    source: Sequence[_EpisodeActions],
    action_ids: RouterActionIds,
) -> str:
    """Choose one fixed arm using source units only and scale-free losses."""

    candidates = action_ids.all
    scores: dict[str, float] = {}
    for action_id in candidates:
        differences = []
        for episode in source:
            candidate = _episode_action_mean(episode, action_id, "adapted_loss")
            full = _episode_action_mean(episode, action_ids.full, "adapted_loss")
            differences.append(_relative_difference(candidate, full))
        scores[action_id] = _mean_finite(differences)
    return min(candidates, key=lambda action_id: (scores[action_id], action_id))


def _evidence_mapping(evidence: Mapping[str, float] | EvidenceBundle) -> dict[str, float]:
    if isinstance(evidence, EvidenceBundle):
        values = evidence.as_dict()
    elif isinstance(evidence, Mapping):
        values = {str(name): float(value) for name, value in evidence.items()}
    else:
        raise TypeError("Routing evidence must be an EvidenceBundle or mapping")
    forbidden = sorted(name for name in values if name.startswith("query_"))
    if forbidden:
        raise ValueError(f"Router received query-derived evidence: {forbidden}")
    return values


def _record_value(record: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is not None:
        return default
    raise AttributeError(f"Utility record is missing required field {name!r}")


def _record_float(record: Any, name: str, *, default: float | None = None) -> float:
    value = _record_value(record, name, default=default)
    return float(value)


def _record_optional_float(record: Any, name: str) -> float | None:
    try:
        value = _record_value(record, name)
    except AttributeError:
        return None
    return None if value is None else float(value)


def _numeric_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("Metric aggregation has no finite values")
    return float(np.mean(finite))


def _relative_improvement(reference: float, adapted: float) -> float:
    return (reference - adapted) / max(abs(reference), 1e-12)


def _relative_difference(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-12)


def _reduction(value: float, baseline: float) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0
    return 1.0 - value / baseline
