"""Action-conditioned heteroscedastic utility controller."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utility_peft.costs import ActionCost, CostTable, filter_feasible
from utility_peft.evidence import FEATURE_SETS, select_feature_mapping
from utility_peft.types import Budget, EvidenceBundle, UtilityRecord
from utility_peft.utils import atomic_write_json, seed_everything


@dataclass(frozen=True, slots=True)
class ControllerTrainingConfig:
    hidden_size: int = 256
    action_embedding_size: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ranking_weight: float = 0.2
    epochs: int = 200
    validation_dataset: str | None = None
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ControllerMetrics:
    best_epoch: int
    validation_ndcg: float
    training_rows: int
    validation_rows: int


@dataclass(frozen=True, slots=True)
class _ControllerRow:
    episode_id: str
    dataset: str
    action_id: str
    evidence: Mapping[str, float]
    gain: float


class UtilityController(nn.Module):
    def __init__(
        self,
        feature_count: int,
        action_count: int,
        *,
        hidden_size: int = 256,
        action_embedding_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, action_embedding_size)
        self.network = nn.Sequential(
            nn.Linear(feature_count + action_embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: Tensor, action_indices: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(torch.cat((features, self.action_embedding(action_indices)), dim=-1))
        return output[:, 0], output[:, 1].clamp(-10, 10)


class ControllerBundle:
    """Serializable controller plus source-only normalization and costs."""

    def __init__(
        self,
        model: UtilityController,
        *,
        feature_names: tuple[str, ...],
        feature_mean: Tensor,
        feature_std: Tensor,
        action_ids: tuple[str, ...],
        costs: CostTable,
        feature_set: str = "full",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.feature_mean = feature_mean.float().cpu()
        self.feature_std = feature_std.float().cpu()
        self.action_ids = action_ids
        self.costs = costs
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown controller feature set: {feature_set}")
        self.feature_set = feature_set

    @torch.no_grad()
    def predict(self, evidence: EvidenceBundle) -> dict[str, tuple[float, float]]:
        raw = evidence.as_dict()
        forbidden = sorted(name for name in raw if name.startswith("query_"))
        if forbidden:
            raise ValueError(
                f"Controller received unknown evidence derived from query data: {forbidden}"
            )
        selected = select_feature_mapping(raw, self.feature_set)
        expected_input = set(raw) if self.feature_set == "full" else set(selected)
        if expected_input - set(self.feature_names):
            unknown = sorted(expected_input - set(self.feature_names))
            raise ValueError(f"Controller received unknown evidence features: {unknown}")
        feature = torch.tensor(
            [selected.get(name, 0.0) for name in self.feature_names],
            dtype=torch.float32,
        )
        feature = (feature - self.feature_mean) / self.feature_std
        features = feature.repeat(len(self.action_ids), 1)
        action_indices = torch.arange(len(self.action_ids))
        self.model.eval()
        mean, log_variance = self.model(features, action_indices)
        return {
            action_id: (float(mean[index]), float(log_variance[index].exp().sqrt()))
            for index, action_id in enumerate(self.action_ids)
        }

    def select(
        self,
        evidence: EvidenceBundle,
        *,
        budget: Budget | None = None,
        cost_weights: Mapping[str, float] | None = None,
    ) -> str:
        predictions = self.predict(evidence)
        feasible = filter_feasible(list(self.action_ids), self.costs, budget or Budget())
        if not feasible:
            raise RuntimeError("No controller action satisfies the hard budget")
        return max(
            feasible,
            key=lambda action_id: self.costs.utility(
                action_id, predictions[action_id][0], cost_weights
            ),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cost_rows = {action_id: asdict(cost) for action_id, cost in self.costs.costs.items()}
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_names": self.feature_names,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "action_ids": self.action_ids,
                "costs": cost_rows,
                "feature_set": self.feature_set,
                "architecture": {
                    "hidden_size": self.model.network[0].out_features,
                    "action_embedding_size": self.model.action_embedding.embedding_dim,
                    "dropout": self.model.network[2].p,
                },
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> ControllerBundle:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        architecture = payload["architecture"]
        model = UtilityController(
            len(payload["feature_names"]),
            len(payload["action_ids"]),
            **architecture,
        )
        model.load_state_dict(payload["state_dict"])
        costs = CostTable(
            {action_id: ActionCost(**values) for action_id, values in payload["costs"].items()}
        )
        return cls(
            model,
            feature_names=tuple(payload["feature_names"]),
            feature_mean=payload["feature_mean"],
            feature_std=payload["feature_std"],
            action_ids=tuple(payload["action_ids"]),
            costs=costs,
            feature_set=str(payload.get("feature_set", "full")),
        )


def train_controller(
    records: list[UtilityRecord],
    output_path: str | Path,
    *,
    config: ControllerTrainingConfig | None = None,
    feature_set: str = "full",
) -> tuple[ControllerBundle, ControllerMetrics]:
    config = config or ControllerTrainingConfig()
    records = _records_for_feature_set(records, feature_set)
    rows = _aggregate_rows(records)
    if not rows:
        raise ValueError("No successful utility records are available")
    datasets = sorted({row.dataset for row in rows})
    if len(datasets) > 1:
        validation_dataset = config.validation_dataset or datasets[-1]
        training_rows = [row for row in rows if row.dataset != validation_dataset]
        validation_rows = [row for row in rows if row.dataset == validation_dataset]
    else:
        episode_ids = sorted({row.episode_id for row in rows})
        split = max(1, int(len(episode_ids) * 0.8))
        train_ids = set(episode_ids[:split])
        training_rows = [row for row in rows if row.episode_id in train_ids]
        validation_rows = [row for row in rows if row.episode_id not in train_ids]
        if not validation_rows:
            validation_rows = training_rows

    feature_names = tuple(sorted({name for row in rows for name in row.evidence}))
    action_ids = tuple(sorted({row.action_id for row in rows}))
    training_features = _feature_tensor(training_rows, feature_names)
    mean = training_features.mean(dim=0)
    std = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    best_epoch, validation_ndcg = _select_epoch(
        training_rows,
        validation_rows,
        feature_names,
        action_ids,
        mean,
        std,
        config,
    )

    all_features = _feature_tensor(rows, feature_names)
    final_mean = all_features.mean(dim=0)
    final_std = all_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    final_model = _fit(
        rows,
        feature_names,
        action_ids,
        final_mean,
        final_std,
        config,
        epochs=best_epoch,
    ).cpu()
    bundle = ControllerBundle(
        final_model,
        feature_names=feature_names,
        feature_mean=final_mean,
        feature_std=final_std,
        action_ids=action_ids,
        costs=CostTable.from_records(records),
        feature_set=feature_set,
    )
    bundle.save(output_path)
    metrics = ControllerMetrics(
        best_epoch=best_epoch,
        validation_ndcg=validation_ndcg,
        training_rows=len(training_rows),
        validation_rows=len(validation_rows),
    )
    atomic_write_json(Path(output_path).with_suffix(".metrics.json"), asdict(metrics))
    return bundle, metrics


def train_controller_nested(
    records: list[UtilityRecord],
    output_path: str | Path,
    *,
    config: ControllerTrainingConfig | None = None,
    feature_set: str = "full",
) -> tuple[ControllerBundle, ControllerMetrics]:
    """Select epochs by leave-one-source-dataset-out NDCG, then refit all sources."""

    config = config or ControllerTrainingConfig()
    filtered_records = _records_for_feature_set(records, feature_set)
    rows = _aggregate_rows(filtered_records)
    datasets = sorted({row.dataset for row in rows})
    if len(datasets) < 2:
        return train_controller(
            records,
            output_path,
            config=config,
            feature_set=feature_set,
        )
    feature_names = tuple(sorted({name for row in rows for name in row.evidence}))
    action_ids = tuple(sorted({row.action_id for row in rows}))
    epochs: list[int] = []
    scores: list[float] = []
    for dataset in datasets:
        training_rows = [row for row in rows if row.dataset != dataset]
        validation_rows = [row for row in rows if row.dataset == dataset]
        training_features = _feature_tensor(training_rows, feature_names)
        mean = training_features.mean(dim=0)
        std = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
        epoch, score = _select_epoch(
            training_rows,
            validation_rows,
            feature_names,
            action_ids,
            mean,
            std,
            replace(config, seed=config.seed + len(epochs)),
        )
        epochs.append(epoch)
        scores.append(score)

    selected_epoch = max(1, int(round(median(epochs))))
    all_features = _feature_tensor(rows, feature_names)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    model = _fit(
        rows,
        feature_names,
        action_ids,
        mean,
        std,
        config,
        epochs=selected_epoch,
    ).cpu()
    bundle = ControllerBundle(
        model,
        feature_names=feature_names,
        feature_mean=mean,
        feature_std=std,
        action_ids=action_ids,
        costs=CostTable.from_records(filtered_records),
        feature_set=feature_set,
    )
    bundle.save(output_path)
    metrics = ControllerMetrics(
        best_epoch=selected_epoch,
        validation_ndcg=float(np.mean(scores)),
        training_rows=len(rows),
        validation_rows=len(rows),
    )
    atomic_write_json(Path(output_path).with_suffix(".metrics.json"), asdict(metrics))
    return bundle, metrics


def ndcg_by_episode(
    model: UtilityController,
    rows: list[_ControllerRow],
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    mean: Tensor,
    std: Tensor,
    device: torch.device,
) -> float:
    if not rows:
        return 0.0
    model.eval()
    predictions = _predict_rows(model, rows, feature_names, action_ids, mean, std, device)
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        grouped.setdefault(row.episode_id, []).append((row.gain, prediction))
    return float(np.mean([_ndcg(values) for values in grouped.values()]))


def _select_epoch(
    training_rows: list[_ControllerRow],
    validation_rows: list[_ControllerRow],
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    mean: Tensor,
    std: Tensor,
    config: ControllerTrainingConfig,
) -> tuple[int, float]:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = UtilityController(
        len(feature_names),
        len(action_ids),
        hidden_size=config.hidden_size,
        action_embedding_size=config.action_embedding_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_ndcg = -math.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        _training_step(
            model,
            optimizer,
            training_rows,
            feature_names,
            action_ids,
            mean,
            std,
            config.ranking_weight,
            device,
        )
        score = ndcg_by_episode(
            model,
            validation_rows,
            feature_names,
            action_ids,
            mean,
            std,
            device,
        )
        if score > best_ndcg:
            best_ndcg = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_epoch, float(best_ndcg)


def _fit(
    rows: list[_ControllerRow],
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    mean: Tensor,
    std: Tensor,
    config: ControllerTrainingConfig,
    *,
    epochs: int,
) -> UtilityController:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = UtilityController(
        len(feature_names),
        len(action_ids),
        hidden_size=config.hidden_size,
        action_embedding_size=config.action_embedding_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(max(epochs, 1)):
        _training_step(
            model,
            optimizer,
            rows,
            feature_names,
            action_ids,
            mean,
            std,
            config.ranking_weight,
            device,
        )
    return model


def _training_step(
    model: UtilityController,
    optimizer: torch.optim.Optimizer,
    rows: list[_ControllerRow],
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    mean: Tensor,
    std: Tensor,
    ranking_weight: float,
    device: torch.device,
) -> None:
    model.train()
    features = ((_feature_tensor(rows, feature_names) - mean) / std).to(device)
    action_lookup = {action_id: index for index, action_id in enumerate(action_ids)}
    actions = torch.tensor(
        [action_lookup[row.action_id] for row in rows], dtype=torch.long, device=device
    )
    targets = torch.tensor([row.gain for row in rows], dtype=torch.float32, device=device)
    predicted, log_variance = model(features, actions)
    regression = (
        0.5 * (torch.exp(-log_variance) * (targets - predicted).square() + log_variance).mean()
    )
    ranking = _pairwise_ranking_loss(predicted, targets, rows)
    loss = regression + ranking_weight * ranking
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _pairwise_ranking_loss(
    predictions: Tensor, targets: Tensor, rows: list[_ControllerRow]
) -> Tensor:
    losses: list[Tensor] = []
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.episode_id, []).append(index)
    for indices in grouped.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                difference = targets[left] - targets[right]
                if float(difference.abs()) < 1e-12:
                    continue
                sign = difference.sign()
                losses.append(F.softplus(-sign * (predictions[left] - predictions[right])))
    return torch.stack(losses).mean() if losses else predictions.sum() * 0


@torch.no_grad()
def _predict_rows(
    model: UtilityController,
    rows: list[_ControllerRow],
    feature_names: tuple[str, ...],
    action_ids: tuple[str, ...],
    mean: Tensor,
    std: Tensor,
    device: torch.device,
) -> list[float]:
    features = ((_feature_tensor(rows, feature_names) - mean) / std).to(device)
    action_lookup = {action_id: index for index, action_id in enumerate(action_ids)}
    actions = torch.tensor(
        [action_lookup[row.action_id] for row in rows], dtype=torch.long, device=device
    )
    predicted, _ = model(features, actions)
    return predicted.cpu().tolist()


def _records_for_feature_set(
    records: list[UtilityRecord], feature_set: str
) -> list[UtilityRecord]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown evidence feature set: {feature_set}")
    if feature_set == "full":
        return records
    return [
        replace(
            record,
            evidence=select_feature_mapping(record.evidence, feature_set),
        )
        for record in records
    ]


def _aggregate_rows(records: list[UtilityRecord]) -> list[_ControllerRow]:
    grouped: dict[tuple[str, str], list[UtilityRecord]] = {}
    for record in records:
        if record.status == "ok" and math.isfinite(record.normalized_gain):
            grouped.setdefault((record.episode_id, record.action_id), []).append(record)
    rows: list[_ControllerRow] = []
    for (_episode_id, _action_id), values in sorted(grouped.items()):
        first = values[0]
        rows.append(
            _ControllerRow(
                episode_id=first.episode_id,
                dataset=first.dataset,
                action_id=first.action_id,
                evidence=dict(first.evidence),
                gain=float(np.mean([record.normalized_gain for record in values])),
            )
        )
    return rows


def _feature_tensor(rows: list[_ControllerRow], feature_names: tuple[str, ...]) -> Tensor:
    return torch.tensor(
        [
            [float(row.evidence.get(feature_name, 0.0)) for feature_name in feature_names]
            for row in rows
        ],
        dtype=torch.float32,
    )


def _ndcg(values: list[tuple[float, float]]) -> float:
    actual = np.asarray([value[0] for value in values], dtype=np.float64)
    predicted = np.asarray([value[1] for value in values], dtype=np.float64)
    relevance = actual - actual.min()
    if np.allclose(relevance, 0):
        return 1.0
    order = np.argsort(-predicted)
    ideal = np.argsort(-relevance)
    discounts = np.log2(np.arange(2, relevance.size + 2))
    dcg = np.sum((2 ** relevance[order] - 1) / discounts)
    idcg = np.sum((2 ** relevance[ideal] - 1) / discounts)
    return float(dcg / idcg)
