from __future__ import annotations

import dataclasses

from utility_peft.actions import ACTION_BY_ID
from utility_peft.controller import ControllerTrainingConfig, train_controller
from utility_peft.episodes import build_episode
from utility_peft.evaluator import TrainingConfig, evaluate_action
from utility_peft.evidence import extract_evidence
from utility_peft.reporting import build_report
from utility_peft.store import UtilityStore
from utility_peft.types import EvidenceBundle


def test_synthetic_utility_store_controller_selection_pipeline(tmp_path, series, template) -> None:
    episodes = [
        build_episode(
            series,
            dataset="integration",
            dataset_family="test",
            lookback=16,
            horizon=8,
            support_size=8,
            query_size=8,
            start=start,
            seed=index,
        )
        for index, start in enumerate((0, 100))
    ]
    store = UtilityStore(tmp_path / "utilities")
    training = TrainingConfig(
        effective_batch_size=4,
        query_batch_size=4,
        bf16=False,
        profile_flops=False,
    )
    actions = (
        ACTION_BY_ID["A0"],
        dataclasses.replace(ACTION_BY_ID["A1"], update_steps=1),
    )
    for episode in episodes:
        evidence = extract_evidence(episode.support, template, include_gradient_probe=False)
        for action in actions:
            result = evaluate_action(
                template,
                episode,
                action,
                evidence,
                seed=0,
                config=training,
                config_hash="integration",
                model_revision="tiny",
                device="cpu",
            )
            assert store.append(result)
            assert not store.append(result)
    records = store.records(statuses={"ok"})
    assert len(records) == 4
    report = build_report(store, tmp_path / "reports")
    report_text = report.read_text(encoding="utf-8")
    assert "Implementation label: Time-PEFT-style" in report_text
    assert "cannot be described as surpassing the published paper" in report_text
    bundle, _ = train_controller(
        records,
        tmp_path / "controller.pt",
        config=ControllerTrainingConfig(
            hidden_size=8,
            action_embedding_size=2,
            dropout=0.0,
            epochs=2,
        ),
    )
    evidence = EvidenceBundle.from_mapping(
        episodes[0].support.manifest.episode_id,
        records[0].evidence,
    )
    assert bundle.select(evidence) in {"A0", "A1"}
