"""Generate compact Markdown and CSV tables from immutable run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utility_peft.store import UtilityStore


def build_report(
    store: UtilityStore,
    output_dir: str | Path,
    *,
    oracle_gate_path: str | Path | None = None,
    lodo_metrics_path: str | Path | None = None,
    parity_manifest_path: str | Path | None = None,
    episode_ids: set[str] | None = None,
    config_hash: str | None = None,
    model_revision: str | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = store.records(
        episode_ids=episode_ids,
        config_hash=config_hash,
        model_revision=model_revision,
    )
    if not records:
        raise ValueError("No utility records are available for reporting")
    frame = pd.DataFrame([record.to_flat_dict() for record in records])
    frame = frame.drop(columns=["evidence"])
    summary = (
        frame.groupby(["dataset", "horizon", "action_id"], as_index=False)
        .agg(
            runs=("status", "size"),
            successful=("status", lambda values: int((values == "ok").sum())),
            frozen_loss=("frozen_loss", "mean"),
            adapted_loss=("adapted_loss", "mean"),
            frozen_mae=("frozen_mae", "mean"),
            adapted_mae=("adapted_mae", "mean"),
            normalized_gain=("normalized_gain", "mean"),
            trainable_parameters=("trainable_parameters", "median"),
            profiled_flops=("profiled_flops", "median"),
            peak_memory_mb=("peak_memory_mb", "median"),
            wall_time_s=("wall_time_s", "median"),
            evidence_wall_time_s=("evidence_wall_time_s", "median"),
        )
        .sort_values(["dataset", "horizon", "action_id"])
    )
    summary_path = output / "utility_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8g")

    parity = None
    if parity_manifest_path and Path(parity_manifest_path).exists():
        with Path(parity_manifest_path).open(encoding="utf-8") as handle:
            parity = json.load(handle)
    baseline_label = (
        parity.get("implementation_label", "Time-PEFT-style")
        if parity
        else "Time-PEFT-style"
    )
    lines = [
        "# Utility-PEFT MVP Report",
        "",
        f"Immutable utility records: {len(records)}",
        f"Matched baseline label: {baseline_label}",
        "",
        "## Utility summary",
        "",
        summary.to_markdown(index=False, floatfmt=".5g"),
    ]
    if oracle_gate_path and Path(oracle_gate_path).exists():
        with Path(oracle_gate_path).open(encoding="utf-8") as handle:
            gate = json.load(handle)
        lines.extend(
            [
                "",
                "## Oracle gate",
                "",
                f"- Screen passed: {gate.get('screen_passed', gate['passed'])}",
                f"- Confirmation ready: {gate.get('confirmation_ready', False)}",
                f"- Passed: {gate['passed']}",
                f"- Winning adapter families: {', '.join(gate['winning_families'])}",
                f"- Best fixed action: {gate['best_fixed_action']}",
                (
                    "- Mean paired oracle regret: "
                    f"{gate['mean_oracle_regret']:.6g} "
                    f"(95% bootstrap CI {gate['bootstrap_ci_low']:.6g}, "
                    f"{gate['bootstrap_ci_high']:.6g})"
                ),
            ]
        )
    if lodo_metrics_path and Path(lodo_metrics_path).exists():
        with Path(lodo_metrics_path).open(encoding="utf-8") as handle:
            lodo = json.load(handle)
        lines.extend(
            [
                "",
                "## Leave-one-dataset-out",
                "",
                f"- Mean controller NDCG: {lodo['mean_controller_ndcg']:.6g}",
                f"- Mean random NDCG: {lodo['mean_random_ndcg']:.6g}",
                f"- Mean complexity-only NDCG: {lodo['mean_complexity_ndcg']:.6g}",
                (
                    "- Mean controller oracle regret: "
                    f"{lodo['mean_controller_oracle_regret']:.6g}"
                ),
                (
                    "- Mean source-fixed oracle regret: "
                    f"{lodo['mean_source_fixed_oracle_regret']:.6g}"
                ),
                (
                    f"- Mean {baseline_label} oracle regret: "
                    f"{lodo['mean_time_peft_oracle_regret']:.6g}"
                ),
                (
                    "- Full-minus-complexity NDCG: "
                    f"{lodo['evidence_comparison']['mean_ndcg_improvement']:.6g} "
                    f"(95% CI {lodo['evidence_comparison']['ndcg_improvement_ci_low']:.6g}, "
                    f"{lodo['evidence_comparison']['ndcg_improvement_ci_high']:.6g})"
                ),
                (
                    f"- Relative MSE versus {baseline_label}: "
                    f"{lodo['superiority']['mean_relative_mse_difference']:.6g} "
                    f"(95% CI {lodo['superiority']['relative_mse_ci_low']:.6g}, "
                    f"{lodo['superiority']['relative_mse_ci_high']:.6g})"
                ),
                (
                    "- End-to-end adaptation time reduction: "
                    f"{lodo['superiority']['time_reduction_fraction']:.2%}"
                ),
                f"- H2 model-aware evidence passed: {lodo['hypothesis_h2_passed']}",
                f"- H3 matched-baseline superiority passed: {lodo['hypothesis_h3_passed']}",
                f"- H4 cross-dataset transfer passed: {lodo['hypothesis_h4_passed']}",
            ]
        )
    if parity:
        lines.extend(
            [
                "",
                "## Time-PEFT parity",
                "",
                f"- Implementation label: {baseline_label}",
                f"- Official parity verified: {parity['verified']}",
                f"- Note: {parity['verification_note']}",
            ]
        )
        if not parity["verified"]:
            lines.append(
                "- Claim guard: these results cannot be described as surpassing "
                "the published paper."
            )
    else:
        lines.extend(
            [
                "",
                "## Time-PEFT parity",
                "",
                "- Implementation label: Time-PEFT-style",
                "- Official parity verified: False",
                (
                    "- Claim guard: these results cannot be described as surpassing "
                    "the published paper."
                ),
            ]
        )
    report_path = output / "mvp_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
