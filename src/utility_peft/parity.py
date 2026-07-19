"""Time-PEFT parity provenance and claim-label safeguards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from utility_peft.utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class TimePeftParityManifest:
    paper_id: str
    protocol: str
    implementation_label: str
    verified: bool
    verification_note: str
    model_revision: str
    datasets: tuple[str, ...]
    horizons: tuple[int, ...]
    actions: tuple[str, ...]
    seeds: tuple[int, ...]
    generated_records: int
    adapter_placement: str = "MOMENT encoder q/v projections plus pre-head representation adapters"
    frequency_operation: str = "top-magnitude FFT bins over the patch axis"
    channel_operation: str = "residual channel-centered bottleneck mixing"
    parity_boundary: str = (
        "Architecture is Time-PEFT-style until official code or an accepted-paper "
        "specification is checked numerically."
    )


def baseline_label(*, verified: bool, configured_label: str) -> str:
    if verified:
        return "Time-PEFT"
    return configured_label if "style" in configured_label.lower() else "Time-PEFT-style"


def write_parity_manifest(path: str | Path, manifest: TimePeftParityManifest) -> None:
    atomic_write_json(path, asdict(manifest))
