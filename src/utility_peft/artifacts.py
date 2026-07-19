"""Conventional local artifact layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ArtifactLayout:
        return cls(Path(root))

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def episodes(self) -> Path:
        return self.root / "episodes"

    @property
    def utilities(self) -> Path:
        return self.root / "utilities"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def oracle_gate(self) -> Path:
        return self.root / "oracle_gate.json"

    def create(self) -> None:
        for path in (
            self.runs,
            self.episodes,
            self.utilities,
            self.checkpoints,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
