"""Utility-PEFT research package."""

import os

# cuBLAS reads this before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from utility_peft.types import (
    ActionSpec,
    Budget,
    EpisodeManifest,
    EvaluationEpisode,
    EvidenceBundle,
    SupportView,
    UtilityRecord,
)

__all__ = [
    "ActionSpec",
    "Budget",
    "EpisodeManifest",
    "EvaluationEpisode",
    "EvidenceBundle",
    "SupportView",
    "UtilityRecord",
]

__version__ = "0.1.0"
