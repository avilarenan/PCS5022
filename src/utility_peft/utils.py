"""Reproducibility and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without silently weakening determinism."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def canonical_json(value: Any) -> str:
    """Return a stable JSON encoding suitable for provenance hashes."""

    return json.dumps(_to_builtin(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any, *, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def preprocessing_hash(settings: Any) -> str:
    return stable_hash({"schema": 1, "preprocessing": settings})


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(_to_builtin(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def environment_metadata() -> dict[str, Any]:
    """Collect enough local state to audit a run without an external tracker."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "implementation_hash": implementation_hash(),
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_mb": torch.cuda.get_device_properties(index).total_memory / 1024**2,
            }
            for index in range(torch.cuda.device_count())
        ]
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
    return metadata


@lru_cache(maxsize=1)
def implementation_hash() -> str:
    """Hash package source so changed evaluator code cannot reuse stale records."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_builtin(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    return value
