"""Versioned dataset loading for the Time-PEFT comparison suite.

Public CSV/EDF sources are described by ``datasets/manifest.yaml``. The five
chaotic systems default to deterministic, protocol-compatible local generators
for legacy experiments. Paper-reproduction runs can explicitly select the
version-pinned official ``dysts`` implementation instead.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}
_REQUIRED_SPLITS = {"train", "validation", "test"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DYSTS_REPRODUCTION_VERSION = "0.96"


def dysts_reproduction_provenance() -> dict[str, Any]:
    """Return and enforce the numerical environment for official trajectories."""

    try:
        dysts_version = metadata.version("dysts")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Official chaotic-data reproduction requires the project-pinned dysts package"
        ) from error
    if dysts_version != _DYSTS_REPRODUCTION_VERSION:
        raise RuntimeError(
            "Official chaotic-data reproduction requires "
            f"dysts=={_DYSTS_REPRODUCTION_VERSION}, found {dysts_version}"
        )
    numba_version = (
        metadata.version("numba") if importlib.util.find_spec("numba") is not None else None
    )
    return {
        "dysts_version": dysts_version,
        "numpy_version": metadata.version("numpy"),
        "scipy_version": metadata.version("scipy"),
        "numba_version": numba_version,
        "trajectory_arguments": {
            "initial_conditions": "dysts-system-metadata-default",
            "integration_step": "dysts-system-metadata-dt",
            "resample": True,
            "return_times": False,
            "standardize": False,
            "postprocess": True,
            "noise": 0.0,
            "timescale": "Fourier",
            "method": "Radau",
            "rtol": 1e-12,
            "atol": 1e-12,
            "verbose": False,
        },
    }


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """One immutable half-open chronological partition."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset split name cannot be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"Dataset split {self.name!r} has invalid range [{self.start}, {self.end})"
            )

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DatasetSeries:
    """A multivariate series and its leakage-safe chronological partitions."""

    name: str
    family: str
    values: Tensor
    splits: tuple[DatasetSplit, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("Dataset name and family cannot be empty")
        if not isinstance(self.values, Tensor) or self.values.ndim != 2:
            raise ValueError("Dataset values must be a torch.Tensor with shape [channels, time]")
        if self.values.shape[0] < 1 or self.values.shape[1] < 3:
            raise ValueError(
                "Dataset values must contain at least one channel and three timestamps"
            )
        if not self.values.dtype.is_floating_point:
            raise ValueError("Dataset values must use a floating-point dtype")
        if torch.isinf(self.values).any():
            raise ValueError("Dataset values contain positive or negative infinity")
        if not torch.isfinite(self.values).any(dim=1).all():
            raise ValueError("Every dataset channel must contain at least one finite value")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("Dataset sha256 must be a lowercase 64-character hexadecimal digest")

        names = {split.name for split in self.splits}
        if names != _REQUIRED_SPLITS or len(self.splits) != len(_REQUIRED_SPLITS):
            raise ValueError("Dataset splits must contain train, validation, and test exactly once")
        ordered = sorted(self.splits, key=lambda split: split.start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.end > current.start:
                raise ValueError(f"Dataset splits {previous.name!r} and {current.name!r} overlap")
        if ordered[-1].end > self.values.shape[1]:
            raise ValueError(
                f"Dataset split ends at {ordered[-1].end}, but the series has only "
                f"{self.values.shape[1]} timestamps"
            )

    def split(self, name: str) -> DatasetSplit:
        """Return a named split; ``val`` and ``valid`` alias ``validation``."""

        normalized = _SPLIT_ALIASES.get(name.strip().lower(), name.strip().lower())
        for partition in self.splits:
            if partition.name == normalized:
                return partition
        choices = ", ".join(split.name for split in self.splits)
        raise KeyError(f"Unknown split {name!r} for {self.name}; choose one of: {choices}")


def load_dataset_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the repository's JSON-compatible YAML manifest."""

    manifest_path = Path(path) if path is not None else _default_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run from the repository checkout "
            "or pass manifest_path explicitly."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Dataset manifest {manifest_path} is not valid JSON-compatible YAML: {error}"
        ) from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("datasets"), dict):
        raise ValueError(
            f"Dataset manifest {manifest_path} must have schema_version=1 and a datasets map"
        )
    return payload


def available_datasets(*, manifest_path: str | Path | None = None) -> tuple[str, ...]:
    """Return canonical dataset names from the configured manifest."""

    manifest = load_dataset_manifest(manifest_path)
    return tuple(manifest["datasets"])


def load_dataset_series(
    name: str,
    root: str | Path,
    *,
    download: bool = False,
    lorenz_length: int | None = None,
    synthetic_length: int | None = None,
    synthetic_generator: str = "compatible",
    synthetic_random_seed: int = 0,
    synthetic_pts_per_period: int = 100,
    local_path: str | Path | None = None,
    source_url: str | None = None,
    manifest_path: str | Path | None = None,
    verify_hash: bool = True,
) -> DatasetSeries:
    """Load one Time-PEFT dataset into ``[channels, time]`` layout.

    ``local_path`` and ``source_url`` are explicit per-call overrides. Without
    overrides, the pinned location and digest in the manifest are enforced.
    ``download`` never overwrites an existing file. ``synthetic_generator`` is
    either ``compatible`` (the legacy local proxies) or ``dysts`` (the official
    Gilpin benchmark package pinned by this project).
    """

    manifest = load_dataset_manifest(manifest_path)
    canonical, entry = _resolve_dataset(name, manifest["datasets"])
    family = _string_field(entry, "family", canonical)
    data_root = Path(root).expanduser()
    if synthetic_generator not in {"compatible", "dysts"}:
        raise ValueError(
            "synthetic_generator must be 'compatible' or 'dysts', got "
            f"{synthetic_generator!r}"
        )

    if "generator" in entry:
        if local_path is not None or source_url is not None:
            raise ValueError(
                f"{canonical} is generated locally and does not accept local_path/source_url"
            )
        length = synthetic_length
        if canonical == "Lorenz" and lorenz_length is not None:
            length = lorenz_length
        if length is None:
            length = int(entry.get("default_length", 12_000))
        if synthetic_generator == "compatible":
            values = _generate_synthetic(canonical, length)
        elif synthetic_generator == "dysts":
            values = _load_or_generate_dysts_synthetic(
                canonical,
                length,
                cache_root=data_root,
                random_seed=synthetic_random_seed,
                pts_per_period=synthetic_pts_per_period,
            )
        splits = _build_splits(entry, values.shape[1], canonical)
        return DatasetSeries(
            name=canonical,
            family=family,
            values=values,
            splits=splits,
            sha256=_tensor_sha256(values),
        )

    path = _locate_data_file(
        data_root,
        canonical,
        entry,
        local_path=local_path,
    )
    downloaded = False
    if not path.is_file():
        if not download:
            expected = entry.get("local_path", path.name)
            raise FileNotFoundError(
                f"{canonical} is not available locally. Expected {data_root / str(expected)}. "
                "Re-run with download=True (CLI: prepare-data --download) or pass "
                "local_path=/path/to/the/file."
            )
        url = source_url or _string_field(entry, "url", canonical)
        _download_file(url, path)
        downloaded = True

    digest = _file_sha256(path)
    pinned_source = source_url is None and local_path is None
    if verify_hash and pinned_source:
        expected_digest = _string_field(entry, "sha256", canonical).lower()
        if digest != expected_digest:
            origin = "downloaded" if downloaded else "local"
            raise ValueError(
                f"SHA-256 mismatch for {origin} {canonical} file {path}: expected "
                f"{expected_digest}, got {digest}. Remove the corrupt file and download again, "
                "or pass an intentional alternate file through local_path."
            )

    strict_shape = pinned_source and verify_hash
    if path.suffix.lower() == ".edf" or canonical.startswith("ECGCA"):
        values = _load_edf(path, canonical, entry)
    else:
        values = _load_csv(path, canonical, entry, strict_shape=strict_shape)
    splits = _build_splits(entry, values.shape[1], canonical)
    return DatasetSeries(
        name=canonical,
        family=family,
        values=values,
        splits=splits,
        sha256=digest,
    )


def _default_manifest_path() -> Path:
    checkout_path = Path(__file__).resolve().parents[3] / "datasets" / "manifest.yaml"
    if checkout_path.is_file():
        return checkout_path
    return Path.cwd() / "datasets" / "manifest.yaml"


def _resolve_dataset(requested: str, datasets: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    normalized = _normalize_name(requested)
    matches: dict[str, str] = {}
    for canonical, raw_entry in datasets.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Manifest entry for {canonical} must be a mapping")
        for alias in (canonical, *raw_entry.get("aliases", [])):
            key = _normalize_name(str(alias))
            previous = matches.setdefault(key, canonical)
            if previous != canonical:
                raise ValueError(f"Dataset alias {alias!r} is ambiguous in the manifest")
    if normalized not in matches:
        choices = ", ".join(datasets)
        raise ValueError(f"Unknown dataset {requested!r}; available datasets: {choices}")
    canonical = matches[normalized]
    return canonical, datasets[canonical]


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _string_field(entry: Mapping[str, Any], field: str, dataset: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest entry {dataset} requires a non-empty {field!r} field")
    return value


def _locate_data_file(
    root: Path,
    canonical: str,
    entry: Mapping[str, Any],
    *,
    local_path: str | Path | None,
) -> Path:
    manifest_relative = Path(_string_field(entry, "local_path", canonical))
    manifest_target = root / manifest_relative
    if local_path is not None:
        override = Path(local_path).expanduser()
        return override if override.is_absolute() else root / override

    candidates = (
        manifest_target,
        root / manifest_relative.name,
        root / canonical / manifest_relative.name,
        root / canonical.lower() / manifest_relative.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return manifest_target


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.download")
    request = urllib.request.Request(url, headers={"User-Agent": "utility-peft/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,  # noqa: S310
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(target)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download dataset from {url}: {error}. Download it manually and pass "
            "local_path, or check network/proxy access."
        ) from error


def _load_csv(
    path: Path,
    canonical: str,
    entry: Mapping[str, Any],
    *,
    strict_shape: bool,
) -> Tensor:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse {canonical} CSV {path}: {error}") from error
    if frame.empty:
        raise ValueError(f"{canonical} CSV {path} contains no data rows")

    time_columns = {
        column
        for column in frame.columns
        if str(column).strip().lower() in {"date", "datetime", "timestamp", "time"}
    }
    candidate = frame.drop(columns=list(time_columns))
    invalid_columns: list[str] = []
    numeric_columns: dict[str, pd.Series] = {}
    for column in candidate.columns:
        converted = pd.to_numeric(candidate[column], errors="coerce")
        newly_missing = candidate[column].notna() & converted.isna()
        if newly_missing.any():
            invalid_columns.append(str(column))
        else:
            numeric_columns[str(column)] = converted
    if invalid_columns:
        raise ValueError(
            f"{canonical} CSV {path} has non-numeric signal columns: "
            f"{', '.join(invalid_columns)}"
        )
    if not numeric_columns:
        raise ValueError(f"{canonical} CSV {path} has no numeric signal columns")

    matrix = pd.DataFrame(numeric_columns).to_numpy(dtype=np.float32, copy=True).T
    expected_channels = int(entry["channels"])
    if matrix.shape[0] != expected_channels:
        raise ValueError(
            f"{canonical} requires {expected_channels} signal channels after removing its time "
            f"column, but {path} contains {matrix.shape[0]}"
        )
    expected_rows = entry.get("rows")
    if strict_shape and expected_rows is not None and matrix.shape[1] != int(expected_rows):
        raise ValueError(
            f"Pinned {canonical} file must have {expected_rows} rows, but {path} has "
            f"{matrix.shape[1]}"
        )
    if np.isinf(matrix).any():
        raise ValueError(f"{canonical} CSV {path} contains positive or negative infinity")
    if not np.isfinite(matrix).any(axis=1).all():
        raise ValueError(f"Every signal column in {canonical} CSV {path} needs finite data")
    return torch.from_numpy(np.ascontiguousarray(matrix))


def _load_edf(path: Path, canonical: str, entry: Mapping[str, Any]) -> Tensor:
    try:
        import pyedflib  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            f"Reading {canonical} requires the optional EDF reader. Install it with "
            "`python -m pip install pyedflib`, then retry."
        ) from error

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = list(reader.getSignalLabels())
        signal_indices = [
            index
            for index, label in enumerate(labels)
            if "thorax" in label.lower() or "abdomen" in label.lower()
        ]
        expected_channels = int(entry["channels"])
        if len(signal_indices) != expected_channels:
            raise ValueError(
                f"{canonical} EDF {path} must expose two Thorax and four Abdomen signals; "
                f"found labels: {labels}"
            )
        sample_rates = [float(reader.getSampleFrequency(index)) for index in signal_indices]
        if max(sample_rates) != min(sample_rates):
            raise ValueError(
                f"{canonical} EDF signal sampling rates differ ({sample_rates}); resampling "
                "must be explicit before forecasting"
            )
        arrays = [
            np.asarray(reader.readSignal(index), dtype=np.float32) for index in signal_indices
        ]
    finally:
        reader.close()

    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"{canonical} EDF signal lengths differ: {sorted(lengths)}")
    matrix = np.stack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{canonical} EDF {path} contains non-finite signal samples")
    return torch.from_numpy(np.ascontiguousarray(matrix))


def _build_splits(
    entry: Mapping[str, Any], length: int, canonical: str
) -> tuple[DatasetSplit, ...]:
    raw_split = entry.get("split")
    if not isinstance(raw_split, Mapping):
        raise ValueError(f"Manifest entry {canonical} requires a split mapping")
    kind = raw_split.get("kind")
    if kind == "fixed":
        train_end = int(raw_split["train_end"])
        validation_end = int(raw_split["validation_end"])
        test_end = int(raw_split["test_end"])
    elif kind == "ratio":
        train_fraction = float(raw_split["train"])
        test_fraction = float(raw_split["test"])
        if not 0 < train_fraction < 1 or not 0 < test_fraction < 1:
            raise ValueError(f"Manifest split fractions for {canonical} must lie in (0, 1)")
        if train_fraction + test_fraction >= 1:
            raise ValueError(f"Manifest split fractions for {canonical} leave no validation data")
        train_end = int(length * train_fraction)
        validation_end = length - int(length * test_fraction)
        test_end = length
    else:
        raise ValueError(f"Manifest split kind for {canonical} must be 'fixed' or 'ratio'")

    if not 0 < train_end < validation_end < test_end <= length:
        raise ValueError(
            f"{canonical} has {length} timestamps, insufficient for configured boundaries "
            f"0 < {train_end} < {validation_end} < {test_end}. Acquire the complete source file."
        )
    return (
        DatasetSplit("train", 0, train_end),
        DatasetSplit("validation", train_end, validation_end),
        DatasetSplit("test", validation_end, test_end),
    )


def _generate_synthetic(canonical: str, length: int) -> Tensor:
    if length < 20:
        raise ValueError(f"Synthetic dataset length must be at least 20, got {length}")
    generators: dict[str, Callable[[int], np.ndarray]] = {
        "Lorenz": _lorenz,
        "CellCycle": _cell_cycle,
        "DoublePendulum": _double_pendulum,
        "Hopfield": _hopfield,
        "LorenzCoupled": _lorenz_coupled,
    }
    try:
        values = generators[canonical](length)
    except KeyError as error:
        raise ValueError(f"No local generator is implemented for {canonical}") from error
    if values.ndim != 2 or values.shape[1] != length or not np.isfinite(values).all():
        raise RuntimeError(f"Synthetic generator {canonical} produced an invalid trajectory")
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _generate_dysts_synthetic(
    canonical: str,
    length: int,
    *,
    random_seed: int,
    pts_per_period: int,
) -> Tensor:
    """Generate a trajectory with the official, version-pinned ``dysts`` package.

    The Time-PEFT paper names the Gilpin benchmark codebase but omits its exact
    release and trajectory arguments. Reproduction configs therefore record
    these explicit assumptions rather than relying on mutable package defaults.
    """

    if length < 20:
        raise ValueError(f"Synthetic dataset length must be at least 20, got {length}")
    if random_seed < 0:
        raise ValueError(f"synthetic_random_seed must be non-negative, got {random_seed}")
    if pts_per_period < 1:
        raise ValueError(
            f"synthetic_pts_per_period must be positive, got {pts_per_period}"
        )

    dysts_reproduction_provenance()
    try:
        from dysts import flows
    except ImportError as error:  # pragma: no cover - dependency installation failure
        raise RuntimeError(
            "Official chaotic-data reproduction requires the project-pinned dysts package"
        ) from error

    system_types: dict[str, type[Any]] = {
        "Lorenz": flows.Lorenz,
        "CellCycle": flows.CellCycle,
        "DoublePendulum": flows.DoublePendulum,
        "Hopfield": flows.Hopfield,
        "LorenzCoupled": flows.LorenzCoupled,
    }
    try:
        system = system_types[canonical]()
    except KeyError as error:
        raise ValueError(f"No dysts generator is available for {canonical}") from error

    numpy_random_state = np.random.get_state()
    try:
        trajectory = np.asarray(
            system.make_trajectory(
                length,
                init_cond=None,
                resample=True,
                pts_per_period=pts_per_period,
                return_times=False,
                standardize=False,
                postprocess=True,
                noise=0.0,
                timescale="Fourier",
                method="Radau",
                random_seed=random_seed,
                rtol=1e-12,
                atol=1e-12,
                verbose=False,
            ),
            dtype=np.float64,
        )
    finally:
        np.random.set_state(numpy_random_state)
    if trajectory.ndim != 2 or trajectory.shape[0] != length:
        raise RuntimeError(
            f"dysts generator {canonical} returned shape {trajectory.shape}, expected "
            f"[{length}, channels]"
        )
    values = trajectory.T
    if not np.isfinite(values).all():
        raise RuntimeError(f"dysts generator {canonical} produced a non-finite trajectory")
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _load_or_generate_dysts_synthetic(
    canonical: str,
    length: int,
    *,
    cache_root: Path,
    random_seed: int,
    pts_per_period: int,
) -> Tensor:
    """Materialize one exact official trajectory and validate it on every reuse."""

    identity: dict[str, Any] = {
        "schema_version": 1,
        "dataset": canonical,
        "length": length,
        "random_seed": random_seed,
        "pts_per_period": pts_per_period,
        "runtime_provenance": dysts_reproduction_provenance(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    cache_path = (
        cache_root
        / ".utility_peft"
        / "dysts"
        / canonical
        / f"trajectory-{cache_key}.pt"
    )
    if cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(
                f"Cannot read materialized dysts trajectory {cache_path}: {error}. "
                "Remove that single cache file and regenerate it."
            ) from error
        return _validate_dysts_cache_payload(payload, identity, cache_path)

    values = _generate_dysts_synthetic(
        canonical,
        length,
        random_seed=random_seed,
        pts_per_period=pts_per_period,
    )
    payload = {
        "identity": identity,
        "values": values,
        "sha256": _tensor_sha256(values),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=f".{cache_path.stem}-",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return values


def _validate_dysts_cache_payload(
    payload: Any,
    expected_identity: dict[str, Any],
    path: Path,
) -> Tensor:
    if not isinstance(payload, dict) or payload.get("identity") != expected_identity:
        raise RuntimeError(f"Materialized dysts trajectory has incompatible identity: {path}")
    values = payload.get("values")
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise RuntimeError(f"Materialized dysts trajectory has invalid values: {path}")
    expected_shape = (
        int(_dysts_channel_count(str(expected_identity["dataset"]))),
        int(expected_identity["length"]),
    )
    if tuple(values.shape) != expected_shape or values.dtype != torch.float32:
        raise RuntimeError(
            f"Materialized dysts trajectory {path} has shape/dtype "
            f"{tuple(values.shape)}/{values.dtype}, expected {expected_shape}/torch.float32"
        )
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"Materialized dysts trajectory contains non-finite values: {path}")
    observed_sha256 = _tensor_sha256(values)
    if payload.get("sha256") != observed_sha256:
        raise RuntimeError(f"Materialized dysts trajectory SHA-256 mismatch: {path}")
    return values


def _dysts_channel_count(canonical: str) -> int:
    try:
        return {
            "Lorenz": 3,
            "CellCycle": 6,
            "DoublePendulum": 4,
            "Hopfield": 6,
            "LorenzCoupled": 6,
        }[canonical]
    except KeyError as error:
        raise ValueError(f"No dysts channel count is registered for {canonical}") from error


def _rk4_trajectory(
    derivative: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    length: int,
    time_step: float,
    burn_in: int,
) -> np.ndarray:
    state = np.asarray(initial, dtype=np.float64).copy()
    trajectory = np.empty((state.size, length), dtype=np.float64)
    for step in range(length + burn_in):
        first = derivative(state)
        second = derivative(state + 0.5 * time_step * first)
        third = derivative(state + 0.5 * time_step * second)
        fourth = derivative(state + time_step * third)
        state = state + (time_step / 6.0) * (first + 2 * second + 2 * third + fourth)
        if step >= burn_in:
            trajectory[:, step - burn_in] = state
    return trajectory.astype(np.float32)


def _lorenz(length: int) -> np.ndarray:
    def derivative(state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array((10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 * z / 3.0))

    return _rk4_trajectory(
        derivative,
        np.array((1.0, 1.0, 1.0)),
        length=length,
        time_step=0.01,
        burn_in=1_000,
    )


def _cell_cycle(length: int) -> np.ndarray:
    """Generate a stable six-state regulatory oscillator (paper-compatible proxy)."""

    def derivative(state: np.ndarray) -> np.ndarray:
        messenger = state[:3]
        protein = state[3:]
        repression = np.roll(protein, 1)
        d_messenger = -messenger + 10.0 / (1.0 + repression**3) + 0.05
        d_protein = -2.0 * (protein - messenger)
        return np.concatenate((d_messenger, d_protein))

    return _rk4_trajectory(
        derivative,
        np.array((0.2, 1.0, 2.0, 2.5, 0.5, 1.5)),
        length=length,
        time_step=0.03,
        burn_in=2_000,
    )


def _double_pendulum(length: int) -> np.ndarray:
    def derivative(state: np.ndarray) -> np.ndarray:
        theta_1, omega_1, theta_2, omega_2 = state
        mass_1 = mass_2 = length_1 = length_2 = 1.0
        gravity = 9.81
        delta = theta_2 - theta_1
        sine_delta = math.sin(delta)
        cosine_delta = math.cos(delta)
        denominator_1 = (
            mass_1 + mass_2
        ) * length_1 - mass_2 * length_1 * cosine_delta * cosine_delta
        denominator_2 = (length_2 / length_1) * denominator_1
        acceleration_1 = (
            mass_2 * length_1 * omega_1**2 * sine_delta * cosine_delta
            + mass_2 * gravity * math.sin(theta_2) * cosine_delta
            + mass_2 * length_2 * omega_2**2 * sine_delta
            - (mass_1 + mass_2) * gravity * math.sin(theta_1)
        ) / denominator_1
        acceleration_2 = (
            -mass_2 * length_2 * omega_2**2 * sine_delta * cosine_delta
            + (mass_1 + mass_2)
            * (
                gravity * math.sin(theta_1) * cosine_delta
                - length_1 * omega_1**2 * sine_delta
                - gravity * math.sin(theta_2)
            )
        ) / denominator_2
        return np.array((omega_1, acceleration_1, omega_2, acceleration_2))

    return _rk4_trajectory(
        derivative,
        np.array((2.0, 0.0, 1.3, 0.0)),
        length=length,
        time_step=0.01,
        burn_in=500,
    )


def _hopfield(length: int) -> np.ndarray:
    """Generate a bounded continuous recurrent-network trajectory."""

    weights = np.array(
        (
            (0.0, 1.4, -1.2, 0.0, 0.4, -0.7),
            (-0.8, 0.0, 1.5, -1.1, 0.0, 0.5),
            (0.6, -1.3, 0.0, 1.4, -0.9, 0.0),
            (0.0, 0.8, -1.0, 0.0, 1.6, -1.2),
            (-1.4, 0.0, 0.7, -0.6, 0.0, 1.3),
            (1.2, -0.9, 0.0, 0.8, -1.1, 0.0),
        )
    )
    bias = np.array((0.10, -0.06, 0.04, -0.08, 0.07, -0.03))

    def derivative(state: np.ndarray) -> np.ndarray:
        return -state + 3.0 * weights @ np.tanh(state) + bias

    return _rk4_trajectory(
        derivative,
        np.array((0.8, -0.4, 0.2, -0.7, 0.5, -0.1)),
        length=length,
        time_step=0.02,
        burn_in=1_000,
    )


def _lorenz_coupled(length: int) -> np.ndarray:
    def derivative(state: np.ndarray) -> np.ndarray:
        first = state[:3]
        second = state[3:]

        def subsystem(current: np.ndarray) -> np.ndarray:
            x, y, z = current
            return np.array((10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 * z / 3.0))

        first_change = subsystem(first)
        second_change = subsystem(second)
        coupling = 0.8 * (second[0] - first[0])
        first_change[0] += coupling
        second_change[0] -= coupling
        return np.concatenate((first_change, second_change))

    return _rk4_trajectory(
        derivative,
        np.array((1.0, 1.0, 1.0, -1.0, -1.0, 15.0)),
        length=length,
        time_step=0.01,
        burn_in=1_000,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


__all__ = [
    "DatasetSeries",
    "DatasetSplit",
    "available_datasets",
    "load_dataset_manifest",
    "load_dataset_series",
]
