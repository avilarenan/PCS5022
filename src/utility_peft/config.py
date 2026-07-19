"""Hydra composition without changing the process working directory."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(
    name: str = "config",
    overrides: list[str] | tuple[str, ...] | None = None,
    *,
    config_dir: str | Path | None = None,
) -> DictConfig:
    root = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(root.resolve())):
        config = compose(config_name=name, overrides=list(overrides or ()))
    OmegaConf.resolve(config)
    return config


def resolved_dict(config: DictConfig) -> dict[str, object]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Root Hydra configuration must be a mapping")
    return value
