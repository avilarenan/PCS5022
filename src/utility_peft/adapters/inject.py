"""Deterministic wrappers around the pinned Hugging Face PEFT linear layers."""

from __future__ import annotations

from collections.abc import Callable

from peft.tuners.fourierft.layer import FourierFTLinear as PeftFourierFTLinear
from peft.tuners.lora.layer import Linear as PeftLoRALinear
from torch import nn


class LoRALinear(PeftLoRALinear):
    """PEFT LoRA with rank 8-style initialization and zero dropout."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int) -> None:
        super().__init__(
            base,
            adapter_name="default",
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
        )


class FourierFTLinear(PeftFourierFTLinear):
    """PEFT FourierFT with published defaults and zero-impact initialization."""

    def __init__(self, base: nn.Linear, *, n_frequency: int = 1000, seed: int = 777) -> None:
        super().__init__(
            base,
            adapter_name="default",
            n_frequency=min(n_frequency, base.weight.numel()),
            scaling=150.0,
            init_weights=True,
            random_loc_seed=seed,
        )


def inject_lora(
    module: nn.Module, target_names: tuple[str, ...], *, rank: int, alpha: int
) -> tuple[str, ...]:
    return _inject(
        module,
        target_names,
        lambda linear, _: LoRALinear(linear, rank=rank, alpha=alpha),
    )


def inject_fourierft(
    module: nn.Module, target_names: tuple[str, ...], *, n_frequency: int = 1000
) -> tuple[str, ...]:
    return _inject(
        module,
        target_names,
        lambda linear, _: FourierFTLinear(linear, n_frequency=n_frequency),
    )


def _inject(
    module: nn.Module,
    target_names: tuple[str, ...],
    factory: Callable[[nn.Linear, str], nn.Module],
) -> tuple[str, ...]:
    replacements: list[tuple[str, nn.Linear]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in target_names:
            replacements.append((name, child))
    for name, child in replacements:
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, leaf, factory(child, name))
    if not replacements:
        raise ValueError(f"No linear modules matched adapter targets {target_names}")
    return tuple(name for name, _ in replacements)
