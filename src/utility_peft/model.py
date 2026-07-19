"""Action-conditioned forecasting model construction."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from torch import Tensor, nn

from utility_peft.adapters.inject import inject_fourierft, inject_lora
from utility_peft.adapters.modules import ChannelAdapter, FrequencyAdapter
from utility_peft.backbones.base import BackboneProtocol
from utility_peft.types import ActionSpec


class AdaptableForecaster(nn.Module):
    """Backbone plus optional representation adapters."""

    def __init__(self, backbone: BackboneProtocol) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("BackboneProtocol implementations must also be torch modules")
        self.backbone = backbone
        self.frequency_adapter = FrequencyAdapter(backbone.d_model)
        self.channel_adapter = ChannelAdapter(backbone.d_model)
        self.frequency_enabled = False
        self.channel_enabled = False
        self.injected_modules: tuple[str, ...] = ()

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        embeddings = self.backbone.encode(x, mask)
        if embeddings.ndim != 4:
            raise RuntimeError(
                "Backbone encode() must return [batch, channels, patches, embedding]"
            )
        if self.frequency_enabled:
            embeddings = self.frequency_adapter(embeddings)
        if self.channel_enabled:
            embeddings = self.channel_adapter(embeddings)
        return embeddings

    def predict(self, x: Tensor, mask: Tensor, horizon: int) -> Tensor:
        return self.backbone.predict_from_embeddings(self.encode(x, mask), horizon)

    def forward(self, x: Tensor, mask: Tensor, horizon: int) -> Tensor:
        return self.predict(x, mask, horizon)


def model_for_action(template: AdaptableForecaster, action: ActionSpec) -> AdaptableForecaster:
    """Clone the pristine template so one action can never contaminate another."""

    model = copy.deepcopy(template)
    activate_action(model, action)
    return model


def activate_action(model: AdaptableForecaster, action: ActionSpec) -> None:
    model.frequency_enabled = "frequency" in action.modules
    model.channel_enabled = "channel" in action.modules

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    injected: tuple[str, ...] = ()
    targets = model.backbone.adapter_targets()
    if "lora" in action.modules:
        if action.rank is None or action.alpha is None:
            raise ValueError("LoRA rank and alpha are required")
        injected = inject_lora(model.backbone, targets, rank=action.rank, alpha=action.alpha)
    if "fourierft" in action.modules:
        injected = inject_fourierft(model.backbone, targets)
    model.injected_modules = injected

    if "full" in action.modules:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
        return
    if "head" in action.modules:
        _set_trainable(model.backbone.head_parameters())
    if "lora" in action.modules:
        _set_named_trainable(model, ("lora_A", "lora_B"))
    if "fourierft" in action.modules:
        _set_named_trainable(model, ("fourierft_spectrum",))
    if model.frequency_enabled:
        _set_trainable(model.frequency_adapter.parameters())
    if model.channel_enabled:
        _set_trainable(model.channel_adapter.parameters())


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def _set_trainable(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.requires_grad_(True)


def _set_named_trainable(module: nn.Module, leaf_names: tuple[str, ...]) -> None:
    matched = 0
    for name, parameter in module.named_parameters():
        if any(fragment in name for fragment in leaf_names):
            parameter.requires_grad_(True)
            matched += 1
    if not matched:
        raise RuntimeError(f"No parameters matched {leaf_names}")
