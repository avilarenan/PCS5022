"""Action-conditioned forecasting model construction."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from torch import Tensor, nn

from utility_peft.adapters.inject import inject_fourierft, inject_lora
from utility_peft.adapters.modules import (
    ChannelAdapter,
    FrequencyAdapter,
    PaperChannelAdapter,
    PaperFrequencyAdapter,
)
from utility_peft.backbones.base import BackboneProtocol
from utility_peft.types import ActionSpec


class AdaptableForecaster(nn.Module):
    """Backbone plus optional representation adapters."""

    def __init__(
        self,
        backbone: BackboneProtocol,
        *,
        channels: int | None = None,
        adapter_implementation: str = "mvp",
        frequency_top_k: int = 3,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("BackboneProtocol implementations must also be torch modules")
        if adapter_implementation not in {"mvp", "paper", "paper_count_inferred"}:
            raise ValueError(
                "adapter_implementation must be 'mvp', 'paper', or "
                "'paper_count_inferred'"
            )
        self.backbone = backbone
        self.adapter_implementation = adapter_implementation
        self.channels = channels
        if adapter_implementation in {"paper", "paper_count_inferred"}:
            if channels is None or channels <= 0:
                raise ValueError("paper adapter implementation requires a channel count")
            count_inferred = adapter_implementation == "paper_count_inferred"
            self.frequency_adapter = PaperFrequencyAdapter(
                backbone.d_model,
                top_k=frequency_top_k,
                bias=count_inferred,
            )
            self.channel_adapter = PaperChannelAdapter(
                backbone.d_model,
                channels,
                dropout=adapter_dropout,
                bias=count_inferred,
            )
            # Algorithm 1 applies LayerNorm after the channel adapter. The
            # analytical formula omits biases and normalization, while the
            # rounded Table 5 count is exactly consistent with ordinary linear
            # biases and affine LayerNorm. Keep both interpretations explicit.
            self.paper_output_norm = nn.LayerNorm(
                backbone.d_model,
                elementwise_affine=count_inferred,
            )
        else:
            self.frequency_adapter = FrequencyAdapter(backbone.d_model)
            self.channel_adapter = ChannelAdapter(backbone.d_model)
            self.paper_output_norm = None
        self.frequency_enabled = False
        self.channel_enabled = False
        self.injected_modules: tuple[str, ...] = ()

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        embeddings = self.backbone.encode(x, mask)
        if embeddings.ndim != 4:
            raise RuntimeError(
                "Backbone encode() must return [batch, channels, patches, embedding]"
            )
        if self.adapter_implementation in {"paper", "paper_count_inferred"}:
            if self.frequency_enabled and self.channel_enabled:
                # Accepted-paper Algorithm 1: the frequency representation and
                # the original backbone representation feed the channel path;
                # only normalized channel embeddings reach the forecast head.
                filtered = self.frequency_adapter(embeddings)
                embeddings = self.paper_output_norm(
                    self.channel_adapter(embeddings, filtered)
                )
            elif self.frequency_enabled:
                # Routing extension: use the paper's F_both ablation and fuse
                # its two h1-sized streams by addition before normalization.
                # The paper reports F_both but does not specify this head-side
                # fusion operator for the channel-free case.
                filtered = self.frequency_adapter(embeddings)
                embeddings = self.paper_output_norm(embeddings + filtered)
            elif self.channel_enabled:
                # Routing extension: preserve the channel adapter's matched
                # (h1+h2)->r capacity while making the absent frequency branch
                # explicit as zeros. No residual bypass is added.
                filtered = embeddings.new_zeros(
                    (*embeddings.shape[:-1], self.frequency_adapter.output_size)
                )
                embeddings = self.paper_output_norm(
                    self.channel_adapter(embeddings, filtered)
                )
        else:
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
    available_targets = set(model.backbone.adapter_targets())
    targets = tuple(name for name in action.target_modules if name in available_targets)
    if "lora" in action.modules:
        if action.rank is None or action.alpha is None:
            raise ValueError("LoRA rank and alpha are required")
        if not targets:
            raise ValueError(
                f"None of the requested LoRA targets {action.target_modules} exist in the backbone"
            )
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
    if (
        model.adapter_implementation == "paper_count_inferred"
        and (model.frequency_enabled or model.channel_enabled)
        and model.paper_output_norm is not None
    ):
        _set_trainable(model.paper_output_norm.parameters())


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
