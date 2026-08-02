"""Representation adapters inserted before a forecasting head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FrequencyAdapter(nn.Module):
    """Residual spectral adapter over the patch axis."""

    def __init__(
        self, d_model: int, *, fraction: float = 0.25, bottleneck: int | None = None
    ) -> None:
        super().__init__()
        if not 0 < fraction <= 1:
            raise ValueError("frequency fraction must be in (0, 1]")
        width = bottleneck or max(d_model // 8, 1)
        self.fraction = fraction
        self.down = nn.Linear(d_model, width)
        self.activation = nn.GELU()
        self.up = nn.Linear(width, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError("FrequencyAdapter expects [batch, channels, patches, embedding]")
        patch_count = embeddings.shape[2]
        spectrum = torch.fft.rfft(embeddings.float(), dim=2, norm="ortho")
        bins = spectrum.shape[2]
        candidates = max(bins - 1, 1)
        selected = max(1, math.ceil(candidates * self.fraction))
        scores = spectrum.abs().mean(dim=(0, 1, 3))
        if bins > 1:
            indices = scores[1:].topk(min(selected, bins - 1)).indices + 1
        else:
            indices = torch.zeros(1, dtype=torch.long, device=scores.device)
        gate = torch.zeros(bins, dtype=spectrum.real.dtype, device=spectrum.device)
        gate[indices] = 1.0
        real_delta = self.up(self.activation(self.down(spectrum.real)))
        imag_delta = self.up(self.activation(self.down(spectrum.imag)))
        # CUDA autocast can return BF16 here, which torch.complex does not accept.
        delta = torch.complex(real_delta.float(), imag_delta.float())
        delta = delta * gate.view(1, 1, -1, 1)
        residual = torch.fft.irfft(delta, n=patch_count, dim=2, norm="ortho")
        return embeddings + residual.to(embeddings.dtype)


class ChannelAdapter(nn.Module):
    """Residual bottleneck using channel-centered representations."""

    def __init__(self, d_model: int, *, bottleneck: int | None = None) -> None:
        super().__init__()
        width = bottleneck or max(d_model // 8, 1)
        self.down = nn.Linear(d_model, width)
        self.activation = nn.GELU()
        self.up = nn.Linear(width, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError("ChannelAdapter expects [batch, channels, patches, embedding]")
        centered = embeddings - embeddings.mean(dim=1, keepdim=True)
        return embeddings + self.up(self.activation(self.down(centered)))


class PaperFrequencyAdapter(nn.Module):
    """Frequency path specified by the accepted Time-PEFT paper.

    The FFT is taken over patches. The top bins are selected independently per
    sample and channel, followed by an inverse FFT and an ``h1 -> h2``
    projection. ``bias=False`` matches the paper's ``h1 * h2`` parameter count.
    """

    def __init__(self, d_model: int, *, output_size: int | None = None, top_k: int = 3) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.output_size = output_size or d_model
        self.projection = nn.Linear(d_model, self.output_size, bias=False)
        # The paper does not publish initialization. Xavier initialization is
        # an explicit local choice; unlike the legacy MVP adapters, this path
        # does not alter Algorithm 1 with a residual zero-impact bypass.
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError("PaperFrequencyAdapter expects [batch, channels, patches, embedding]")
        patch_count = embeddings.shape[2]
        spectrum = torch.fft.rfft(embeddings.float(), dim=2, norm="ortho")
        amplitudes = spectrum.abs().mean(dim=-1)
        selected = min(self.top_k, spectrum.shape[2])
        indices = amplitudes.topk(selected, dim=2).indices
        mask = torch.zeros_like(amplitudes)
        mask.scatter_(2, indices, 1.0)
        filtered = torch.fft.irfft(
            spectrum * mask.unsqueeze(-1),
            n=patch_count,
            dim=2,
            norm="ortho",
        )
        return self.projection(filtered.to(embeddings.dtype))


class PaperChannelAdapter(nn.Module):
    """Shared-down/channel-specific-up path specified by Time-PEFT."""

    def __init__(
        self,
        d_model: int,
        channels: int,
        *,
        frequency_size: int | None = None,
        bottleneck: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        frequency_size = frequency_size or d_model
        width = bottleneck or max(d_model // 2, 1)
        self.channels = channels
        self.shared_down = nn.Linear(d_model + frequency_size, width, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.channel_up = nn.Parameter(torch.empty(channels, width, d_model))
        # Algorithm 1 sends this tensor directly to LayerNorm and the forecast
        # head, so a zero matrix would collapse every initial forecast. The
        # paper leaves initialization unspecified; initialize each channel's
        # matrix independently with the same standard Xavier rule.
        for channel in range(channels):
            nn.init.xavier_uniform_(self.channel_up[channel])

    def forward(self, embeddings: Tensor, filtered: Tensor) -> Tensor:
        if embeddings.ndim != 4 or filtered.ndim != 4:
            raise ValueError("PaperChannelAdapter expects two rank-four tensors")
        if embeddings.shape[:3] != filtered.shape[:3]:
            raise ValueError("Backbone and filtered embeddings must share their first axes")
        if embeddings.shape[1] != self.channels:
            raise ValueError(
                f"Adapter was built for {self.channels} channels, got {embeddings.shape[1]}"
            )
        combined = torch.cat((embeddings, filtered), dim=-1)
        hidden = self.dropout(self.activation(self.shared_down(combined)))
        return torch.einsum("bckr,crd->bckd", hidden, self.channel_up)
