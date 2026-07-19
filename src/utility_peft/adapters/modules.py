"""Zero-impact representation adapters inserted before a forecasting head."""

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
