"""Small deterministic backbone for CPU tests and pipeline reproduction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, patches, width = x.shape
        normalized = self.norm1(x)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(batch, patches, self.heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q(normalized))
        k = split_heads(self.k(normalized))
        v = split_heads(self.v(normalized))
        attention = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, patches, width)
        x = x + self.o(mixed)
        return x + self.ff(self.norm2(x))


class TinyBackbone(nn.Module):
    """Patch-transformer with MOMENT-compatible representation shapes."""

    def __init__(
        self,
        *,
        d_model: int = 32,
        patch_len: int = 8,
        depth: int = 2,
        heads: int = 4,
        max_horizon: int = 336,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.max_horizon = max_horizon
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.encoder = nn.ModuleList(
            [TinyAttentionBlock(d_model=d_model, heads=heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.forecast_head = nn.Linear(d_model, max_horizon)

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.ndim != 3 or mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError("Expected x [batch, channels, time] and mask [batch, time]")
        batch, channels, time = x.shape
        padding = (-time) % self.patch_len
        if padding:
            x = F.pad(x, (0, padding))
        patches = x.unfold(-1, self.patch_len, self.patch_len)
        patch_count = patches.shape[2]
        hidden = self.patch_embedding(patches).reshape(batch * channels, patch_count, self.d_model)
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return hidden.reshape(batch, channels, patch_count, self.d_model)

    def predict_from_embeddings(self, embeddings: Tensor, horizon: int) -> Tensor:
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")
        pooled = embeddings.mean(dim=2)
        return self.forecast_head(pooled)[..., :horizon]

    def predict(self, x: Tensor, mask: Tensor, horizon: int) -> Tensor:
        return self.predict_from_embeddings(self.encode(x, mask), horizon)

    def adapter_targets(self) -> tuple[str, ...]:
        return ("q", "v")

    def source_statistics(self) -> Mapping[str, float]:
        return {
            "source_d_model": float(self.d_model),
            "source_patch_len": float(self.patch_len),
            "source_encoder_depth": float(len(self.encoder)),
        }

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.forecast_head.parameters())

    def save_source_head(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.forecast_head.state_dict(), target)
