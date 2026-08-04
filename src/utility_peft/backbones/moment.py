"""MOMENT-base wrapper with canonical representations and exact checkpoint pins."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

MOMENT_MODEL_ID = "AutonLab/MOMENT-1-base"
MOMENT_MODEL_REVISION = "5e44b0ea26376a176360f87831124e018f876d96"


class MomentBackbone(nn.Module):
    """Expose MOMENT encoder output as [batch, channels, patches, embedding]."""

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        model_id: str = MOMENT_MODEL_ID,
        revision: str = MOMENT_MODEL_REVISION,
        source_head_checkpoint: str | Path | None = None,
        allow_random_head: bool = False,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        try:
            from momentfm import MOMENTPipeline
        except ImportError as error:
            raise RuntimeError("Install the pinned momentfm dependency") from error
        self.lookback = lookback
        self.horizon = horizon
        self.model_id = model_id
        self.revision = revision
        self.pipeline = MOMENTPipeline.from_pretrained(
            model_id,
            revision=revision,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": horizon,
                "seq_len": lookback,
                # MOMENT and TSLib commonly use 0.1, but historical project
                # workflows used 0.0. Keep the value explicit at the wrapper
                # boundary so reproduction configs can record the assumption.
                "head_dropout": head_dropout,
                "freeze_embedder": True,
                "freeze_encoder": True,
                "freeze_head": False,
                # Directly injected adapter weights need gradients even though
                # the frozen patch embeddings do not require them.
                "enable_gradient_checkpointing": False,
            },
        )
        self.pipeline.init()
        self.d_model = int(self.pipeline.config.d_model)
        self.random_forecasting_head = source_head_checkpoint is None
        if source_head_checkpoint is not None:
            state = torch.load(source_head_checkpoint, map_location="cpu", weights_only=True)
            self.pipeline.head.load_state_dict(state)
        elif not allow_random_head:
            raise ValueError(
                "MOMENT-1-base has no pretrained long-horizon head. Provide "
                "source_head_checkpoint or explicitly set allow_random_head=True "
                "for a pilot-only run."
            )

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.shape[-1] != self.lookback:
            raise ValueError(f"MOMENT wrapper expects lookback {self.lookback}, got {x.shape[-1]}")
        batch, channels, _ = x.shape
        input_mask = mask.to(dtype=x.dtype)
        normalized = self.pipeline.normalizer(x=x, mask=input_mask, mode="norm")
        normalized = torch.nan_to_num(normalized, nan=0, posinf=0, neginf=0)
        patches = self.pipeline.tokenizer(x=normalized)
        embedded = self.pipeline.patch_embedding(patches, mask=torch.ones_like(input_mask))
        patch_count = embedded.shape[2]
        embedded = embedded.reshape(batch * channels, patch_count, self.d_model)
        from momentfm.utils.masking import Masking

        patch_mask = Masking.convert_seq_to_patch_view(input_mask, self.pipeline.patch_len)
        attention_mask = patch_mask.repeat_interleave(channels, dim=0)
        encoded = self.pipeline.encoder(
            inputs_embeds=embedded, attention_mask=attention_mask
        ).last_hidden_state
        return encoded.reshape(batch, channels, patch_count, self.d_model)

    def predict_from_embeddings(self, embeddings: Tensor, horizon: int) -> Tensor:
        if horizon != self.horizon:
            raise ValueError(f"MOMENT forecasting head was built for horizon {self.horizon}")
        normalized_forecast = self.pipeline.head(embeddings)
        return self.pipeline.normalizer(x=normalized_forecast, mode="denorm")

    def predict(self, x: Tensor, mask: Tensor, horizon: int) -> Tensor:
        return self.predict_from_embeddings(self.encode(x, mask), horizon)

    def adapter_targets(self) -> tuple[str, ...]:
        return ("q", "k", "v")

    def source_statistics(self) -> Mapping[str, float]:
        return {
            "source_d_model": float(self.d_model),
            "source_patch_len": float(self.pipeline.patch_len),
            "source_encoder_depth": float(len(self.pipeline.encoder.block)),
            "source_random_forecasting_head": float(self.random_forecasting_head),
        }

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.pipeline.head.parameters())

    def save_source_head(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.pipeline.head.state_dict(), target)
