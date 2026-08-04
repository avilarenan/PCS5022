from __future__ import annotations

import pytest

from utility_peft.backbones.moment import MomentBackbone


@pytest.mark.parametrize("value", (-0.1, 1.0))
def test_moment_backbone_rejects_invalid_head_dropout_before_model_load(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="head_dropout must be in"):
        MomentBackbone(
            lookback=96,
            horizon=96,
            allow_random_head=True,
            head_dropout=value,
        )
