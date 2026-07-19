from __future__ import annotations

import torch
from torch import nn

from utility_peft.actions import ACTION_BY_ID, MVP_ACTIONS
from utility_peft.adapters.inject import FourierFTLinear, LoRALinear
from utility_peft.model import model_for_action, trainable_parameter_names
from utility_peft.utils import seed_everything


def test_every_adapter_has_zero_impact_before_training(template, episode) -> None:
    template.eval()
    x = episode.support.x[:2]
    mask = episode.support.mask[:2]
    baseline = model_for_action(template, ACTION_BY_ID["A0"]).eval().predict(x, mask, 8)
    for action in MVP_ACTIONS[1:]:
        seed_everything(101)
        candidate = model_for_action(template, action).eval().predict(x, mask, 8)
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)


def test_exact_trainable_parameter_sets(template) -> None:
    expected_fragments = {
        "A0": (),
        "A1": ("forecast_head",),
        "A2": ("forecast_head", "lora_A", "lora_B"),
        "A3": ("forecast_head", "lora_A", "lora_B", "frequency_adapter"),
        "A4": ("forecast_head", "lora_A", "lora_B", "channel_adapter"),
        "A5": (
            "forecast_head",
            "lora_A",
            "lora_B",
            "frequency_adapter",
            "channel_adapter",
        ),
        "A6": ("forecast_head", "fourierft_spectrum"),
    }
    for action in MVP_ACTIONS:
        names = trainable_parameter_names(model_for_action(template, action))
        if action.action_id == "A0":
            assert names == ()
            continue
        assert names
        assert all(
            any(fragment in name for fragment in expected_fragments[action.action_id])
            for name in names
        )
        for fragment in expected_fragments[action.action_id]:
            assert any(fragment in name for name in names)


def test_lora_and_fourier_target_query_and_value_in_every_block(template) -> None:
    lora = model_for_action(template, ACTION_BY_ID["A2"])
    assert isinstance(lora.backbone.encoder[0].q, LoRALinear)
    assert isinstance(lora.backbone.encoder[0].v, LoRALinear)
    assert isinstance(lora.backbone.encoder[0].k, nn.Linear)
    assert lora.backbone.encoder[0].q.r["default"] == 8
    assert lora.backbone.encoder[0].q.lora_alpha["default"] == 16
    assert isinstance(lora.backbone.encoder[0].q.lora_dropout["default"], nn.Identity)
    fourier = model_for_action(template, ACTION_BY_ID["A6"])
    assert isinstance(fourier.backbone.encoder[0].q, FourierFTLinear)
    assert isinstance(fourier.backbone.encoder[0].v, FourierFTLinear)
    assert isinstance(fourier.backbone.encoder[0].k, nn.Linear)
    assert fourier.backbone.encoder[0].q.fourierft_scaling["default"] == 150.0
    assert fourier.backbone.encoder[0].q.fourierft_random_loc_seed["default"] == 777
    assert torch.count_nonzero(fourier.backbone.encoder[0].q.fourierft_spectrum["default"]) == 0


def test_a0_a1_do_not_mutate_backbone_structure_or_weights(template) -> None:
    pristine = template.backbone.state_dict()
    for action_id in ("A0", "A1"):
        candidate = model_for_action(template, ACTION_BY_ID[action_id])
        assert isinstance(candidate.backbone.encoder[0].q, nn.Linear)
        assert pristine.keys() == candidate.backbone.state_dict().keys()
        for name, value in pristine.items():
            assert torch.equal(value, candidate.backbone.state_dict()[name])


def test_fresh_clone_restores_exact_initial_state(template) -> None:
    seed_everything(99)
    first = model_for_action(template, ACTION_BY_ID["A3"])
    initial = {name: value.clone() for name, value in first.state_dict().items()}
    with torch.no_grad():
        for parameter in first.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    seed_everything(99)
    restored = model_for_action(template, ACTION_BY_ID["A3"])
    assert initial.keys() == restored.state_dict().keys()
    for name, value in initial.items():
        assert torch.equal(value, restored.state_dict()[name])


def test_full_tuning_excludes_project_adapters(template) -> None:
    model = model_for_action(template, ACTION_BY_ID["A7"])
    names = trainable_parameter_names(model)
    assert names
    assert all(name.startswith("backbone.") for name in names)
    assert not any("frequency_adapter" in name or "channel_adapter" in name for name in names)
