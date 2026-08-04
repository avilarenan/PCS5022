from __future__ import annotations

import torch

from utility_peft.actions import resolve_time_peft_actions
from utility_peft.adapters.modules import PaperChannelAdapter, PaperFrequencyAdapter
from utility_peft.backbones.tiny import TinyBackbone
from utility_peft.config import load_config
from utility_peft.model import AdaptableForecaster, model_for_action


def test_paper_adapter_parameter_formula_and_shapes() -> None:
    width = 16
    channels = 3
    frequency = PaperFrequencyAdapter(width, top_k=3)
    channel = PaperChannelAdapter(width, channels)
    assert sum(parameter.numel() for parameter in frequency.parameters()) == width**2
    assert sum(parameter.numel() for parameter in channel.parameters()) == (
        width**2 + channels * width**2 // 2
    )
    embeddings = torch.randn(2, channels, 8, width)
    filtered = frequency(embeddings)
    assert filtered.shape == embeddings.shape
    assert channel(embeddings, filtered).shape == embeddings.shape
    assert torch.count_nonzero(frequency.projection.weight) > 0
    assert torch.count_nonzero(channel.channel_up) > 0


def test_count_inferred_variant_matches_paper_table_rounded_increment() -> None:
    """Biases and affine normalization explain the paper's 4.429M count exactly."""

    width = 768
    channels = 3
    frequency = PaperFrequencyAdapter(width, top_k=3, bias=True)
    channel = PaperChannelAdapter(width, channels, bias=True)
    normalization = torch.nn.LayerNorm(width, elementwise_affine=True)
    optional = sum(
        parameter.numel()
        for module in (frequency, channel, normalization)
        for parameter in module.parameters()
    )
    assert optional == 2_069_376
    # Paper Table 5 averages the horizon-specific head size. Together with
    # Q/K/V LoRA, the fixed term is 2,359,504 parameters.
    assert round((2_359_504 + optional) / 1_000_000, 3) == 4.429


def test_count_inferred_variant_trains_biases_and_affine_norm() -> None:
    torch.manual_seed(8)
    template = AdaptableForecaster(
        TinyBackbone(d_model=16, patch_len=4, depth=1, heads=2, max_horizon=8),
        channels=3,
        adapter_implementation="paper_count_inferred",
    )
    model = model_for_action(
        template,
        resolve_time_peft_actions(("LFC",), update_steps=1)[0],
    )
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    assert "frequency_adapter.projection.bias" in names
    assert "channel_adapter.shared_down.bias" in names
    assert "channel_adapter.channel_up_bias" in names
    assert "paper_output_norm.weight" in names
    assert "paper_output_norm.bias" in names
    assert torch.count_nonzero(model.channel_adapter.channel_up_bias) > 0


def test_paper_routes_have_exact_trainable_modules() -> None:
    torch.manual_seed(4)
    template = AdaptableForecaster(
        TinyBackbone(d_model=16, patch_len=4, depth=1, heads=2, max_horizon=8),
        channels=3,
        adapter_implementation="paper",
    )
    x = torch.randn(2, 3, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    for action in resolve_time_peft_actions(("L", "LF", "LC", "LFC"), update_steps=1):
        model = model_for_action(template, action)
        prediction = model.predict(x, mask, 8)
        assert prediction.shape == (2, 3, 8)
        assert torch.isfinite(prediction).all()
        names = {name for name, value in model.named_parameters() if value.requires_grad}
        assert any("lora_A" in name for name in names)
        assert any(".k." in name and "lora_" in name for name in names)
        assert any("forecast_head" in name for name in names)
        assert any("frequency_adapter" in name for name in names) == ("frequency" in action.modules)
        assert any("channel_adapter" in name for name in names) == ("channel" in action.modules)
        assert not any("paper_output_norm" in name for name in names)


def test_lfc_follows_accepted_paper_algorithm_one_dataflow() -> None:
    torch.manual_seed(5)
    template = AdaptableForecaster(
        TinyBackbone(d_model=16, patch_len=4, depth=1, heads=2, max_horizon=8),
        channels=3,
        adapter_implementation="paper",
    )
    model = model_for_action(
        template,
        resolve_time_peft_actions(("LFC",), update_steps=1)[0],
    )
    captured: dict[str, object] = {}

    def frequency_hook(_module, arguments, output) -> None:
        captured["backbone"] = arguments[0].detach()
        captured["filtered"] = output.detach()

    def channel_pre_hook(_module, arguments) -> None:
        captured["channel_backbone"] = arguments[0].detach()
        captured["channel_filtered"] = arguments[1].detach()

    def channel_hook(_module, _arguments, output) -> None:
        captured["channel"] = output.detach()

    def norm_pre_hook(_module, arguments) -> None:
        captured["norm_input"] = arguments[0].detach()

    handles = (
        model.frequency_adapter.register_forward_hook(frequency_hook),
        model.channel_adapter.register_forward_pre_hook(channel_pre_hook),
        model.channel_adapter.register_forward_hook(channel_hook),
        model.paper_output_norm.register_forward_pre_hook(norm_pre_hook),
    )
    try:
        encoded = model.encode(
            torch.randn(2, 3, 16),
            torch.ones(2, 16, dtype=torch.bool),
        )
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(captured["channel_backbone"], captured["backbone"])
    torch.testing.assert_close(captured["channel_filtered"], captured["filtered"])
    torch.testing.assert_close(captured["norm_input"], captured["channel"])
    torch.testing.assert_close(encoded, model.paper_output_norm(captured["channel"]))


def test_smaller_routing_arms_have_explicit_ablation_dataflows() -> None:
    torch.manual_seed(6)
    template = AdaptableForecaster(
        TinyBackbone(d_model=16, patch_len=4, depth=1, heads=2, max_horizon=8),
        channels=3,
        adapter_implementation="paper",
    )
    x = torch.randn(2, 3, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)

    lf = model_for_action(template, resolve_time_peft_actions(("LF",), update_steps=1)[0])
    lf_values: dict[str, torch.Tensor] = {}
    lf_handles = (
        lf.frequency_adapter.register_forward_hook(
            lambda _module, arguments, output: lf_values.update(
                backbone=arguments[0].detach(), filtered=output.detach()
            )
        ),
        lf.paper_output_norm.register_forward_pre_hook(
            lambda _module, arguments: lf_values.update(norm_input=arguments[0].detach())
        ),
    )
    try:
        lf.encode(x, mask)
    finally:
        for handle in lf_handles:
            handle.remove()
    torch.testing.assert_close(
        lf_values["norm_input"],
        lf_values["backbone"] + lf_values["filtered"],
    )

    lc = model_for_action(template, resolve_time_peft_actions(("LC",), update_steps=1)[0])
    lc_values: dict[str, torch.Tensor] = {}
    lc_handles = (
        lc.channel_adapter.register_forward_pre_hook(
            lambda _module, arguments: lc_values.update(
                backbone=arguments[0].detach(), filtered=arguments[1].detach()
            )
        ),
        lc.channel_adapter.register_forward_hook(
            lambda _module, _arguments, output: lc_values.update(channel=output.detach())
        ),
        lc.paper_output_norm.register_forward_pre_hook(
            lambda _module, arguments: lc_values.update(norm_input=arguments[0].detach())
        ),
    )
    try:
        lc.encode(x, mask)
    finally:
        for handle in lc_handles:
            handle.remove()
    assert torch.count_nonzero(lc_values["filtered"]) == 0
    torch.testing.assert_close(lc_values["norm_input"], lc_values["channel"])


def test_paper_model_rejects_unknown_channel_count() -> None:
    backbone = TinyBackbone(d_model=8, patch_len=4, depth=1, heads=2, max_horizon=8)
    try:
        AdaptableForecaster(backbone, adapter_implementation="paper")
    except ValueError as error:
        assert "channel count" in str(error)
    else:
        raise AssertionError("paper adapter mode must require a channel count")


def test_full_correlation_config_covers_paper_suite_and_expected_record_count() -> None:
    config = load_config("correlation")
    assert config.model.adapter_implementation == "paper"
    assert config.correlation.noninferiority_margin == 0.01
    assert config.correlation.random_state == 17
    assert "bootstrap_samples" not in config.correlation
    assert set(config.experiment.actions) == {"L", "LF", "LC", "LFC"}
    assert len(config.experiment.datasets) == 13
    assert "ETTm2" in config.experiment.datasets
    assert config.source_head.dataset == "Electricity"
    expected = (
        len(config.experiment.datasets)
        * len(config.experiment.horizons)
        * config.experiment.episodes_per_dataset_horizon
        * len(config.experiment.seeds)
        * len(config.experiment.actions)
    )
    assert expected == 1_404
