"""Fixed adaptation actions used by the forecasting MVP."""

from __future__ import annotations

from utility_peft.types import ActionSpec

MVP_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("A0", frozenset(), update_steps=0),
    ActionSpec("A1", frozenset({"head"})),
    ActionSpec("A2", frozenset({"head", "lora"}), rank=8, alpha=16),
    ActionSpec("A3", frozenset({"head", "lora", "frequency"}), rank=8, alpha=16),
    ActionSpec("A4", frozenset({"head", "lora", "channel"}), rank=8, alpha=16),
    ActionSpec(
        "A5",
        frozenset({"head", "lora", "frequency", "channel"}),
        rank=8,
        alpha=16,
    ),
    ActionSpec("A6", frozenset({"head", "fourierft"})),
)

REFERENCE_ACTION = ActionSpec("A7", frozenset({"full"}), controller_action=False)
ALL_ACTIONS = MVP_ACTIONS + (REFERENCE_ACTION,)
ACTION_BY_ID = {action.action_id: action for action in ALL_ACTIONS}

# Matched comparison arms around the accepted-paper LFC baseline. The paper uses
# rank-8/alpha-32 LoRA on query, key, and value projections and trains the
# forecast head in all non-zero-shot configurations. LF and LC are explicit
# routing ablations whose dataflow is defined in AdaptableForecaster.encode().
TIME_PEFT_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "L",
        frozenset({"head", "lora"}),
        rank=8,
        alpha=32,
        target_modules=("q", "k", "v"),
    ),
    ActionSpec(
        "LF",
        frozenset({"head", "lora", "frequency"}),
        rank=8,
        alpha=32,
        target_modules=("q", "k", "v"),
    ),
    ActionSpec(
        "LC",
        frozenset({"head", "lora", "channel"}),
        rank=8,
        alpha=32,
        target_modules=("q", "k", "v"),
    ),
    ActionSpec(
        "LFC",
        frozenset({"head", "lora", "frequency", "channel"}),
        rank=8,
        alpha=32,
        target_modules=("q", "k", "v"),
    ),
)
TIME_PEFT_ACTION_BY_ID = {action.action_id: action for action in TIME_PEFT_ACTIONS}


def resolve_time_peft_actions(
    action_ids: list[str] | tuple[str, ...], *, update_steps: int | None = None
) -> list[ActionSpec]:
    """Resolve L/LF/LC/LFC with an optional matched update budget."""

    from dataclasses import replace

    unknown = sorted(set(action_ids) - TIME_PEFT_ACTION_BY_ID.keys())
    if unknown:
        raise ValueError(f"Unknown Time-PEFT comparison actions: {unknown}")
    actions = [TIME_PEFT_ACTION_BY_ID[action_id] for action_id in action_ids]
    if update_steps is not None:
        if update_steps <= 0:
            raise ValueError("update_steps must be positive")
        actions = [replace(action, update_steps=update_steps) for action in actions]
    return actions


def resolve_actions(
    action_ids: list[str] | tuple[str, ...], *, allow_reference: bool = True
) -> list[ActionSpec]:
    unknown = sorted(set(action_ids) - ACTION_BY_ID.keys())
    if unknown:
        raise ValueError(f"Unknown actions: {unknown}")
    actions = [ACTION_BY_ID[action_id] for action_id in action_ids]
    if not allow_reference and any(not action.controller_action for action in actions):
        raise ValueError("Reference action A7 cannot be selected by the controller")
    return actions
