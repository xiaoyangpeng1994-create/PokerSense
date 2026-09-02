"""Optional adapter for versioned HU preflop blueprint assets.

The adapter deliberately depends on the small loader interface rather than on
the upstream package at import time.  Production construction uses
``from_asset_dir``; tests can inject a loader double without installing the
solver or bundling its assets with PokerSense.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Any, Protocol, Sequence

from poker_engine.core.enums import ActionType, Position, Street
from poker_engine.core.events import EventType
from poker_engine.core.value_objects import Card, ChipAmount

from .contracts import ActionAmountSemantics, DecisionContext, GameType
from .provider import (
    ActionOption,
    LookupState,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)


PROVIDER_ID = "amaster97-hu-preflop-blueprint"
UPSTREAM_URL = "https://github.com/amaster97/poker_solver"
_PLAYER_ACTION_EVENTS = frozenset({
    EventType.FOLD,
    EventType.CHECK,
    EventType.CALL,
    EventType.BET,
    EventType.RAISE,
    EventType.ALL_IN,
})


class BlueprintLoaderLike(Protocol):
    manifest: Any

    def actions(
        self, *, stack_bb: int, ante: float, action_history: str = ""
    ) -> list[str] | None: ...

    def lookup(
        self,
        *,
        stack_bb: int,
        ante: float,
        hand: str,
        action_history: str = "",
    ) -> Sequence[float] | None: ...


class HuPreflopBlueprintProvider:
    """Exact-context adapter for the upstream 169-class HU root strategy.

    Action history is accepted only when every player event carries a seat and
    aggressive actions explicitly use total-street amount semantics.  This
    prevents a chip delta from being silently encoded as a raise-to amount.
    """

    def __init__(
        self,
        loader: BlueprintLoaderLike,
        *,
        solver_version: str,
        source_revision: str,
        manifest_sha256: str,
    ) -> None:
        for name, value in (
            ("solver_version", solver_version),
            ("source_revision", source_revision),
            ("manifest_sha256", manifest_sha256),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if len(manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in manifest_sha256.lower()
        ):
            raise ValueError("manifest_sha256 must be a 64-character hex digest")
        entries = tuple(getattr(loader.manifest, "entries", ()))
        if not entries:
            raise ValueError("blueprint manifest must contain shard entries")
        schema = getattr(loader.manifest, "schema_version", None)
        asset_version = getattr(loader.manifest, "premium_a_version", None)
        if not schema or not asset_version:
            raise ValueError("blueprint manifest version metadata is missing")

        stacks = tuple(sorted({Decimal(entry.stack_bb) for entry in entries}))
        antes = tuple(sorted({Decimal(str(entry.ante_bb)) for entry in entries}))
        self._loader = loader
        self._solver_version = solver_version
        self._source_revision = source_revision
        self._manifest_sha256 = manifest_sha256.lower()
        self._schema_version = str(schema)
        self._asset_version = str(asset_version)
        self._entries = entries
        self._capability = ProviderCapability(
            player_counts=frozenset({2}),
            streets=frozenset({Street.PREFLOP}),
            game_types=frozenset({GameType.CASH}),
            stack_buckets_bb=stacks,
            ante_values=tuple(ChipAmount(value) for value in antes),
            rake_percent_values=(Decimal("0"),),
            action_lines=frozenset({
                "unopened", "limp", "raise", "three_bet", "four_bet",
                "iso_raise", "all_in",
            }),
            base_match_kind=MatchKind.EXACT,
            allow_stack_interpolation=False,
            priority=20,
            hero_positions=frozenset({Position.BTN, Position.BB}),
            ante_values_are_bb=True,
            stack_ante_pairs_bb=tuple(sorted({
                (Decimal(entry.stack_bb), Decimal(str(entry.ante_bb)))
                for entry in entries
            })),
        )

    @classmethod
    def from_asset_dir(
        cls,
        asset_dir: str | Path,
        *,
        solver_version: str,
        source_revision: str,
    ) -> "HuPreflopBlueprintProvider":
        """Load an installed upstream asset bundle with SHA verification on."""
        root = Path(asset_dir)
        manifest_path = root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        module = importlib.import_module("poker_solver.blueprint_loader")
        loader = module.BlueprintLoader.from_dir(root, verify_sha256=True)
        return cls(
            loader,
            solver_version=solver_version,
            source_revision=source_revision,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    @property
    def source_version(self) -> str:
        return (
            f"poker_solver/{self._solver_version}"
            f"@{self._source_revision}:premium-a/{self._asset_version}"
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    def query(self, context: DecisionContext) -> ProviderResult:
        match = self.capability.match(context)
        if not match.applicable:
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=match.reasons,
            )
        rejection = self._validate_context(context)
        if rejection:
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=rejection,
            )

        stack_bb = int(context.effective_stack_bb)
        ante_bb = float(
            context.game_config.ante.value / context.game_config.big_blind.value
        )
        hand = hand_class(context.hero_cards)
        try:
            history = action_history_token(context)
        except (TypeError, ValueError) as exc:
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(f"invalid_action_history:{exc}",),
            )
        try:
            labels = self._loader.actions(
                stack_bb=stack_bb, ante=ante_bb, action_history=history
            )
            values = self._loader.lookup(
                stack_bb=stack_bb,
                ante=ante_bb,
                hand=hand,
                action_history=history,
            )
        except Exception as exc:  # adapter boundary: never crash the router
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(f"blueprint_loader_error:{type(exc).__name__}",),
            )
        if labels is None or values is None:
            return ProviderResult(
                LookupState.NOT_FOUND,
                self.provider_id,
                reasons=("blueprint_infoset_or_hand_not_found",),
            )
        try:
            probabilities, sizes, options = self._convert_strategy(
                labels, values, context.game_config.big_blind
            )
            shard_sha = self._shard_sha(stack_bb, ante_bb)
        except (TypeError, ValueError) as exc:
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(f"invalid_blueprint_output:{exc}",),
            )

        candidate = StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            match_kind=MatchKind.EXACT,
            state_match_score=1.0,
            action_probabilities=probabilities,
            recommended_sizes=sizes,
            action_options=options,
            confidence=0.75,
            evidence=(
                f"{UPSTREAM_URL}/tree/{self._source_revision}",
                f"manifest_sha256:{self._manifest_sha256}",
                f"shard_sha256:{shard_sha}",
                f"blueprint:{stack_bb}bb:ante={ante_bb}:history={history}:{hand}",
            ),
            assumptions=(
                "hu_preflop_169_class_abstraction",
                "upstream_final_exploitability_unreported",
            ),
            expires_at=context.request.expires_at,
        )
        return ProviderResult(LookupState.HIT_EXACT, self.provider_id, candidate)

    def _validate_context(self, context: DecisionContext) -> tuple[str, ...]:
        reasons = []
        if context.game_config.variant.upper() not in {"NLHE", "HUNL"}:
            reasons.append("unsupported_variant")
        if len(context.hero_cards) != 2:
            reasons.append("missing_hero_cards")
        if context.actor_seat != context.hero_seat:
            reasons.append("hero_not_actor")
        actions = tuple(
            event for event in context.action_history
            if event.event_type in _PLAYER_ACTION_EVENTS
        )
        dealer = next((seat for seat in context.seats if seat.is_dealer), None)
        if dealer is None or dealer.position not in (Position.BTN, Position.SB):
            reasons.append("missing_hu_button")
        elif not actions and context.hero_seat != dealer.seat_id:
            reasons.append("hero_not_hu_button_root_actor")
        return tuple(reasons)

    def _convert_strategy(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        big_blind: ChipAmount,
    ) -> tuple[
        dict[ActionType, Decimal],
        dict[ActionType, tuple[ChipAmount, ...]],
        tuple[ActionOption, ...],
    ]:
        if not labels or len(labels) != len(values):
            raise ValueError("action labels and probabilities must align")
        raw_options = []
        for label, raw in zip(labels, values, strict=True):
            if not isinstance(label, str) or not label:
                raise TypeError("action labels must be non-empty strings")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("probabilities must be finite and non-negative")
            probability = Decimal(str(value))
            action, amount = _parse_action_label(label, big_blind)
            raw_options.append((action, amount, label, probability))
        total = sum((item[3] for item in raw_options), Decimal("0"))
        if total <= 0:
            raise ValueError("probability total must be positive")
        option_values = _normalize_probabilities(
            tuple(item[3] for item in raw_options), total
        )
        options = tuple(
            ActionOption(action, probability, amount, label)
            for (action, amount, label, _), probability
            in zip(raw_options, option_values, strict=True)
        )
        grouped: dict[ActionType, Decimal] = {}
        sizes: dict[ActionType, list[ChipAmount]] = {}
        for option in options:
            grouped[option.action] = (
                grouped.get(option.action, Decimal("0")) + option.probability
            )
            if option.amount is not None and option.probability > 0:
                sizes.setdefault(option.action, []).append(option.amount)
        # Re-grouping normalized options can round in a different order under
        # Decimal's active context. Preserve the strict probability contract
        # by assigning the residual to the final source action (normally the
        # all-in branch), exactly as done for the final source option above.
        final_action = options[-1].action
        grouped[final_action] = Decimal("1") - sum(
            (
                probability for action, probability in grouped.items()
                if action is not final_action
            ),
            Decimal("0"),
        )
        return (
            grouped,
            {
                action: tuple(dict.fromkeys(values))
                for action, values in sizes.items()
            },
            options,
        )

    def _shard_sha(self, stack_bb: int, ante_bb: float) -> str:
        for entry in self._entries:
            if entry.stack_bb == stack_bb and math.isclose(
                float(entry.ante_bb), ante_bb, rel_tol=0.0, abs_tol=1e-12
            ):
                value = getattr(entry, "sha256", None)
                if isinstance(value, str) and value:
                    return value
        raise ValueError("matching manifest shard metadata not found")


def hand_class(cards: Sequence[Card]) -> str:
    """Convert two concrete cards to a canonical 169-class label."""
    if len(cards) != 2 or not all(isinstance(card, Card) for card in cards):
        raise ValueError("exactly two Card values are required")
    first, second = sorted(cards, key=lambda card: card.rank_value, reverse=True)
    ranks = first.rank.value + second.rank.value
    if first.rank == second.rank:
        return ranks
    return ranks + ("s" if first.suit == second.suit else "o")


def action_history_token(context: DecisionContext) -> str:
    """Build the upstream HU preflop token from authoritative StateEvents."""
    if not isinstance(context, DecisionContext):
        raise TypeError("context must be a DecisionContext")
    seats = tuple(seat.seat_id for seat in context.seats)
    if len(seats) != 2:
        raise ValueError("HU history requires exactly two seats")
    dealer = next((seat.seat_id for seat in context.seats if seat.is_dealer), None)
    if dealer is None:
        raise ValueError("HU history requires a dealer seat")
    expected_actor = dealer
    tokens = []
    aggressive_count = 0
    saw_limp = False
    saw_all_in = False
    for event in context.action_history:
        if event.event_type not in _PLAYER_ACTION_EVENTS:
            continue
        if event.hand_id != context.hand_id:
            raise ValueError("event hand_id does not match context")
        seat_id = event.payload.get("seat_id")
        if not isinstance(seat_id, int) or isinstance(seat_id, bool):
            raise ValueError("player event requires integer seat_id")
        if seat_id != expected_actor:
            raise ValueError("player events do not alternate from the HU button")
        event_type = event.event_type
        if event_type is EventType.FOLD:
            tokens.append("f")
        elif event_type is EventType.CHECK:
            tokens.append("x")
        elif event_type is EventType.CALL:
            tokens.append("c")
            if not tokens[:-1]:
                saw_limp = True
        elif event_type is EventType.ALL_IN:
            tokens.append("A")
            saw_all_in = True
            aggressive_count += 1
        elif event_type in (EventType.BET, EventType.RAISE):
            semantics = event.payload.get("amount_semantics")
            if isinstance(semantics, ActionAmountSemantics):
                semantics = semantics.value
            if semantics != ActionAmountSemantics.TOTAL_STREET.value:
                raise ValueError("aggressive event requires total_street amount")
            amount = _event_amount(event.payload.get("amount"))
            unit = amount / context.game_config.big_blind.value * Decimal("100")
            if unit != unit.to_integral_value():
                raise ValueError("aggressive amount is not an exact 0.01BB unit")
            prefix = "b" if aggressive_count == 0 else "r"
            tokens.append(prefix + str(int(unit)))
            aggressive_count += 1
        expected_actor = next(seat for seat in seats if seat != seat_id)
    if context.actor_seat != expected_actor:
        raise ValueError("actor_seat does not follow the HU event sequence")
    expected_line = _history_action_line(aggressive_count, saw_limp, saw_all_in)
    if context.action_line != expected_line:
        raise ValueError(
            f"action_line {context.action_line!r} does not match {expected_line!r}"
        )
    return "".join(tokens)


def _event_amount(value: object) -> Decimal:
    if isinstance(value, ChipAmount):
        return value.value
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, (str, int)) and not isinstance(value, bool):
        amount = Decimal(value)
    else:
        raise TypeError("aggressive event amount must be ChipAmount/Decimal/str/int")
    if not amount.is_finite() or amount < 0:
        raise ValueError("aggressive event amount must be finite and non-negative")
    return amount


def _history_action_line(
    aggressive_count: int, saw_limp: bool, saw_all_in: bool
) -> str:
    if saw_all_in:
        return "all_in"
    if aggressive_count == 0:
        return "limp" if saw_limp else "unopened"
    if saw_limp and aggressive_count == 1:
        return "iso_raise"
    if aggressive_count == 1:
        return "raise"
    if aggressive_count == 2:
        return "three_bet"
    return "four_bet"


def _parse_action_label(
    label: str, big_blind: ChipAmount
) -> tuple[ActionType, ChipAmount | None]:
    fixed = {
        "fold": ActionType.FOLD,
        "check": ActionType.CHECK,
        "call": ActionType.CALL,
        "all_in": ActionType.ALL_IN,
    }
    if label in fixed:
        return fixed[label], None
    for prefix in ("open_to_", "raise_to_"):
        if label.startswith(prefix):
            digits = label[len(prefix):]
            if not digits.isdigit():
                break
            amount = big_blind.value * Decimal(digits) / Decimal("100")
            return ActionType.RAISE, ChipAmount(amount)
    raise ValueError(f"unsupported action label {json.dumps(label)}")


def _normalize_probabilities(
    values: tuple[Decimal, ...], total: Decimal
) -> list[Decimal]:
    """Allocate 1e-26 units by largest remainder for exact stable sums."""
    scale = 10 ** 26
    with localcontext() as decimal_context:
        decimal_context.prec = 80
        scaled = [value / total * scale for value in values]
    units = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in scaled]
    remaining = scale - sum(units)
    order = sorted(
        range(len(values)),
        key=lambda index: (scaled[index] - units[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        units[index] += 1
    divisor = Decimal(scale)
    return [Decimal(value) / divisor for value in units]


__all__ = [
    "BlueprintLoaderLike",
    "HuPreflopBlueprintProvider",
    "PROVIDER_ID",
    "UPSTREAM_URL",
    "action_history_token",
    "hand_class",
]
