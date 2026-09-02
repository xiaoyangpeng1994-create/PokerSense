from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from poker_engine.core.enums import ActionType, Rank, Street, Suit
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.strategy.advice import AdviceStatus, build_advice
from poker_engine.strategy.blueprint_provider import (
    HuPreflopBlueprintProvider,
    action_history_token,
    hand_class,
)
from poker_engine.strategy.contracts import ActionAmountSemantics, LegalAction
from poker_engine.strategy.provider import LookupState
from poker_engine.strategy.router import StrategyRouter
from poker_engine.strategy.serialization import (
    strategy_deserialize,
    strategy_serialize,
)
from poker_engine.strategy.advice import Advice

from .helpers import NOW, context


GOLDEN_PATH = (
    Path(__file__).parents[1]
    / "fixtures/strategy/provider/hu_preflop_blueprint_golden.json"
)
GOLDEN_DATA = json.loads(GOLDEN_PATH.read_text())


@dataclass(frozen=True)
class Entry:
    stack_bb: int
    ante_bb: float
    sha256: str


class GoldenLoader:
    def __init__(self, golden, *, values=None, error: Exception | None = None):
        source = golden["source"]
        entry_specs = {
            (
                golden["spot"]["stack_bb"],
                golden["spot"]["ante_bb"],
                source["shard_sha256"],
            ),
            *{
                (spot["stack_bb"], spot["ante_bb"], spot["shard_sha256"])
                for spot in golden.get("additional_spots", [])
            },
        }
        self.manifest = SimpleNamespace(
            schema_version=source["manifest_schema"],
            premium_a_version=source["asset_version"],
            entries=tuple(Entry(*item) for item in sorted(entry_specs)),
        )
        self._golden = golden
        self._values = values
        self._error = error
        self.queries = []

    def actions(self, **query):
        if self._error:
            raise self._error
        self.queries.append(("actions", query))
        for spot in self._golden.get("additional_spots", []):
            if (
                spot["stack_bb"] == query["stack_bb"]
                and spot["ante_bb"] == query["ante"]
                and spot["action_history"] == query["action_history"]
            ):
                return spot["action_labels"]
        root = self._golden["spot"]
        if (
            root["stack_bb"] == query["stack_bb"]
            and root["ante_bb"] == query["ante"]
            and root["action_history"] == query["action_history"]
        ):
            return root["action_labels"]
        return None

    def lookup(self, *, hand, **query):
        if self._error:
            raise self._error
        self.queries.append(("lookup", {**query, "hand": hand}))
        if self._values is not None:
            return self._values
        for spot in self._golden.get("additional_spots", []):
            if (spot["stack_bb"] == query["stack_bb"]
                    and spot["ante_bb"] == query["ante"]
                    and spot["action_history"] == query["action_history"]
                    and spot["hand"] == hand):
                return [float(value) for value in spot["probabilities"]]
        root = self._golden["spot"]
        values = None
        if (root["stack_bb"] == query["stack_bb"]
                and root["ante_bb"] == query["ante"]
                and root["action_history"] == query["action_history"]):
            values = self._golden["hands"].get(hand)
        return None if values is None else [float(value) for value in values]


@pytest.fixture
def golden():
    return json.loads(GOLDEN_PATH.read_text())


def provider(golden, **loader_options):
    source = golden["source"]
    return HuPreflopBlueprintProvider(
        GoldenLoader(golden, **loader_options),
        solver_version=source["package_version"],
        source_revision=source["commit"],
        manifest_sha256=source["manifest_sha256"],
    )


def root_context(*, cards=(Card(Rank.ACE, Suit.SPADES),
                           Card(Rank.ACE, Suit.HEARTS))):
    value = context(2)
    seats = tuple(
        replace(seat, is_hero=seat.seat_id == 0)
        for seat in value.seats
    )
    return replace(
        value,
        seats=seats,
        hero_seat=0,
        actor_seat=0,
        hero_cards=cards,
        hero_range=None,
    )


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ((Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)), "AA"),
        ((Card(Rank.KING, Suit.SPADES), Card(Rank.ACE, Suit.SPADES)), "AKs"),
        ((Card(Rank.ACE, Suit.CLUBS), Card(Rank.KING, Suit.HEARTS)), "AKo"),
        ((Card(Rank.TWO, Suit.CLUBS), Card(Rank.SEVEN, Suit.DIAMONDS)), "72o"),
    ],
)
def test_hand_class_is_canonical(cards, expected):
    assert hand_class(cards) == expected


@pytest.mark.parametrize("hand", tuple(GOLDEN_DATA["hands"]))
def test_provider_matches_direct_upstream_golden(golden, hand):
    first = Rank(hand[0])
    second = Rank(hand[1])
    if len(hand) == 2:
        cards = (Card(first, Suit.SPADES), Card(second, Suit.HEARTS))
    elif hand[2] == "s":
        cards = (Card(first, Suit.SPADES), Card(second, Suit.SPADES))
    else:
        cards = (Card(first, Suit.SPADES), Card(second, Suit.HEARTS))
    result = provider(golden).query(root_context(cards=cards))

    assert result.state is LookupState.HIT_EXACT
    candidate = result.candidate
    assert candidate is not None
    raw = [Decimal(value) for value in golden["hands"][hand]]
    total = sum(raw, Decimal("0"))
    expected_raise = sum(raw[2:6], Decimal("0")) / total
    tolerance = Decimal("1e-25")
    assert abs(
        candidate.action_probabilities[ActionType.RAISE] - expected_raise
    ) <= tolerance
    assert abs(
        candidate.action_probabilities[ActionType.FOLD] - raw[0] / total
    ) <= tolerance
    assert abs(
        candidate.action_probabilities[ActionType.CALL] - raw[1] / total
    ) <= tolerance
    assert candidate.recommended_sizes[ActionType.RAISE] == (
        ChipAmount("2"), ChipAmount("3"), ChipAmount("4"), ChipAmount("5")
    )
    assert tuple(option.source_label for option in candidate.action_options) == (
        tuple(golden["spot"]["action_labels"])
    )
    assert sum(
        (option.probability for option in candidate.action_options),
        Decimal("0"),
    ) == Decimal("1")
    assert golden["source"]["commit"] in candidate.provider_version
    assert any(
        golden["source"]["shard_sha256"] in item
        for item in candidate.evidence
    )


def test_capability_is_hu_preflop_root_only(golden):
    capability = provider(golden).capability
    assert capability.player_counts == frozenset({2})
    assert capability.streets == frozenset({Street.PREFLOP})
    assert capability.action_lines == frozenset({
        "unopened", "limp", "raise", "three_bet", "four_bet",
        "iso_raise", "all_in",
    })
    assert capability.stack_buckets_bb == (Decimal("20"), Decimal("100"))
    assert capability.ante_values_are_bb is True


@pytest.mark.parametrize(
    ("big_blind", "ante", "expected_ante_bb"),
    [("2", "1", 0.5), ("2", "2", 1.0)],
)
def test_nonzero_ante_is_matched_in_big_blind_units(
    golden, big_blind, ante, expected_ante_bb
):
    loader = GoldenLoader(golden)
    source = golden["source"]
    adapter = HuPreflopBlueprintProvider(
        loader,
        solver_version=source["package_version"],
        source_revision=source["commit"],
        manifest_sha256=source["manifest_sha256"],
    )
    value = root_context()
    value = replace(
        value,
        game_config=replace(
            value.game_config,
            small_blind=ChipAmount("1"),
            big_blind=ChipAmount(big_blind),
            ante=ChipAmount(ante),
            minimum_chip=ChipAmount("1"),
        ),
    )
    result = adapter.query(value)
    assert result.state is LookupState.HIT_EXACT
    assert ("actions", {
        "stack_bb": 100,
        "ante": expected_ante_bb,
        "action_history": "",
    }) in loader.queries
    expected_shard = next(
        spot["shard_sha256"] for spot in golden["additional_spots"]
        if spot["stack_bb"] == 100 and spot["ante_bb"] == expected_ante_bb
    )
    assert any(expected_shard in item for item in result.candidate.evidence)


def test_unsupported_ante_bb_is_not_applicable(golden):
    value = root_context()
    value = replace(
        value,
        game_config=replace(
            value.game_config,
            small_blind=ChipAmount("1"),
            big_blind=ChipAmount("2"),
            ante=ChipAmount("0.5"),
            minimum_chip=ChipAmount("0.5"),
        ),
    )
    result = provider(golden).query(value)
    assert result.state is LookupState.NOT_APPLICABLE
    assert result.reasons == ("unsupported_ante",)


@pytest.mark.parametrize(
    "spot", tuple(GOLDEN_DATA["additional_spots"]),
    ids=lambda spot: (
        f"{spot['stack_bb']}bb-a{spot['ante_bb']}-"
        f"{spot['action_history'] or 'root'}-{spot['hand']}"
    ),
)
def test_additional_nodes_match_golden_through_adapter(golden, spot):
    ranks = (Rank(spot["hand"][0]), Rank(spot["hand"][1]))
    suited = len(spot["hand"]) == 3 and spot["hand"][2] == "s"
    cards = (
        Card(ranks[0], Suit.SPADES),
        Card(ranks[1], Suit.SPADES if suited else Suit.HEARTS),
    )
    value = root_context(cards=cards)
    if spot["position"] == "BB":
        value = replace(
            value,
            seats=tuple(replace(
                seat, is_hero=seat.seat_id == 1
            ) for seat in value.seats),
            hero_seat=1,
        )
    big_blind = Decimal("2")
    events = []
    actor = 0
    aggressive = 0
    saw_limp = False
    saw_all_in = False
    for token in re.findall(r"[fcxA]|[br]\d+", spot["action_history"]):
        payload = {"seat_id": actor}
        if token == "f":
            event_type = EventType.FOLD
        elif token == "c":
            event_type = EventType.CALL
            if not events:
                saw_limp = True
        elif token == "x":
            event_type = EventType.CHECK
        elif token == "A":
            event_type = EventType.ALL_IN
            saw_all_in = True
            aggressive += 1
        else:
            event_type = EventType.RAISE
            payload.update({
                "amount": ChipAmount(
                    big_blind * Decimal(token[1:]) / Decimal("100")
                ),
                "amount_semantics": "total_street",
            })
            aggressive += 1
        events.append(StateEvent(
            event_type, value.hand_id, value.state_version, payload, NOW
        ))
        actor = 1 - actor
    action_line = (
        "all_in" if saw_all_in
        else "limp" if aggressive == 0 and saw_limp
        else "unopened" if aggressive == 0
        else "iso_raise" if aggressive == 1 and saw_limp
        else "raise" if aggressive == 1
        else "three_bet" if aggressive == 2
        else "four_bet"
    )
    value = replace(
        value,
        game_config=replace(
            value.game_config,
            small_blind=ChipAmount("1"),
            big_blind=ChipAmount(big_blind),
            ante=ChipAmount(big_blind * Decimal(str(spot["ante_bb"]))),
            minimum_chip=ChipAmount("0.5"),
        ),
        effective_stack_bb=Decimal(spot["stack_bb"]),
        action_history=tuple(events),
        actor_seat=actor,
        action_line=action_line,
    )
    result = provider(golden).query(value)
    assert result.state is LookupState.HIT_EXACT
    assert tuple(
        option.source_label for option in result.candidate.action_options
    ) == tuple(spot["action_labels"])
    for option, expected in zip(
        result.candidate.action_options, spot["probabilities"], strict=True
    ):
        raw = [Decimal(item) for item in spot["probabilities"]]
        expected_probability = Decimal(expected) / sum(raw, Decimal("0"))
        assert abs(option.probability - expected_probability) <= Decimal("1e-25")
    assert any(
        spot["shard_sha256"] in item for item in result.candidate.evidence
    )


def test_golden_root_runs_provider_router_advice_end_to_end(golden):
    value = replace(
        root_context(),
        legal_actions=(
            LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.CALL, ChipAmount("0.5"), ChipAmount("0.5")),
            LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("100")),
            LegalAction(
                ActionType.ALL_IN, ChipAmount("99.5"), ChipAmount("99.5")
            ),
        ),
    )
    route = StrategyRouter((provider(golden),)).route(value, now=NOW)
    advice = build_advice(value, route, now=NOW)

    assert route.state is LookupState.HIT_EXACT
    assert advice.status is AdviceStatus.READY
    assert advice.preferred_action is ActionType.RAISE
    assert advice.recommended_sizes[ActionType.RAISE] == (
        ChipAmount("2"), ChipAmount("3"), ChipAmount("4"), ChipAmount("5")
    )
    assert len(advice.action_options) == 7
    assert advice.action_options[2].amount == ChipAmount("2")
    assert advice.strategy_version == route.selected.provider_version
    assert strategy_deserialize(
        Advice, strategy_serialize(advice)
    ) == advice


def test_illegal_source_sizes_are_removed_and_remaining_options_reweighted(golden):
    value = replace(
        root_context(),
        legal_actions=(
            LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.CALL, ChipAmount("0.5"), ChipAmount("0.5")),
            LegalAction(ActionType.RAISE, ChipAmount("4"), ChipAmount("4")),
        ),
    )
    route = StrategyRouter((provider(golden),)).route(value, now=NOW)
    advice = build_advice(value, route, now=NOW)

    assert advice.status is AdviceStatus.READY
    assert advice.recommended_sizes == {
        ActionType.RAISE: (ChipAmount("4"),)
    }
    assert tuple(option.source_label for option in advice.action_options) == (
        "fold", "call", "open_to_400",
    )
    assert sum(advice.action_probabilities.values()) == Decimal("1")
    assert sum(
        (option.probability for option in advice.action_options), Decimal("0")
    ) == Decimal("1")


def test_non_hu_context_is_not_applicable(golden):
    result = provider(golden).query(context(3))
    assert result.state is LookupState.NOT_APPLICABLE
    assert "unsupported_player_count" in result.reasons


def test_inconsistent_history_is_rejected(golden):
    value = root_context()
    event = StateEvent(
        EventType.RAISE,
        value.hand_id,
        value.state_version,
        {
            "seat_id": 0,
            "amount": "3",
            "amount_semantics": "total_street",
        },
        NOW,
    )
    result = provider(golden).query(replace(value, action_history=(event,)))
    assert result.state is LookupState.REJECTED
    assert "actor_seat does not follow" in result.reasons[0]


def test_raise_event_maps_to_real_b300_golden(golden):
    value = context(2, action_line="raise")
    event = StateEvent(
        EventType.RAISE,
        value.hand_id,
        value.state_version,
        {
            "seat_id": 0,
            "amount": ChipAmount("3"),
            "amount_semantics": ActionAmountSemantics.TOTAL_STREET,
        },
        NOW,
    )
    value = replace(
        value,
        hero_cards=(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)),
        action_history=(event,),
    )
    result = provider(golden).query(value)

    assert action_history_token(value) == "b300"
    assert result.state is LookupState.HIT_EXACT
    assert result.candidate.recommended_sizes[ActionType.RAISE] == (
        ChipAmount("7"), ChipAmount("9"), ChipAmount("11"), ChipAmount("13")
    )


def test_history_maps_call_three_bet_and_all_in_tokens():
    base = root_context()

    def event(event_type, seat_id, *, amount=None):
        payload = {"seat_id": seat_id}
        if amount is not None:
            payload.update({
                "amount": ChipAmount(amount),
                "amount_semantics": "total_street",
            })
        return StateEvent(
            event_type, base.hand_id, base.state_version, payload, NOW
        )

    limp = replace(
        context(2, action_line="limp"),
        action_history=(event(EventType.CALL, 0),),
    )
    assert action_history_token(limp) == "c"

    three_bet = replace(
        base,
        action_line="three_bet",
        action_history=(
            event(EventType.RAISE, 0, amount="3"),
            event(EventType.RAISE, 1, amount="7"),
        ),
    )
    assert action_history_token(three_bet) == "b300r700"

    all_in = replace(
        context(2, action_line="all_in"),
        action_history=(event(EventType.ALL_IN, 0),),
    )
    assert action_history_token(all_in) == "A"


def test_history_amount_is_normalized_by_big_blind():
    value = context(2, action_line="raise")
    config = replace(
        value.game_config,
        small_blind=ChipAmount("1"),
        big_blind=ChipAmount("2"),
        minimum_chip=ChipAmount("1"),
    )
    event = StateEvent(
        EventType.RAISE,
        value.hand_id,
        value.state_version,
        {
            "seat_id": 0,
            "amount": ChipAmount("6"),
            "amount_semantics": "total_street",
        },
        NOW,
    )
    assert action_history_token(replace(
        value, game_config=config, action_history=(event,)
    )) == "b300"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"amount": "3", "amount_semantics": "total_street"}, "seat_id"),
        ({"seat_id": 0, "amount": "3"}, "total_street"),
        ({"seat_id": 0, "amount": "3.001", "amount_semantics": "total_street"},
         "0.01BB"),
    ],
)
def test_history_requires_explicit_precise_event_payload(golden, payload, reason):
    value = context(2, action_line="raise")
    event = StateEvent(
        EventType.RAISE, value.hand_id, value.state_version, payload, NOW
    )
    result = provider(golden).query(replace(value, action_history=(event,)))
    assert result.state is LookupState.REJECTED
    assert reason in result.reasons[0]


def test_unknown_hand_is_not_found(golden):
    loader = GoldenLoader(golden)
    loader._golden = {**golden, "hands": {}}
    source = golden["source"]
    value = HuPreflopBlueprintProvider(
        loader,
        solver_version=source["package_version"],
        source_revision=source["commit"],
        manifest_sha256=source["manifest_sha256"],
    )
    assert value.query(root_context()).state is LookupState.NOT_FOUND


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([1.0], "action labels and probabilities must align"),
        ([float("nan")] * 7, "probabilities must be finite"),
        ([0.0] * 7, "probability total must be positive"),
    ],
)
def test_malformed_upstream_output_is_rejected(golden, values, reason):
    result = provider(golden, values=values).query(root_context())
    assert result.state is LookupState.REJECTED
    assert reason in result.reasons[0]


def test_loader_failure_is_contained(golden):
    result = provider(golden, error=OSError("broken shard")).query(root_context())
    assert result.state is LookupState.REJECTED
    assert result.reasons == ("blueprint_loader_error:OSError",)


def test_manifest_digest_is_required(golden):
    with pytest.raises(ValueError, match="64-character"):
        HuPreflopBlueprintProvider(
            GoldenLoader(golden),
            solver_version="1.11.0",
            source_revision="revision",
            manifest_sha256="bad",
        )
