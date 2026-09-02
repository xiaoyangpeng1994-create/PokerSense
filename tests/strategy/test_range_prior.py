"""Versioned RFI concrete-combo prior selection and refusal boundaries."""

from __future__ import annotations

from decimal import Decimal

import pytest

from poker_engine.core.enums import Position, Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.strategy.heuristic_provider import PreflopRfiHeuristicProvider
from poker_engine.strategy.range_prior import (
    PreflopRfiRangePrior,
    RangePriorQuery,
    RangePriorState,
    expand_hand_class,
)
from poker_engine.strategy.range_tracker import parse_concrete_combo


POSITIONS = {
    6: (Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB),
    9: (
        Position.UTG, Position.UTG1, Position.UTG2, Position.LJ,
        Position.HJ, Position.CO, Position.BTN, Position.SB,
    ),
}


def _query(
    players=6,
    position=Position.CO,
    *,
    stack="100",
    action_line="open_raise",
    known_cards=(),
):
    return RangePriorQuery(
        seat_id=2,
        player_count=players,
        position=position,
        effective_stack_bb=Decimal(stack),
        action_line=action_line,
        known_cards=known_cards,
    )


@pytest.mark.parametrize(
    ("hand", "count"),
    (("AA", 6), ("AKs", 4), ("AKo", 12)),
)
def test_hand_class_expansion_has_canonical_combo_count(hand, count):
    expanded = expand_hand_class(hand)

    assert len(expanded) == count
    assert len({combo for combo, _ in expanded}) == count
    assert all(parse_concrete_combo(combo) == cards for combo, cards in expanded)


@pytest.mark.parametrize("hand", ("A", "AK", "AAs", "AKx", "ZZ", 42))
def test_invalid_hand_classes_are_rejected(hand):
    with pytest.raises((TypeError, ValueError)):
        expand_hand_class(hand)


@pytest.mark.parametrize(
    ("players", "position"),
    tuple(
        (players, position)
        for players, positions in POSITIONS.items()
        for position in positions
    ),
)
def test_every_explicit_6_and_9_player_position_returns_concrete_prior(
    players, position,
):
    result = PreflopRfiRangePrior.from_builtin().lookup(
        _query(players, position)
    )

    assert result.state is RangePriorState.HIT
    assert result.distribution is not None
    assert result.distribution.seat_id == 2
    assert sum(result.distribution.combo_weights.values()) == 1
    assert all(len(combo) == 4 for combo in result.distribution.combo_weights)
    assert result.distribution.confidence == 0.4
    assert result.distribution.effective_sample_size == 0
    assert result.evidence


def test_prior_is_uniform_over_concrete_combos_not_abstract_classes():
    result = PreflopRfiRangePrior.from_builtin().lookup(
        _query(9, Position.UTG)
    )
    weights = tuple(result.distribution.combo_weights.values())

    assert max(weights) - min(weights) <= Decimal("2e-27")
    assert len(result.distribution.combo_weights) == sum(
        len(expand_hand_class(hand))
        for hand in PreflopRfiHeuristicProvider.from_builtin().explicit_range(
            9, Position.UTG
        )
    )


def test_known_card_blockers_remove_every_colliding_combo_and_renormalize():
    known = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.DIAMONDS))
    result = PreflopRfiRangePrior.from_builtin().lookup(
        _query(6, Position.CO, known_cards=known)
    )

    blocked = set(known)
    assert sum(result.distribution.combo_weights.values()) == 1
    assert all(
        not blocked.intersection(parse_concrete_combo(combo))
        for combo in result.distribution.combo_weights
    )


def test_all_cards_blocked_returns_unknown_not_random_range():
    deck = tuple(Card(rank, suit) for rank in Rank for suit in Suit)
    result = PreflopRfiRangePrior.from_builtin().lookup(
        _query(6, Position.CO, known_cards=deck)
    )

    assert result.state is RangePriorState.UNKNOWN
    assert result.distribution is None
    assert result.reasons == ("range_card_collision",)


@pytest.mark.parametrize("players", (2, 3, 4, 5, 7, 8))
def test_unsupported_player_count_never_inherits_neighbor_range(players):
    result = PreflopRfiRangePrior.from_builtin().lookup(
        _query(players, Position.BTN)
    )

    assert result.state is RangePriorState.NOT_APPLICABLE
    assert "unsupported_player_count" in result.reasons


@pytest.mark.parametrize(
    ("query", "reason"),
    (
        (_query(9, Position.BB), "unsupported_position"),
        (_query(6, Position.CO, stack="99"), "unsupported_stack"),
        (_query(6, Position.CO, action_line="call_raise"),
         "unsupported_action_line"),
    ),
)
def test_out_of_scope_position_stack_and_action_are_explicit(query, reason):
    result = PreflopRfiRangePrior.from_builtin().lookup(query)

    assert result.state is RangePriorState.NOT_APPLICABLE
    assert reason in result.reasons


def test_prior_source_version_and_evidence_pin_asset_and_conversion():
    prior = PreflopRfiRangePrior.from_builtin()
    result = prior.lookup(_query())

    assert "uniform-combo-prior/v1" in prior.source_version
    assert result.distribution.source == "preflopr-explicit-rfi-heuristic"
    assert result.distribution.source_version == prior.source_version
    assert any(item.startswith("asset_sha256:") for item in result.evidence)
    assert any(item.startswith("combo_expansion:uniform-combo-v1:")
               for item in result.evidence)


def test_lookup_is_deterministic_and_distribution_is_immutable():
    prior = PreflopRfiRangePrior.from_builtin()
    first = prior.lookup(_query())
    second = prior.lookup(_query())

    assert first == second
    with pytest.raises(TypeError):
        first.distribution.combo_weights["AsAh"] = Decimal("1")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seat_id": -1},
        {"player_count": 1},
        {"effective_stack_bb": Decimal("NaN")},
        {"action_line": ""},
        {"known_cards": (Card(Rank.ACE, Suit.SPADES),) * 2},
    ),
)
def test_invalid_query_is_rejected(kwargs):
    values = {
        "seat_id": 0,
        "player_count": 6,
        "position": Position.CO,
        "effective_stack_bb": Decimal("100"),
        "action_line": "open_raise",
        "known_cards": (),
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RangePriorQuery(**values)
