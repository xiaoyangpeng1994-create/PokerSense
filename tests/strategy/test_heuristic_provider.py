"""Golden and contract tests for the reviewed PreflopR heuristic asset."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from importlib import resources

import pytest

from poker_engine.core.enums import ActionType, Position, Street
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import LegalAction
from poker_engine.strategy.heuristic_provider import (
    ASSET_NAME,
    PreflopRfiHeuristicProvider,
)
from poker_engine.strategy.provider import LookupState, MatchKind
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, card, context


RANKS = "AKQJT98765432"
POSITIONS = {
    "6_UTG": Position.UTG,
    "6_HJ": Position.HJ,
    "6_CO": Position.CO,
    "6_BTN": Position.BTN,
    "6_SB": Position.SB,
    "9_UTG": Position.UTG,
    "9_UTG+1": Position.UTG1,
    "9_UTG+2": Position.UTG2,
    "9_MP": Position.LJ,
    "9_HJ": Position.HJ,
    "9_CO": Position.CO,
    "9_BTN": Position.BTN,
    "9_SB": Position.SB,
}


def _payload():
    asset = resources.files("poker_engine.strategy.assets").joinpath(ASSET_NAME)
    return json.loads(asset.read_text(encoding="utf-8"))


def _all_classes():
    values = []
    for first_index, first in enumerate(RANKS):
        values.append(first * 2)
        for second in RANKS[first_index + 1:]:
            values.extend((first + second + "s", first + second + "o"))
    return tuple(values)


def _cards(hand: str):
    if len(hand) == 2:
        return card(hand[0] + "s"), card(hand[1] + "h")
    second_suit = "s" if hand[2] == "s" else "h"
    return card(hand[0] + "s"), card(hand[1] + second_suit)


def _rfi_context(players: int, position: Position, hand: str = "AKo"):
    value = context(players)
    hero = next(seat for seat in value.seats if seat.seat_id == value.hero_seat)
    occupant = next(
        (seat for seat in value.seats if seat.position is position),
        None,
    )
    seats = []
    for seat in value.seats:
        new_position = seat.position
        if seat.seat_id == hero.seat_id:
            new_position = position
        elif occupant is not None and seat.seat_id == occupant.seat_id:
            new_position = hero.position
        seats.append(replace(
            seat,
            position=new_position,
            is_dealer=new_position is Position.BTN,
        ))
    return replace(
        value,
        seats=tuple(seats),
        hero_cards=_cards(hand),
        legal_actions=(
            LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("100")),
        ),
    )


def test_builtin_provider_declares_only_reviewed_scope_and_provenance():
    provider = PreflopRfiHeuristicProvider.from_builtin()

    # 7/8-handed keys are derived, not authored upstream: they reach the
    # router through derived_ranges rather than through player_counts alone.
    assert provider.capability.player_counts == frozenset({6, 7, 8, 9})
    assert provider.capability.streets == frozenset({Street.PREFLOP})
    assert provider.capability.stack_buckets_bb == (Decimal("100"),)
    assert provider.capability.action_lines == frozenset({"unopened"})
    assert provider.capability.base_match_kind is MatchKind.HEURISTIC
    assert "aed511d0451aea33a14f7e9204595fc2211f233f" in provider.source_version


@pytest.mark.parametrize("range_key", sorted(POSITIONS))
def test_every_169_class_matches_committed_explicit_range(range_key):
    provider = PreflopRfiHeuristicProvider.from_builtin()
    payload = _payload()
    expected_raise = frozenset(payload["ranges"][range_key])
    players = int(range_key.split("_", 1)[0])
    position = POSITIONS[range_key]

    for hand in _all_classes():
        result = provider.query(_rfi_context(players, position, hand))
        assert result.state is LookupState.HIT_APPROXIMATE
        assert result.candidate is not None
        assert result.candidate.match_kind is MatchKind.HEURISTIC
        expected = Decimal("1") if hand in expected_raise else Decimal("0")
        assert result.candidate.action_probabilities[ActionType.RAISE] == expected
        assert result.candidate.action_probabilities[ActionType.FOLD] == 1 - expected


def test_candidate_discloses_heuristic_limits_without_inventing_size_or_ev():
    provider = PreflopRfiHeuristicProvider.from_builtin()
    value = provider.query(_rfi_context(9, Position.UTG, "AA")).candidate

    assert value is not None
    assert value.confidence == 0.4
    assert value.recommended_sizes == {}
    assert value.action_ev == {}
    assert "heuristic_open_raise_chart_not_solver_derived" in value.assumptions
    assert "no_raise_size_or_ev" in value.assumptions
    assert any(item.startswith("source_sha256:") for item in value.evidence)
    assert any(item == "explicit_range:9_UTG:AA" for item in value.evidence)


# 7/8-handed tables: coverage exists, rebuilt from the same-named 9-handed key.
DERIVED_POSITIONS = {
    "7_UTG": (7, Position.UTG),
    "7_MP": (7, Position.LJ),
    "7_HJ": (7, Position.HJ),
    "7_CO": (7, Position.CO),
    "7_BTN": (7, Position.BTN),
    "7_SB": (7, Position.SB),
    "8_UTG": (8, Position.UTG),
    "8_UTG+1": (8, Position.UTG1),
    "8_MP": (8, Position.LJ),
    "8_HJ": (8, Position.HJ),
    "8_CO": (8, Position.CO),
    "8_BTN": (8, Position.BTN),
    "8_SB": (8, Position.SB),
}


@pytest.mark.parametrize("range_key", sorted(DERIVED_POSITIONS))
def test_every_169_class_of_a_derived_key_matches_its_source(range_key):
    """A derived key must reproduce its 9-handed source exactly, class by class.

    This pins the substitution itself. If someone repoints a derived key at a
    different chart -- reintroducing upstream's last-resort trick -- these
    169 comparisons move.
    """
    provider = PreflopRfiHeuristicProvider.from_builtin()
    payload = _payload()
    source_key = payload["derived_ranges"][range_key]
    expected_raise = frozenset(payload["ranges"][source_key])
    players, position = DERIVED_POSITIONS[range_key]

    for hand in _all_classes():
        result = provider.query(_rfi_context(players, position, hand))
        assert result.state is LookupState.HIT_APPROXIMATE
        expected = Decimal("1") if hand in expected_raise else Decimal("0")
        assert result.candidate.action_probabilities[ActionType.RAISE] == expected
        assert result.candidate.action_probabilities[ActionType.FOLD] == 1 - expected


@pytest.mark.parametrize("range_key", sorted(DERIVED_POSITIONS))
def test_derived_candidate_is_labelled_and_down_weighted(range_key):
    """Derived advice must be distinguishable from authored advice."""
    provider = PreflopRfiHeuristicProvider.from_builtin()
    payload = _payload()
    source_key = payload["derived_ranges"][range_key]
    players, position = DERIVED_POSITIONS[range_key]

    value = provider.query(_rfi_context(players, position, "AA")).candidate

    assert value is not None
    assert value.confidence == pytest.approx(0.3)
    assert f"derived_range:{range_key}<-{source_key}:AA" in value.evidence
    assert "7_and_8_handed_derived_from_same_named_9_handed_ranges" in (
        value.assumptions
    )
    assert "derived_range_is_tighter_than_true_table_size" in value.assumptions


def test_authored_advice_keeps_full_confidence_and_no_derivation_label():
    """The 6/9-handed path must be unchanged by the 7/8-handed addition."""
    provider = PreflopRfiHeuristicProvider.from_builtin()

    value = provider.query(_rfi_context(6, Position.UTG, "AA")).candidate

    assert value is not None
    assert value.confidence == 0.4
    assert "explicit_range:6_UTG:AA" in value.evidence
    assert not any(item.startswith("derived_range:") for item in value.evidence)


def test_derived_range_can_only_point_at_the_same_named_9_handed_key():
    """Repointing a derived key must abort loading instead of widening quietly.

    Upstream's fallback ends at ``9_BTN`` -- the widest chart in the asset.
    Accepting that substitution would render, say, an 8-handed UTG open as a
    button open, so the loader refuses any target but the same-named key.
    """
    payload = _payload()
    payload["derived_ranges"]["7_UTG"] = "9_BTN"

    with pytest.raises(ValueError, match="must read 9_UTG"):
        PreflopRfiHeuristicProvider(payload, asset_sha256="0" * 64)


def test_derived_ranges_are_published_for_inspection():
    provider = PreflopRfiHeuristicProvider.from_builtin()

    assert provider.derived_ranges["7_UTG"] == "9_UTG"
    assert provider.derived_ranges["8_UTG+1"] == "9_UTG+1"
    # Authored keys are never listed as derived.
    assert "6_UTG" not in provider.derived_ranges


@pytest.mark.parametrize("players", (2, 3, 4, 5))
def test_synthetic_or_unsupported_player_counts_are_never_claimed(players):
    """3-5 handed still has no evidence-backed chart, so it stays closed.

    7/8 handed moved to derived coverage (see above); 2-5 handed has no
    same-named 9-handed key for every position and is not a table size the
    owner plays, so it must still refuse rather than interpolate.
    """
    provider = PreflopRfiHeuristicProvider.from_builtin()
    result = provider.query(context(players))

    assert result.state is LookupState.NOT_APPLICABLE
    assert "unsupported_player_count" in result.reasons


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (lambda value: replace(value, street=Street.FLOP,
                               board_cards=(card("2c"), card("7d"), card("Jh"))),
         "unsupported_street"),
        (lambda value: replace(value, effective_stack_bb=Decimal("99")),
         "unsupported_stack"),
        (lambda value: replace(value, action_line="raise"),
         "unsupported_action_line"),
        (lambda value: replace(value, legal_actions=(
            LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("100")),
        )), "legal_actions_do_not_support_raise_fold"),
    ),
)
def test_out_of_scope_contexts_fail_closed(mutation, reason):
    provider = PreflopRfiHeuristicProvider.from_builtin()
    result = provider.query(mutation(_rfi_context(6, Position.CO)))

    assert result.state is LookupState.NOT_APPLICABLE
    assert reason in result.reasons


def test_big_blind_is_not_mapped_to_upstream_last_resort_button_range():
    provider = PreflopRfiHeuristicProvider.from_builtin()
    result = provider.query(_rfi_context(9, Position.BB, "72o"))

    assert result.state is LookupState.NOT_APPLICABLE
    assert result.reasons == ("unsupported_open_raise_position",)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("license", "unknown"),
        ("schema_version", 2),
        ("limitations", []),
        ("source_sha256", "bad"),
    ),
)
def test_asset_metadata_corruption_is_rejected(field, value):
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError):
        PreflopRfiHeuristicProvider(payload, asset_sha256="0" * 64)


def test_exact_provider_still_wins_over_heuristic_candidate():
    heuristic = PreflopRfiHeuristicProvider.from_builtin()
    ctx = _rfi_context(6, Position.CO)
    exact_result = candidate(
        ctx,
        provider_id="exact-6max",
        match_kind=MatchKind.EXACT,
        probabilities={ActionType.FOLD: Decimal("1")},
    )

    from poker_engine.strategy.provider import FakeProvider, ProviderResult

    exact = FakeProvider(
        "exact-6max",
        "v1",
        capability(player_counts=(6,), match_kind=MatchKind.EXACT, priority=999),
        ProviderResult(LookupState.HIT_EXACT, "exact-6max", exact_result),
    )
    route = StrategyRouter((heuristic, exact)).route(ctx, now=NOW)

    assert route.selected is exact_result
    assert route.state is LookupState.HIT_EXACT
