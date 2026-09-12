from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from poker_engine.core.enums import Position
from poker_engine.core.request_context import RequestContext
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa8_shadow import (
    AA8ShadowStatus,
    assess_aa8_shadow_input,
    build_aa8_shadow_context,
)
from poker_engine.strategy.contracts import GameConfig, GameType
from poker_engine.strategy.heuristic_provider import PreflopRfiHeuristicProvider
from poker_engine.strategy.provider import LookupState
from poker_engine.strategy.router import StrategyRouter


NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def config(count=8, *, ante="0", rake="0"):
    return GameConfig(
        "NLHE", GameType.CASH, 8, count,
        ChipAmount("2"), ChipAmount("4"),
        ante=ChipAmount(ante), rake_percent=Decimal(rake),
        rake_cap=ChipAmount("8"), minimum_chip=ChipAmount("1"),
    )


def request(hand="observed_deal_100", version=10):
    return RequestContext(
        hand, version, "shadow-request", NOW,
        expires_at=NOW + timedelta(seconds=2), deadline_ms=300,
    )


def row(count=8, *, actor=4):
    occupied = set(range(count))
    participants = {
        str(seat): {"state": "active" if seat in occupied else "empty"}
        for seat in range(8)
    }
    wagers = {str(seat): "0" for seat in range(8)}
    wagers.update({"0": "2", "1": "4"})
    commitments = dict(wagers)
    stacks = {
        str(seat): {"value": "400" if seat in occupied else None}
        for seat in range(8)
    }
    authority = {
        "schema_version": 1,
        "source_frame": 100,
        "source_sha256": "a" * 64,
        "confidence": .99,
        "evidence_ref": "replay://aa8/base/100",
        "fields": {
            name: True for name in (
                "hand_boundary", "cards", "pot", "stacks", "street_wagers",
                "hand_commitments", "actions", "participation", "dealer",
                "action_line",
            )
        },
    }
    return {
        "frame": 100,
        "source_sha256": "a" * 64,
        "scene_supported": True,
        "visual_scope": {
            "scope_id": "aa8_base_visual_v1",
            "status": "BASE_VISUAL_PASS",
            "normal_mode_verified": True,
            "observed_deferred_modes": [],
        },
        "strategy_input_authority": authority,
        "observed_state_v2": {
            "observed_epoch": "observed_deal_100",
            "street_candidate": "preflop",
            "board_candidate": [],
            "participants": participants,
            "authoritative_hand_boundary": True,
            "complete_legal_state": True,
        },
        "cards": {"hero": ["As", "Kd"]},
        "pot": {"value": "6"},
        "stacks": stacks,
        "current_actor": actor,
        "causal_street_wagers_v2": {
            "wagers": wagers, "canonical_verified": True,
        },
        "hand_commitments": commitments,
        "hand_commitments_canonical_verified": True,
        "actions_complete_and_canonical_verified": True,
        "dealer_seat": 0,
        "dealer_seat_canonical_verified": True,
        "canonical_state_version": 10,
        "action_line": "unopened",
    }


@pytest.mark.parametrize("count", (6, 7, 8))
def test_builds_reusable_six_to_eight_player_context(count):
    result = build_aa8_shadow_context(row(count), request(), config(count))
    assert result.status is AA8ShadowStatus.READY
    assert result.context.is_decision_ready
    assert result.context.game_config.dealt_player_count == count
    seats = {seat.seat_id: seat for seat in result.context.seats}
    assert seats[0].position is Position.BTN
    assert seats[4].is_hero
    assert result.state.to_call == ChipAmount("4")
    assert sum(pot.amount.value for pot in result.context.pots) == Decimal("6")


def test_nonhero_actor_builds_state_but_waits_without_advice():
    result = build_aa8_shadow_context(row(actor=3), request(), config())
    assert result.status is AA8ShadowStatus.WAITING
    assert result.reasons == ("hero_not_actor",)
    assert not result.context.is_decision_ready


def test_current_candidate_output_cannot_be_promoted():
    value = row()
    value["visual_scope"].update(
        status="BASE_CANDIDATE_UNVERIFIED", normal_mode_verified=False,
    )
    value["strategy_input_authority"] = {}
    value["observed_state_v2"].update(
        authoritative_hand_boundary=False, complete_legal_state=False,
    )
    value["causal_street_wagers_v2"]["canonical_verified"] = False
    result = build_aa8_shadow_context(value, request(), config())
    assert result.status is AA8ShadowStatus.ABSTAIN
    assert "base_visual_scope_not_verified" in result.reasons
    assert "legal_state_incomplete" in result.reasons
    assert "street_wagers_not_canonical" in result.reasons
    assert result.state is None and result.context is None


def test_deferred_special_mode_wins_over_other_blockers():
    value = row()
    value["visual_scope"]["observed_deferred_modes"] = ["insurance"]
    value["strategy_input_authority"] = {}
    result = build_aa8_shadow_context(value, request(), config())
    assert result.status is AA8ShadowStatus.DEFERRED_SPECIAL_MODE
    assert "deferred_special_mode:insurance" in result.reasons
    assert result.context is None


@pytest.mark.parametrize(
    "change", ("pot", "dealer", "version", "action_line", "negative_stack")
)
def test_inconsistent_canonical_payload_abstains(change):
    value = row()
    if change == "pot":
        value["pot"]["value"] = "7"
    elif change == "dealer":
        value["dealer_seat"] = 9
    elif change == "version":
        value["canonical_state_version"] = 11
    elif change == "negative_stack":
        value["stacks"]["3"]["value"] = "-1"
    else:
        value["action_line"] = None
    result = build_aa8_shadow_context(value, request(), config())
    assert result.status is AA8ShadowStatus.ABSTAIN
    assert result.reasons[0].startswith("invalid_canonical_strategy_input:")


def test_authority_source_identity_must_match_frame():
    value = row()
    value["strategy_input_authority"]["source_frame"] = 99
    reasons = assess_aa8_shadow_input(value, config())
    assert "strategy_authority_frame_mismatch" in reasons


def test_existing_rfi_provider_is_reusable_but_not_for_ante_rake_profile():
    base = build_aa8_shadow_context(row(), request(), config()).context
    provider = PreflopRfiHeuristicProvider.from_builtin()
    route = StrategyRouter((provider,)).route(base, now=NOW)
    assert route.state in (LookupState.HIT_EXACT, LookupState.HIT_APPROXIMATE)
    priced = build_aa8_shadow_context(
        row(), request(), config(ante="2", rake="0.03"),
    ).context
    route = StrategyRouter((provider,)).route(priced, now=NOW)
    assert route.state is LookupState.NO_STRATEGY
    assert any(result.state is LookupState.NOT_APPLICABLE
               for result in route.provider_results)
