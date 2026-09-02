from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import ActionType, Rank, Street, Suit
from poker_engine.core.value_objects import ChipAmount, ChipDelta
from poker_engine.desktop.advice_view import advice_to_view
from poker_engine.desktop.serialize import DesktopFrame, analysis_to_dict
from poker_engine.core.value_objects import Card
from poker_engine.realtime.analysis import (
    ConfidenceSnapshot,
    EquitySnapshot,
    RealtimeAnalysis,
    StateSnapshot,
)
from poker_engine.strategy.advice import AdviceStatus, build_advice, mark_stale
from poker_engine.strategy.provider import (
    FakeProvider,
    LookupState,
    MatchDimension,
    MatchKind,
    ProviderResult,
)
from poker_engine.strategy.router import RouteResult, StrategyRouter
from poker_engine.strategy.safety import GateResult, GateStatus

from .helpers import NOW, candidate, capability, context, hit_result


ROOT = Path(__file__).resolve().parents[2]


def ready_advice():
    ctx = context()
    value = candidate(
        ctx,
        probabilities={
            ActionType.CHECK: Decimal("0.25"),
            ActionType.RAISE: Decimal("0.75"),
        },
    )
    provider = FakeProvider(
        value.provider_id,
        value.provider_version,
        capability((2,)),
        hit_result(value),
    )
    return build_advice(
        ctx, StrategyRouter((provider,)).route(ctx, now=NOW), now=NOW
    )


def test_ready_view_exposes_sorted_actions_and_identity():
    advice = ready_advice()
    view = advice_to_view(advice, now=NOW)
    assert view["status"] == "READY"
    assert view["show_actions"] is True
    assert [item["action"] for item in view["actions"]] == ["raise", "check"]
    assert [item["probability_exact"] for item in view["actions"]] == [
        "0.75", "0.25"
    ]
    assert view["identity"] == {
        "hand_id": advice.hand_id,
        "state_version": advice.state_version,
        "request_id": advice.request_id,
        "player_count": 2,
        "active_player_count": 2,
    }


def test_approximate_view_exposes_requested_matched_distance_and_score():
    ctx = context()
    dimension = MatchDimension(
        "pot_bb", "1.5", "1", Decimal("0.5"), Decimal("1")
    )
    value = candidate(
        ctx,
        match_kind=MatchKind.INTERPOLATED,
        score=0.5,
        match_dimensions=(dimension,),
    )
    advice = build_advice(
        ctx,
        RouteResult(LookupState.HIT_APPROXIMATE, value, ()),
        now=NOW,
    )

    view = advice_to_view(advice, now=NOW)

    assert view["match_dimensions"] == [{
        "name": "pot_bb",
        "requested": "1.5",
        "matched": "1",
        "distance": "0.5",
        "maximum_distance": "1",
        "score": 0.5,
    }]
    source = (ROOT / "ui" / "app.js").read_text()
    assert "advice.match_dimensions" in source
    assert 't("differences")' in source


def test_view_exposes_structured_hard_gate_results():
    ctx = context()
    advice = build_advice(
        ctx,
        RouteResult(LookupState.HIT_EXACT, candidate(ctx), ()),
        hard_gates=(GateResult("range_integrity", GateStatus.PASS),),
        now=NOW,
    )

    view = advice_to_view(advice, now=NOW)

    assert {
        "name": "range_integrity", "status": "PASS", "reasons": [],
    } in view["gate_results"]
    source = (ROOT / "ui" / "app.js").read_text()
    assert "advice.gate_results" in source
    assert 't("gates")' in source


def test_ready_view_carries_sizes_ev_source_match_and_evidence():
    advice = replace(
        ready_advice(),
        recommended_sizes={ActionType.RAISE: (ChipAmount("3"),)},
        action_ev={
            ActionType.CHECK: ChipDelta("0"),
            ActionType.RAISE: ChipDelta("1.25"),
        },
        ev_gap=ChipDelta("1.25"),
        assumptions=("test_assumption",),
    )
    view = advice_to_view(advice, now=NOW)
    raise_row = view["actions"][0]
    assert raise_row["sizes"] == ["3"]
    assert raise_row["ev"] == "1.25"
    assert view["ev_gap"] == "1.25"
    assert view["strategy_source"] == advice.strategy_source
    assert view["strategy_version"] == advice.strategy_version
    assert view["match_kind"] == "exact"
    assert view["assumptions"] == ["test_assumption"]
    assert view["input_provenance"] == [
        {
            "field_name": "hero_cards",
            "source": "vision",
            "status": "VALID",
            "confidence": 0.99,
        },
        {
            "field_name": "stacks",
            "source": "manual",
            "status": "VALID",
            "confidence": 1.0,
        },
    ]


def test_partial_view_shows_no_strategy_actions():
    ctx = context()
    route = RouteResult(
        LookupState.NO_STRATEGY,
        None,
        (ProviderResult(LookupState.NOT_FOUND, "none"),),
        ("strategy_unavailable",),
    )
    advice = build_advice(ctx, route, math_report={"equity": 0.52}, now=NOW)
    view = advice_to_view(advice, now=NOW)
    assert advice.status is AdviceStatus.PARTIAL
    assert view["show_actions"] is False
    assert view["actions"] == []
    assert view["rejection_reasons"] == ["strategy_unavailable"]


def test_abstain_view_never_exposes_actions_and_keeps_reasons():
    ctx = context(missing_fields=("hero_position",))
    advice = build_advice(
        ctx,
        RouteResult(
            LookupState.NO_STRATEGY, None, (), ("hero_position",)
        ),
        now=NOW,
    )
    view = advice_to_view(advice, now=NOW)
    assert view["status"] == "ABSTAIN"
    assert view["show_actions"] is False
    assert view["actions"] == []
    assert "hero_position" in view["rejection_reasons"]


def test_stale_view_never_exposes_actions():
    stale = mark_stale(ready_advice(), now=NOW, reason="new_state")
    view = advice_to_view(stale, now=NOW)
    assert view["status"] == "STALE"
    assert view["show_actions"] is False
    assert view["actions"] == []
    assert view["rejection_reasons"] == ["new_state"]


def test_view_enforces_expiry_even_if_caller_still_holds_ready_advice():
    advice = ready_advice()
    view = advice_to_view(advice, now=advice.expires_at + timedelta(microseconds=1))
    assert view["status"] == "STALE"
    assert view["actions"] == []
    assert view["rejection_reasons"] == ["expired_advice"]


def test_view_is_deterministic_at_same_time():
    advice = ready_advice()
    assert advice_to_view(advice, now=NOW) == advice_to_view(advice, now=NOW)


def test_view_rejects_wrong_type_and_naive_clock():
    with pytest.raises(TypeError, match="Advice"):
        advice_to_view(object(), now=NOW)
    with pytest.raises(TypeError, match="timezone-aware"):
        advice_to_view(ready_advice(), now=NOW.replace(tzinfo=None))


def test_ui_contains_all_advice_contract_targets_and_no_inline_html_sink():
    html = (ROOT / "ui/index.html").read_text()
    javascript = (ROOT / "ui/app.js").read_text()
    stylesheet = (ROOT / "ui/style.css").read_text()
    for element_id in (
        "advice-panel", "advice-status", "advice-confidence",
        "advice-actions", "advice-message", "advice-meta",
        "advice-badges", "advice-evidence-content",
    ):
        assert f'id="{element_id}"' in html
    assert "advice.show_actions" in javascript
    assert "source-${source}" in javascript
    assert ".advice-badge.source-manual" in stylesheet
    assert "innerHTML" not in javascript[javascript.index("function renderAdvice"):]


def test_analysis_wire_payload_embeds_ready_advice_view():
    analysis = RealtimeAnalysis(
        frame_seq=17,
        state=StateSnapshot(
            hand_id="h-2-preflop",
            state_version=1,
            street=Street.PREFLOP,
            hero_cards=(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
            board_cards=(),
            pot=ChipAmount("1.5"),
        ),
        equity=EquitySnapshot(win_rate=0.61, tie_rate=0.02),
        confidence=ConfidenceSnapshot(
            overall_confidence=0.95,
            field_status=(("hero_cards", "valid"),),
        ),
    )
    payload = analysis_to_dict(analysis, ready_advice(), now=NOW)
    assert payload["frame_seq"] == 17
    assert payload["state"]["hand_id"] == "h-2-preflop"
    assert payload["state"]["state_version"] == 1
    assert payload["advice"]["status"] == "READY"
    assert payload["advice"]["show_actions"] is True


def test_analysis_wire_payload_enforces_advice_expiry():
    advice = ready_advice()
    analysis = RealtimeAnalysis(
        frame_seq=18,
        state=StateSnapshot(
            hand_id="h-2-preflop",
            state_version=1,
            street=Street.PREFLOP,
            hero_cards=(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
            board_cards=(),
            pot=ChipAmount("1.5"),
        ),
        equity=EquitySnapshot(win_rate=0.61, tie_rate=0.02),
        confidence=ConfidenceSnapshot(
            overall_confidence=0.95,
            field_status=(("hero_cards", "valid"),),
        ),
    )
    payload = analysis_to_dict(
        analysis,
        advice,
        now=advice.expires_at + timedelta(microseconds=1),
    )
    assert payload["advice"]["status"] == "STALE"
    assert payload["advice"]["show_actions"] is False
    assert payload["advice"]["actions"] == []


def test_analysis_wire_payload_rejects_advice_from_another_state():
    advice = ready_advice()
    analysis = RealtimeAnalysis(
        frame_seq=19,
        state=StateSnapshot(
            hand_id=advice.hand_id,
            state_version=advice.state_version + 1,
            street=Street.PREFLOP,
            hero_cards=(
                Card(Rank.ACE, Suit.SPADES),
                Card(Rank.KING, Suit.HEARTS),
            ),
            board_cards=(),
            pot=ChipAmount("1.5"),
        ),
        equity=EquitySnapshot(win_rate=0.61, tie_rate=0.02),
        confidence=ConfidenceSnapshot(
            overall_confidence=0.95,
            field_status=(("hero_cards", "valid"),),
        ),
    )
    payload = analysis_to_dict(analysis, advice, now=NOW)
    assert payload["advice"]["status"] == "STALE"
    assert payload["advice"]["show_actions"] is False
    assert payload["advice"]["actions"] == []
    assert payload["advice"]["rejection_reasons"] == [
        "analysis_identity_mismatch"
    ]


def test_state_snapshot_rejects_invalid_wire_identity():
    with pytest.raises(ValueError, match="hand_id"):
        StateSnapshot(
            hand_id="",
            state_version=0,
            street=Street.PREFLOP,
            hero_cards=(),
            board_cards=(),
            pot=ChipAmount("0"),
        )
    with pytest.raises(TypeError, match="state_version"):
        StateSnapshot(
            hand_id="h",
            state_version=True,
            street=Street.PREFLOP,
            hero_cards=(),
            board_cards=(),
            pot=ChipAmount("0"),
        )


def test_websocket_sequence_refines_then_hides_old_state_advice():
    pytest.importorskip("fastapi")
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using `httpx` with `starlette.testclient`.*"
        )
        from fastapi.testclient import TestClient

    from poker_engine.desktop.server import create_app

    advice = replace(
        ready_advice(), expires_at=NOW + timedelta(days=3650)
    )
    refined = replace(
        advice,
        strategy_source="slow-resolver",
        strategy_version="v2",
        action_probabilities={
            ActionType.CHECK: Decimal("0.1"),
            ActionType.RAISE: Decimal("0.9"),
        },
    )

    def analysis(state_version: int) -> RealtimeAnalysis:
        return RealtimeAnalysis(
            frame_seq=state_version,
            state=StateSnapshot(
                hand_id=advice.hand_id,
                state_version=state_version,
                street=Street.PREFLOP,
                hero_cards=(
                    Card(Rank.ACE, Suit.SPADES),
                    Card(Rank.KING, Suit.HEARTS),
                ),
                board_cards=(),
                pot=ChipAmount("1.5"),
            ),
            equity=EquitySnapshot(win_rate=0.61, tie_rate=0.02),
            confidence=ConfidenceSnapshot(
                overall_confidence=0.95,
                field_status=(("hero_cards", "valid"),),
            ),
        )

    async def stream():
        yield DesktopFrame(analysis(advice.state_version), advice)
        yield DesktopFrame(analysis(advice.state_version), refined)
        yield DesktopFrame(analysis(advice.state_version + 1), refined)

    with TestClient(create_app(stream)) as client:
        with client.websocket_connect("/ws") as websocket:
            current = websocket.receive_json()
            upgraded = websocket.receive_json()
            updated = websocket.receive_json()

    assert current["advice"]["status"] == "READY"
    assert current["advice"]["show_actions"] is True
    assert upgraded["advice"]["status"] == "READY"
    assert upgraded["advice"]["strategy_version"] == "v2"
    assert upgraded["advice"]["actions"][0]["probability_exact"] == "0.9"
    assert updated["state"]["state_version"] == advice.state_version + 1
    assert updated["advice"]["status"] == "STALE"
    assert updated["advice"]["show_actions"] is False
    assert updated["advice"]["actions"] == []
