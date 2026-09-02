"""Wire-contract tests for the engine -> UI JSON payload."""

from __future__ import annotations

import cv2
import pytest

from pathlib import Path

from poker_engine.desktop.serialize import analysis_to_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CARD_FIXTURES = _REPO_ROOT / "tests" / "vision" / "fixtures" / "wepoker"

# The hero cards the assembled frame contains, in slot order.
_HERO = ("QD", "TS")


def _build_table_frame_image():
    """A 1440x2560 ADB frame carrying real card art at calibrated coordinates.

    Assembled rather than committed as a screenshot: a raw capture of a live
    table also contains browser chrome, bookmarks and another player's
    username, none of which belongs in this repository. The card crops are
    genuine captures and the paste positions come from the committed
    TableMap, so the geometry under test is the real thing.
    """
    import json

    import numpy as np

    from poker_engine.perceptual.vision.card_layout import hero_layout_from_dict
    from poker_engine.perceptual.vision.table_map import TableMap

    table_map = TableMap.from_json((
        _REPO_ROOT / "configs" / "platform"
        / "wepoker_android__ldplayer_portrait_1440x2560.json"
    ).read_text())
    width, height = table_map.reference_size
    # Table felt, sampled from a real capture.
    image = np.full((height, width, 3), (74, 110, 61), dtype=np.uint8)

    hero_roi = next(r for r in table_map.rois if r.kind.value == "hero_cards")
    hero_layout = hero_layout_from_dict(
        json.loads(
            (_REPO_ROOT / "configs" / "vision" / "wepoker_android"
             / "hero_slot_layout.json").read_text()
        )
    )
    rx, ry = int(hero_roi.x * width), int(hero_roi.y * height)
    rw, rh = int(hero_roi.width * width), int(hero_roi.height * height)
    for label, sub in zip(_HERO, hero_layout.slots):
        card = cv2.imread(str(_CARD_FIXTURES / f"{label}.png"))
        assert card is not None, f"card fixture missing: {label}"
        x0, y0 = rx + int(sub.x * rw), ry + int(sub.y * rh)
        target_w, target_h = int(sub.width * rw), int(sub.height * rh)
        card = cv2.resize(card, (target_w, target_h))
        image[y0:y0 + target_h, x0:x0 + target_w] = card
    return image


@pytest.fixture(scope="module")
def live_analysis():
    """A RealtimeAnalysis produced by the real pipeline from real card art.

    Uses the committed calibration and the production pipeline, so this
    exercises the same path the desktop app runs — no scripted stand-in.
    """
    from datetime import datetime, timezone

    from poker_engine.desktop.live import (
        _seed_state,
        build_confidence_gate,
        load_calibration,
        load_measured_calibrations,
    )
    from poker_engine.memory.hand_memory import InMemoryHandMemory
    from poker_engine.orchestrator import ApplicationOrchestrator
    from poker_engine.perceptual.capture.base import Frame, WindowRect
    from poker_engine.realtime.frame_source import SyntheticFrameSource
    from poker_engine.realtime.pipeline import RealtimePipeline
    from poker_engine.state_engine.engine import StateEngine

    image = _build_table_frame_image()
    height, width = image.shape[:2]
    frame = Frame(
        frame_seq=0,
        timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
        window_id="emulator-5556",
        window_rect=WindowRect(0, 0, width, height),
        image=image,
        width=width,
        height=height,
    )
    table_map, vision = load_calibration()
    measured = load_measured_calibrations(
        _REPO_ROOT / "configs" / "vision" / "wepoker_android"
    )
    orchestrator = ApplicationOrchestrator(
        StateEngine(), InMemoryHandMemory(), build_confidence_gate(measured)
    )
    orchestrator.start_hand(_seed_state())
    pipeline = RealtimePipeline(
        SyntheticFrameSource((frame,)), vision, table_map, orchestrator
    )
    step = pipeline.step()
    assert step is not None
    return step.analysis


def test_recognizes_hero_cards_from_real_capture(live_analysis):
    """Real card art at real coordinates must survive the whole pipeline."""
    assert [str(c) for c in live_analysis.state.hero_cards] == ["Qd", "Ts"]


def test_equity_is_computed_for_the_recognized_hand(live_analysis):
    equity = live_analysis.equity
    assert 0.0 < equity.win_rate < 1.0
    assert 0.0 <= equity.tie_rate < 1.0


def test_calibrated_and_uncalibrated_fields_are_separated(live_analysis):
    """Android board/street are measured; remaining table fields stay closed."""
    status = dict(live_analysis.confidence.field_status)
    assert status["hero_cards"] == "valid"
    assert status["board_cards"] == "valid"
    assert status["street"] == "valid"
    for field in ("pot", "stacks", "bet_size", "action"):
        assert status[field] == "unknown", f"{field} claimed without calibration"


def test_analysis_to_dict_matches_wire_contract(live_analysis):
    payload = analysis_to_dict(live_analysis)
    assert payload["state"]["hero_cards"] == ["Qd", "Ts"]
    assert isinstance(payload["state"]["pot"], str)
    assert isinstance(payload["equity"]["win_rate"], float)
    assert isinstance(payload["confidence"]["field_status"], list)


def test_analysis_to_dict_is_json_serializable(live_analysis):
    import json

    json.dumps(analysis_to_dict(live_analysis))


def test_analysis_to_dict_rejects_wrong_type():
    with pytest.raises(TypeError):
        analysis_to_dict({"not": "an analysis"})


def test_create_app_serves_ui_and_ws_route():
    pytest.importorskip("fastapi")
    from poker_engine.desktop.server import create_app

    app = create_app()
    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/ws" in routes
