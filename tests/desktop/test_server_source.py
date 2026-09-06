"""Command-line capture-source wiring for the local server.

``live.py`` already knows how to build an ADB or a UVC capture-card pipeline;
these tests pin the server layer that exposes the choice, so the capture-card
path cannot silently regress back to ADB-only.
"""

import pytest

from poker_engine.desktop.server import (
    CAPTURE_SOURCES,
    build_stream_factory,
    main,
)


@pytest.fixture
def captured_run(monkeypatch):
    """Capture the keyword arguments ``main`` forwards to ``run``."""
    calls = []
    monkeypatch.setattr(
        "poker_engine.desktop.server.run",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


def test_adb_is_the_default_source(captured_run):
    main([])
    assert captured_run[0]["source"] == "adb"


def test_capture_card_source_is_selectable(captured_run):
    main(["--source", "capture-card"])
    assert captured_run[0]["source"] == "capture-card"


def test_capture_card_options_reach_run(captured_run):
    main([
        "--source", "capture-card",
        "--device-index", "2",
        "--api", "DSHOW",
        "--port", "9001",
    ])
    call = captured_run[0]
    assert call["device_index"] == 2
    assert call["api"] == "DSHOW"
    assert call["port"] == 9001


def test_unknown_source_is_rejected():
    with pytest.raises(SystemExit):
        main(["--source", "telepathy"])


def test_capture_card_is_an_advertised_choice():
    assert "capture-card" in CAPTURE_SOURCES


def test_stream_factory_forwards_source_to_live_stream(monkeypatch):
    seen = {}

    def fake_stream(device_serial, interval_seconds=1.0, source="adb", **kwargs):
        seen["device_serial"] = device_serial
        seen["source"] = source
        seen["kwargs"] = kwargs

        async def _empty():
            if False:  # pragma: no cover - generator, never iterated here
                yield None

        return _empty()

    monkeypatch.setattr(
        "poker_engine.desktop.server.live_analysis_stream", fake_stream
    )

    factory = build_stream_factory(
        device_serial="emulator-5554",
        source="capture-card",
        device_index=3,
        api="MSMF",
    )
    factory()

    assert seen["device_serial"] == "emulator-5554"
    assert seen["source"] == "capture-card"
    assert seen["kwargs"]["device_index"] == 3
    assert seen["kwargs"]["api"] == "MSMF"


def test_stream_factory_defaults_to_adb(monkeypatch):
    seen = {}

    def fake_stream(device_serial, interval_seconds=1.0, source="adb", **kwargs):
        seen["source"] = source

        async def _empty():
            if False:  # pragma: no cover
                yield None

        return _empty()

    monkeypatch.setattr(
        "poker_engine.desktop.server.live_analysis_stream", fake_stream
    )

    build_stream_factory()()

    assert seen["source"] == "adb"
