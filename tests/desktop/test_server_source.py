"""Command-line capture-source wiring for the local server.

``live.py`` retains historical ADB internals, but the product server exposes
only a physical UVC capture card so emulator input cannot be selected.
"""

import pytest

from poker_engine.desktop.app import _create_server
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


def test_capture_card_is_the_only_default_source(captured_run):
    main([])
    assert captured_run[0]["source"] == "capture-card"


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


def test_adb_emulator_source_is_rejected():
    with pytest.raises(SystemExit):
        main(["--source", "adb"])


def test_capture_card_is_the_only_advertised_choice():
    assert CAPTURE_SOURCES == ("capture-card",)


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


def test_stream_factory_defaults_to_capture_card(monkeypatch):
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

    assert seen["source"] == "capture-card"


def test_programmatic_stream_factory_rejects_adb():
    with pytest.raises(ValueError, match="physical capture-card"):
        build_stream_factory(source="adb")


def test_native_app_server_rejects_adb_before_startup():
    with pytest.raises(ValueError, match="physical capture-card"):
        _create_server("emulator-5554", source="adb")
