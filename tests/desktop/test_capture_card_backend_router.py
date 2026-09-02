"""Tests for live.py capture-backend routing (ADB vs capture-card).

These test only the *selection* logic — which backend class gets instantiated
for a given ``source``. They monkeypatch the backend constructors to avoid any
real ADB device / UVC capture card, and assert that an unknown source fails
closed.
"""

from __future__ import annotations

import pytest

from poker_engine.desktop import live


def _defuse_adb_backend(monkeypatch, captured):
    """Stub AdbBackend so construction needs no ADB binary."""
    import poker_engine.perceptual.capture.adb_backend as mod

    def fake_init(self):
        captured["adb"] = True

    monkeypatch.setattr(mod, "AdbBackend", type("FakeAdb", (), {"__init__": fake_init}))


def _defuse_capture_card(monkeypatch, captured):
    """Stub CaptureCardBackend so construction needs no UVC device."""
    import poker_engine.perceptual.capture.capture_card_backend as mod

    def fake_init(self, **kwargs):
        captured["card"] = kwargs

    monkeypatch.setattr(
        mod, "CaptureCardBackend", type("FakeCard", (), {"__init__": fake_init})
    )


def test_unknown_source_fails_closed():
    with pytest.raises(live.LiveCaptureError, match="unknown capture source"):
        live.build_capture_backend("bogus")


def test_adb_soure_returns_adb_backend(monkeypatch):
    captured = {}
    _defuse_adb_backend(monkeypatch, captured)
    backend = live.build_capture_backend("adb")
    assert captured["adb"] is True
    assert backend is not None


def test_capture_card_source_passes_options_through(monkeypatch):
    captured = {}
    _defuse_capture_card(monkeypatch, captured)
    backend = live.build_capture_backend(
        "capture-card",
        device_index=3,
        api="MSMF",
        normalization="norm-config",
    )
    assert captured["card"]["device_index"] == 3
    assert captured["card"]["api"] == "MSMF"
    assert captured["card"]["normalization"] == "norm-config"
    assert backend is not None


def test_normalization_ignored_for_adb(monkeypatch):
    # The ADB path ignores the normalization kwarg entirely.
    captured = {}
    _defuse_adb_backend(monkeypatch, captured)
    backend = live.build_capture_backend("adb", normalization="should-be-ignored")
    assert captured["adb"] is True
    assert backend is not None
