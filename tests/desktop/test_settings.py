"""Regression tests for durable desktop preferences."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from poker_engine.desktop import settings
from poker_engine.desktop.server import create_app


def test_language_preference_survives_a_reload(tmp_path, monkeypatch):
    path = tmp_path / "PokerSense" / "settings.json"
    monkeypatch.setattr(settings, "_settings_path", lambda: path)

    assert settings.load_settings() == {"language": "auto"}
    assert settings.save_language("zh") == {"language": "zh"}
    assert settings.load_settings() == {"language": "zh"}


def test_invalid_language_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_path", lambda: tmp_path / "settings.json")
    with pytest.raises(ValueError, match="unsupported language"):
        settings.save_language("fr")


def test_settings_endpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_path", lambda: tmp_path / "settings.json")
    client = TestClient(create_app())

    assert client.get("/settings").json() == {"language": "auto"}
    assert client.put("/settings", json={"language": "zh"}).json() == {
        "language": "zh"
    }
    assert client.get("/settings").json() == {"language": "zh"}
