"""Resource-location contracts for source and frozen desktop builds."""

from __future__ import annotations

from poker_engine.desktop import live


def test_resource_root_uses_pyinstaller_bundle_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(live.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert live._resource_root() == tmp_path


def test_source_resource_root_contains_committed_calibration():
    root = live._resource_root()
    assert (
        root / "configs" / "vision" / "wepoker_android" / "calibration.json"
    ).is_file()
