"""Desktop-shell startup regression tests."""

from __future__ import annotations

import pytest

from poker_engine.desktop import app


class _Server:
    def __init__(self, started: bool) -> None:
        self.started = started


class _Thread:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_wait_for_server_returns_only_after_uvicorn_started():
    app._wait_for_server(_Server(started=True), _Thread(alive=True), 0.01)


def test_wait_for_server_reports_early_server_exit():
    with pytest.raises(RuntimeError, match="stopped during startup"):
        app._wait_for_server(_Server(started=False), _Thread(alive=False), 0.01)


def test_wait_for_server_reports_timeout(monkeypatch):
    times = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr(app.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(app.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="did not start"):
        app._wait_for_server(_Server(started=False), _Thread(alive=True), 0.01)
