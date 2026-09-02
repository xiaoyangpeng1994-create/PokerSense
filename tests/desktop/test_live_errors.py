"""Live-stream error-boundary regression tests."""

from __future__ import annotations

import asyncio

import pytest

from poker_engine.desktop import live


def test_pipeline_initialization_error_becomes_recoverable_capture_error(
    monkeypatch,
):
    def fail_build(*args, **kwargs):
        raise RuntimeError("mss is not installed")

    monkeypatch.setattr(live, "build_pipeline", fail_build)

    async def read_first_item():
        stream = live.live_analysis_stream()
        return await anext(stream)

    with pytest.raises(
        live.LiveCaptureError,
        match="capture engine initialization failed: mss is not installed",
    ):
        asyncio.run(read_first_item())


def test_pipeline_step_error_becomes_recoverable_capture_error(monkeypatch):
    class BrokenPipeline:
        def step(self):
            raise RuntimeError("desktop capture failed")

    monkeypatch.setattr(
        live, "build_pipeline", lambda *args, **kwargs: BrokenPipeline()
    )

    async def read_first_item():
        stream = live.live_analysis_stream()
        return await anext(stream)

    with pytest.raises(
        live.LiveCaptureError,
        match="capture engine failed: desktop capture failed",
    ):
        asyncio.run(read_first_item())
