"""Tests for the stage A/B recording helpers (record.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.capture_card_calibration import record as rec


def _frame(w: int = 2, h: int = 3, value: int = 42) -> np.ndarray:
    """A small non-black BGR frame (avoids the black-frame signal path)."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class _FakeCap:
    """A minimal VideoCapture double that yields ``frames`` then fails."""

    def __init__(self, frames, *, opened=True, width=2, height=3, fps=30.0):
        self._frames = list(frames)
        self._opened = opened
        self._width = width
        self._height = height
        self._fps = fps
        self._sets = {}
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self._sets[int(prop)] = value
        return True

    def get(self, prop):
        names = {
            3: self._width,
            4: self._height,
            5: self._fps,
        }
        return names.get(int(prop), 0.0)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class _FakeWriter:
    written = 0
    released = False

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return True

    def write(self, frame):
        self.written += 1

    def release(self):
        self.released = True


def _fourcc_factory(*codec):
    return 0x01


# --- helper functions ------------------------------------------------------


def test_fourcc_to_str_decodes_known_code():
    # "YUY2" as a little-endian FOURCC int.
    code = ord("Y") | (ord("U") << 8) | (ord("Y") << 16) | (ord("2") << 24)
    assert rec._fourcc_to_str(code) == "YUY2"


def test_fourcc_to_str_strips_nul_and_unknown():
    assert rec._fourcc_to_str(0) == "unknown"


def test_is_black_frame_detects_black():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    assert rec._is_black_frame(black) is True
    assert rec._is_black_frame(_frame()) is False
    assert rec._is_black_frame(None) is True


def test_resolve_api_constant_returns_int():
    assert isinstance(rec.resolve_api_constant("MSMF"), int)
    assert isinstance(rec.resolve_api_constant("DSHOW"), int)
    assert isinstance(rec.resolve_api_constant("ANY"), int)


# --- probe_device ----------------------------------------------------------


def test_probe_device_reports_negotiated_params():
    cap = _FakeCap([_frame()], width=1920, height=1080, fps=29.97)
    info = rec.probe_device(video_capture_factory=lambda *a, **k: cap)
    assert info["frame_size"] == [1920, 1080]
    assert info["fps"] == round(29.97, 3)


def test_probe_device_fails_closed_when_cannot_open():
    cap = _FakeCap([], opened=False)
    with pytest.raises(rec.SchemaError):
        rec.probe_device(video_capture_factory=lambda *a, **k: cap)


def test_probe_device_fails_closed_when_no_frame():
    cap = _FakeCap([])
    with pytest.raises(rec.SchemaError):
        rec.probe_device(video_capture_factory=lambda *a, **k: cap)


# --- record_session --------------------------------------------------------


def _record(tmp_path, frames, **kwargs):
    kwargs.setdefault("stop_on_eof", True)
    return rec.record_session(
        tmp_path / "session_001.mkv",
        session_id="session_001",
        video_capture_factory=lambda *a, **k: _FakeCap(frames),
        video_writer_factory=_fourcc_factory,
        video_writer_class=_FakeWriter,
        **kwargs,
    )


def test_record_session_writes_frames(tmp_path):
    log = _record(tmp_path, [_frame(), _frame(), _frame()])
    assert log.session_id == "session_001"
    assert log.written_frames == 3
    assert log.width == 2
    assert log.height == 3
    # start + stop events at minimum.
    events = [e.event for e in log.events]
    assert "start" in events and "stop" in events


def test_record_session_marks_disconnect_then_reconnect(tmp_path):
    # Frame, a simulated unplug (read fails), then a frame again: the log must
    # record a disconnect followed by a reconnect.
    class _FlakyCap(_FakeCap):
        def __init__(self):
            super().__init__([_frame(), _frame()])
            self._read_count = 0

        def read(self):
            self._read_count += 1
            if self._read_count == 2:
                return False, None  # simulate unplug
            return super().read()

    log = rec.record_session(
        tmp_path / "session_001.mkv",
        session_id="session_001",
        stop_on_eof=False,
        video_capture_factory=lambda *a, **k: _FlakyCap(),
        video_writer_factory=_fourcc_factory,
        video_writer_class=_FakeWriter,
        max_seconds=0.25,
    )
    events = [e.event for e in log.events]
    assert "disconnect" in events
    assert "reconnect" in events


def test_record_session_never_writes_black_frame(tmp_path):
    frames = [_frame(), np.zeros((3, 2, 3), dtype=np.uint8), _frame()]
    log = _record(tmp_path, frames, black_frame_threshold_ms=0)
    # The black frame is skipped; only 2 non-black frames are written.
    assert log.written_frames == 2


def test_record_session_records_black_frame_event(tmp_path):
    # Two consecutive black frames with a zero threshold must log an event.
    frames = [
        np.zeros((3, 2, 3), dtype=np.uint8),
        np.zeros((3, 2, 3), dtype=np.uint8),
    ]
    log = _record(tmp_path, frames, black_frame_threshold_ms=0)
    assert "black_frame" in [e.event for e in log.events]


# --- session log serialization --------------------------------------------


def test_session_log_roundtrip(tmp_path):
    log = rec.CaptureSessionLog(session_id="session_001")
    log.events.append(rec.SignalEvent("start", 0, 0, "x"))
    rec.write_session_log(tmp_path / "log.json", log)
    data = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    restored = rec.CaptureSessionLog.from_dict(data)
    assert restored.session_id == log.session_id
    assert restored.events[0].event == "start"


# --- update_device_manifest ------------------------------------------------


def _write_minimal_manifest(root: Path) -> Path:
    from tools.capture_card_calibration.dataset import write_device_template

    return write_device_template(root / "source" / "device_and_capture.json")


def test_update_device_manifest_merges_uvc_and_recording(tmp_path):
    root = tmp_path / "calib"
    (root / "source").mkdir(parents=True)
    _write_minimal_manifest(root)

    updated = rec.update_device_manifest(
        root,
        uvc={"frame_size": [1920, 1080], "fps": 30, "pixel_format": "YUY2"},
        recording={"codec": "FFV1", "container": "mkv"},
        sessions=[{"id": "session_001", "raw": "session_001.mkv"}],
    )
    assert updated.uvc["pixel_format"] == "YUY2"
    assert updated.recording["codec"] == "FFV1"
    assert len(updated.sessions) == 1
    # Re-read from disk to confirm persistence.
    reloaded = json.loads(
        (root / "source" / "device_and_capture.json").read_text(encoding="utf-8")
    )
    assert reloaded["uvc"]["pixel_format"] == "YUY2"


def test_update_device_manifest_keeps_operator_fields(tmp_path):
    root = tmp_path / "calib"
    (root / "source").mkdir(parents=True)
    path = _write_minimal_manifest(root)
    # Operator fills in the phone model before recording.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phone"]["model"] = "Redmi K60"
    path.write_text(json.dumps(data), encoding="utf-8")

    updated = rec.update_device_manifest(root, uvc={"fps": 30})
    assert updated.phone["model"] == "Redmi K60"


def test_update_device_manifest_requires_existing_manifest(tmp_path):
    with pytest.raises(rec.SchemaError):
        rec.update_device_manifest(tmp_path / "nope", uvc={"fps": 30})
