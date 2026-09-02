"""Tests for tools.capture_card_calibration.sampler (stage D).

These use synthetic frames and an injected reader, so they run without a real
capture card or a large MKV, while still exercising the sampling, dedup, scene
classification and manifest emission paths.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig

from tools.capture_card_calibration.dataset import read_frame_manifest
from tools.capture_card_calibration.sampler import (
    SampleOptions,
    _dhash,
    _hamming,
    _is_black,
    _is_table,
    classify_scene,
    sample_session,
)


def _table_frame(size=(1080, 1920)) -> np.ndarray:
    """A synthetic raw UVC frame (1920x1080) with a green-felt content strip."""
    return _raw_frame(size)


def _blank_frame(size=(1080, 1920)) -> np.ndarray:
    return np.zeros((size[0], size[1], 3), dtype=np.uint8)


def _textured_frame(size=(1080, 498)) -> np.ndarray:
    """A high-contrast textured normalized frame so dHash can distinguish it."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        img[:, x, 0] = (x * 255) // w
        img[:, x, 1] = (x * 128) // w
    cv2.rectangle(img, (50, 50), (250, 300), (255, 255, 255), -1)
    return img


def _cfg() -> NormalizationConfig:
    return NormalizationConfig(
        rotate_degrees=0,
        crop_after_rotation=(711, 0, 1209, 1080),
        output_size=(498, 1080),
        source_size=(1920, 1080),
        version="capture-card-normalization-v1",
    )


def _raw_frame(size=(1080, 1920)) -> np.ndarray:
    """Simulate a raw UVC frame (1920x1080), which the normalizer crops to
    498x1080. Green felt fills the content column."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, 711:1209] = (60, 140, 60)  # green content strip, black letterbox
    return img


class _SeqReader:
    """Sequential synthetic reader: returns frames 0..N-1 then EOF."""

    def __init__(self, frameset: list[np.ndarray], fps: float = 30.0) -> None:
        self.frameset = frameset
        self.fps = fps
        self.max_frames = len(frameset)

    def __call__(self, frame_index: int):
        if frame_index >= len(self.frameset):
            return False, None, float(frame_index / self.fps * 1000.0)
        return True, self.frameset[frame_index], float(frame_index / self.fps * 1000.0)


def test_dhash_and_hamming_identical():
    img = _table_frame()
    d1 = _dhash(img)
    d2 = _dhash(img)
    assert _hamming(d1, d2) == 0


def test_dhash_hamming_large_for_different_frames():
    a = _textured_frame()
    b = cv2.flip(_textured_frame(), 1)  # mirrored differs
    d1 = _dhash(a)
    d2 = _dhash(b)
    assert _hamming(d1, d2) > 0


def test_is_table_green():
    assert _is_table(
        cv2.cvtColor(_table_frame(), cv2.COLOR_BGR2GRAY),
        _table_frame(),
        green_min=0.10,
    )


def test_is_black_blank():
    assert _is_black(cv2.cvtColor(_blank_frame(), cv2.COLOR_BGR2GRAY), black_max=4.0)


def test_classify_table():
    frame = _table_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert classify_scene(gray, frame, green_min=0.10).value == "table"


def test_classify_black_signal_loss():
    frame = _blank_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert classify_scene(gray, frame, green_min=0.10).value == "signal_loss"


def test_classify_menu_nongame():
    frame = np.full((1080, 498, 3), 250, dtype=np.uint8)  # white-ish
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert classify_scene(gray, frame, green_min=0.10).value == "menu"


def test_sample_writes_manifest(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "source" / "raw").mkdir(parents=True)
    cfg = _cfg()
    frameset = [_table_frame() for _ in range(100)]
    reader = _SeqReader(frameset)
    written, entries = sample_session(
        root,
        "session_001",
        normalization_config=cfg,
        options=SampleOptions(stable_interval_ms=700, event_context_frames=1),
        reader=reader,
    )
    assert written > 0
    assert entries == written
    manifest = read_frame_manifest(root / "normalized" / "manifest.json")
    assert len(manifest) == entries
    # Every kept frame must be a table scene with a non-empty reason.
    for entry in manifest:
        assert entry.scene.value == "table"
        assert entry.reason


def test_sample_excludes_nongame_when_requested(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "source" / "raw").mkdir(parents=True)
    cfg = _cfg()
    frameset = [_blank_frame() for _ in range(20)]
    reader = _SeqReader(frameset)
    written, entries = sample_session(
        root,
        "session_001",
        normalization_config=cfg,
        options=SampleOptions(exclude_nongame=True),
        reader=reader,
    )
    assert written == 0
    assert entries == 0


def test_sample_keeps_nongame_by_default(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "source" / "raw").mkdir(parents=True)
    cfg = _cfg()
    frameset = [_blank_frame() for _ in range(20)]
    reader = _SeqReader(frameset)
    written, entries = sample_session(
        root,
        "session_001",
        normalization_config=cfg,
        reader=reader,
    )
    assert written > 0
    for entry in read_frame_manifest(root / "normalized" / "manifest.json"):
        assert entry.scene.value == "signal_loss"


def test_frame_filename_format():
    from tools.capture_card_calibration.dataset import frame_filename

    name = frame_filename("session_001", 123456, 789, "a" * 64)
    assert name.startswith("session_001__t_00123456__f_000789__")
    assert name.endswith(".png")
