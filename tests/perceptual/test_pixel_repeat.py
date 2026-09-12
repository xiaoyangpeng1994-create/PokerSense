"""Identical pixels alone cannot distinguish a live still scene from a freeze."""

import numpy as np

from poker_engine.perceptual.capture.pixel_repeat import PixelRepeatObserver
from poker_engine.perceptual.capture.capture_card_backend import CaptureCardBackend
from poker_engine.perceptual.capture.base import CaptureTarget


def test_repeats_with_new_host_frame_ids_are_not_declared_fresh_or_frozen():
    observer = PixelRepeatObserver()
    image = np.full((40, 30, 3), 120, np.uint8)
    first = observer.observe(image, 1, 100)
    assert first.pixels_changed is None
    for seq in range(2, 8):
        result = observer.observe(image, seq, 100 + seq)
    assert result.consecutive_identical_frames == 7
    assert result.unchanged_host_seconds == 7
    assert result.source_freshness == "UNKNOWN"
    assert result.pixels_changed is False


def test_pixel_change_resets_count_but_does_not_prove_device_liveness():
    observer = PixelRepeatObserver()
    image = np.full((40, 30, 3), 120, np.uint8)
    observer.observe(image, 1, 0)
    image[0, 0, 0] = 119
    result = observer.observe(image, 2, 1)
    assert result.pixels_changed is True
    assert result.consecutive_identical_frames == 1
    assert result.source_freshness == "UNKNOWN"


def test_shape_change_and_reset_do_not_carry_repetition_history():
    observer = PixelRepeatObserver()
    observer.observe(np.zeros((4, 3), np.uint8), 1, 0)
    result = observer.observe(np.zeros((3, 4), np.uint8), 2, 1)
    assert result.pixels_changed is True
    observer.reset()
    assert observer.evidence is None


def test_backend_does_not_reject_a_legitimate_static_table():
    class Capture:
        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            return True, np.full((40, 30, 3), (30, 110, 50), np.uint8)

        def release(self):
            pass

    backend = CaptureCardBackend(video_capture_factory=lambda *args: Capture())
    target = CaptureTarget("uvc-0")
    frames = [backend.capture(target) for _ in range(5)]
    assert len({frame.frame_seq for frame in frames}) == 5
    assert backend.pixel_repeat_evidence.consecutive_identical_frames == 5
    assert backend.pixel_repeat_evidence.source_freshness == "UNKNOWN"
    backend.release()
    assert backend.pixel_repeat_evidence is None
