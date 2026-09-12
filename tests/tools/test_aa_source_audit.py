"""Source audit safety and deterministic sequential-index checks."""

import json

import numpy as np
import pytest

from tools import aa_source_audit as module


class FakeCapture:
    def __init__(self, *args):
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == module.cv2.CAP_PROP_FRAME_COUNT:
            return 3
        return (self.index - 1) * 1000 / 30

    def read(self):
        self.index += 1
        if self.index > 3:
            return False, None
        return True, np.full((10, 12, 3), self.index, dtype=np.uint8)

    def release(self):
        pass


def test_sequential_audit_preserves_source_and_rejects_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "原片.mkv"
    source.write_bytes(b"source")
    output = tmp_path / "派生"
    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(module, "probe_video", lambda path: {})
    result = module.audit(source, output, every=2)
    assert source.read_bytes() == b"source"
    assert result["decoded_frames"] == 3
    assert result["nonincreasing_decoder_times"] == 0
    assert [row["source_frame_index"] for row in result["samples"]] == [0, 2]
    assert not result["vision_ready_for_strategy"]
    rows = [json.loads(line) for line in
            (output / "decode_index.jsonl").read_text().splitlines()]
    assert [row["source_frame_index"] for row in rows] == [0, 1, 2]
    with pytest.raises(FileExistsError):
        module.audit(source, output)


def test_invalid_interval_creates_nothing(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        module.audit(tmp_path / "missing", tmp_path / "output", 0)
    assert not (tmp_path / "output").exists()


def test_short_decode_retains_failure_report(tmp_path, monkeypatch):
    class ShortCapture(FakeCapture):
        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FRAME_COUNT:
                return 4
            return super().get(prop)

    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    monkeypatch.setattr(module.cv2, "VideoCapture", ShortCapture)
    monkeypatch.setattr(module, "probe_video", lambda path: {})
    with pytest.raises(ValueError, match="count check failed"):
        module.audit(source, tmp_path / "out")
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert not report["count_matches_header"]
