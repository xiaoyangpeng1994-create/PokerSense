"""Source-aligned window extraction must not use approximate seeking."""

import json

import cv2
import numpy as np
import pytest

from tools.wpk_hand_review import extract_window
from tools.wpk_video_dataset import pixels_digest


@pytest.mark.parametrize("selected", [None, {2, 5}])
def test_sequential_skip_keeps_exact_indices_and_pixel_anchors(
    tmp_path, monkeypatch, selected,
):
    old, corpus = tmp_path / "old", tmp_path / "corpus"
    (old / "source/raw").mkdir(parents=True)
    (old / "normalization").mkdir()
    (corpus / "session_002").mkdir(parents=True)
    video = old / "source/raw/session_002.mkv"
    video.write_bytes(b"mock input")
    stat = video.stat()
    (corpus / "sources.json").write_text(json.dumps({"sources": [{
        "session": "session_002", "path": str(video), "sha256": "source-hash",
        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}), encoding="utf-8")
    (old / "normalization/normalization.json").write_text(json.dumps({
        "schema_version": 1, "rotate_degrees": 0, "output_size": [12, 24],
        "source_size": [12, 24]}), encoding="utf-8")
    frames = [np.full((24, 12, 3), i, np.uint8) for i in range(6)]
    (corpus / "session_002/samples.json").write_text(
        json.dumps([{"source_frame": 3, "pixel_sha256": pixels_digest(frames[3])}]),
        encoding="utf-8")

    class Capture:
        index = 0

        def isOpened(self):
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            index = self.index
            ok = self.grab()
            return ok, frames[index] if ok else None

        def get(self, prop):
            return (self.index - 1) * 33.333

        def release(self):
            pass

        def set(self, *args):
            raise AssertionError("window extraction must not seek")

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    output = corpus / "window"
    result = extract_window(corpus, output, 2, 5, 2, selected_frames=selected)
    rows = json.loads((output / "samples.json").read_text())
    assert [row["source_frame"] for row in rows] == (
        [2, 3, 4, 5] if selected is None else [2, 5])
    assert result["verified_pixel_anchors"] == ([3] if selected is None else [])
    assert result["complete_hand_verified"] is False
    assert rows[0]["pixel_sha256"] == pixels_digest(frames[2])


def test_refuses_writing_over_existing_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="child"):
        extract_window(corpus, corpus, 0, 1, 1)
    with pytest.raises(ValueError, match="child"):
        extract_window(corpus, tmp_path / "outside", 0, 1, 1)


@pytest.mark.parametrize("selected", [set(), {0}, {6}, {True}])
def test_invalid_explicit_indices_are_rejected_before_reading_sources(
    tmp_path, selected,
):
    corpus = tmp_path / "corpus"
    with pytest.raises(ValueError, match="selected frames"):
        extract_window(corpus, corpus / "out", 2, 5, 1, selected_frames=selected)
