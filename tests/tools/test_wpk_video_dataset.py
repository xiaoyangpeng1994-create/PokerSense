"""No private footage: exact source mapping and review-boundary tests."""

import json

import cv2
import numpy as np
import pytest

from tools.capture_card_calibration.hashing import sha256_file
from tools.wpk_video_dataset import (
    apply_review_anchors, frame_ranges, pixels_digest, review_page,
    safe_reference_path, scan_video,
)


def test_repeated_matches_remain_explicit_ranges():
    assert frame_ranges([5, 2, 3, 5, 9]) == [[2, 3], [5, 5], [9, 9]]
    assert frame_ranges([]) == []


def test_quick_signature_does_not_replace_full_pixel_identity():
    a = np.zeros((24, 12, 3), np.uint8)
    b = a.copy()
    b[1, 1] = 255
    assert pixels_digest(a, quick=True) == pixels_digest(b, quick=True)
    assert pixels_digest(a) != pixels_digest(b)


def test_reference_path_cannot_escape_dataset(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_reference_path(tmp_path, "../private.png")


def test_review_page_does_not_promote_candidate_to_truth():
    page = review_page([{"source_frame": 3, "container_pts_ms": 100,
                         "image": "frames/000003.jpg"}], "<script>")
    assert "&lt;script&gt;" in page
    assert "均待复核" in page
    assert "不使用按钮缺失推断旁观" in page


def test_mapping_finds_wrong_legacy_index_and_duplicates(tmp_path, monkeypatch):
    root, output = tmp_path / "old", tmp_path / "new"
    for name in ("labels", "normalized/frames", "normalization"):
        (root / name).mkdir(parents=True)
    output.mkdir()
    frames = [np.full((24, 12, 3), value, np.uint8) for value in (80, 90, 90, 100)]
    reference = root / "normalized/frames/ref.png"
    reference.write_bytes(cv2.imencode(".png", frames[1])[1].tobytes())
    label = {"session_id": "session_002", "frame": "ref.png", "hand_id": "h1",
             "sha256": sha256_file(reference)}
    (root / "labels/frames.jsonl").write_text(json.dumps(label), encoding="utf-8")
    (root / "normalized/manifest.json").write_text(json.dumps({
        "frames": [{"file": "ref.png", "source_frame": 0}]}), encoding="utf-8")
    (root / "normalization/normalization.json").write_text(json.dumps({
        "schema_version": 1, "rotate_degrees": 0, "output_size": [12, 24],
        "source_size": [12, 24]}), encoding="utf-8")
    video = tmp_path / "fake.mkv"
    video.write_bytes(b"fixture")

    class Capture:
        index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return len(frames) if prop == cv2.CAP_PROP_FRAME_COUNT else self.index * 30

        def read(self):
            if self.index == len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    stat = video.stat()
    summary = scan_video({"session": "session_002", "path": str(video),
                          "sha256": sha256_file(video), "size_bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns}, root, output, 1)
    assert summary["decoded_frames"] == 4
    assert summary["nominal_reference_exact"] == 0
    assert summary["reference_status_counts"] == {"MULTIPLE": 1}
    mapped = json.loads((output / "session_002/reference-map.json").read_text())
    assert mapped[0]["source_frame_ranges"] == [[1, 2]]
    samples = json.loads((output / "session_002/samples.json").read_text())
    assert all(row["owner_presence"] == "UNKNOWN" for row in samples)


@pytest.mark.parametrize("bad", ("pixel_hash", "all_in_decision"))
def test_bad_review_cannot_partially_change_samples(tmp_path, bad):
    sample = {"source_frame": 1, "pixel_sha256": "pixel-a"}
    (tmp_path / "samples.json").write_text(json.dumps([sample]), encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps({"source_sha256": "source-a"}), encoding="utf-8")
    anchor = sample | {
        "owner_presence": "SEATED", "hand_participation": "IN_HAND",
        "hero_turn": "HERO"}
    if bad == "pixel_hash":
        anchor["pixel_sha256"] = "wrong"
    else:
        anchor["hand_participation"] = "ALL_IN"
    (tmp_path / "review-anchors.json").write_text(json.dumps({
        "source_sha256": "source-a", "anchors": [anchor]}), encoding="utf-8")
    before = (tmp_path / "samples.json").read_bytes()
    with pytest.raises(ValueError):
        apply_review_anchors(tmp_path)
    assert (tmp_path / "samples.json").read_bytes() == before
