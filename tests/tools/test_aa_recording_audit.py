from decimal import Decimal

import pytest

from tools.audit_aa_recording import validate_segments


def test_segment_end_is_absolute_timeline_not_per_file_duration(tmp_path):
    for name in ("segment_0000.mkv", "segment_0001.mkv"):
        (tmp_path / name).write_bytes(b"fixture")
    result = validate_segments(tmp_path, [
        ["segment_0000.mkv", "0", "60"], ["segment_0001.mkv", "60.001", "95"]])
    assert Decimal(result[1]["csv_end"]) - Decimal(result[1]["csv_start"]) == (
        Decimal("34.999"))


@pytest.mark.parametrize("start,end", [
    ("59", "95"), ("61", "95"), ("NaN", "95"), ("60", "59")])
def test_bad_timeline_rejected(tmp_path, start, end):
    for name in ("segment_0000.mkv", "segment_0001.mkv"):
        (tmp_path / name).write_bytes(b"fixture")
    with pytest.raises(ValueError):
        validate_segments(tmp_path, [["segment_0000.mkv", "0", "60"],
                                     ["segment_0001.mkv", start, end]])


def test_unlisted_video_and_traversal_rejected(tmp_path):
    (tmp_path / "segment_0000.mkv").write_bytes(b"fixture")
    (tmp_path / "segment_0001.mkv").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="match files"):
        validate_segments(tmp_path, [["segment_0000.mkv", "0", "60"]])
    with pytest.raises(ValueError, match="name/order"):
        validate_segments(tmp_path, [["../segment_0000.mkv", "0", "60"]])
