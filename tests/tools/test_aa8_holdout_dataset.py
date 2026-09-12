import pytest

from tools.aa8_holdout_dataset import selected_index


def meta():
    return {"role": "holdout", "audit_sha256": "audit", "allowed_start_seconds": 600,
            "allowed_end_seconds": 820, "hands": [{
                "id": "h", "first_frame": 101, "last_frame": 103,
                "preroll_first_frame": 100, "boundaries_verified": True}]}


def index():
    return [{"global_frame": f, "pts_seconds": str(600 + f / 30),
             "segment": "segment.mkv", "local_frame": f} for f in range(99, 105)]


def test_only_registered_ownership_and_preroll_selected_without_media():
    result = selected_index(index(), meta())
    assert list(result) == [100, 101, 102, 103]
    assert all(r["role"] == "holdout" for r in result.values())


def test_missing_owned_frame_rejected():
    values = [r for r in index() if r["global_frame"] != 101]
    with pytest.raises(ValueError, match="exact ordered"):
        selected_index(values, meta())


def test_duplicate_index_frame_rejected():
    values = index()
    values.append(values[2])
    with pytest.raises(ValueError, match="duplicate"):
        selected_index(values, meta())
