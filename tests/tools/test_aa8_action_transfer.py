import json

import pytest

from tools.aa8_action_transfer import inventory, load


def write_pool(tmp_path, ids, role="development", audit="audit"):
    rows = [{"global_frame": i, "role": role} for i in ids]
    (tmp_path / "samples.json").write_text(json.dumps({
        "audit_sha256": audit, "samples": rows}))


def test_contiguous_development_inventory(tmp_path):
    write_pool(tmp_path, [10, 11, 12])
    assert list(inventory(tmp_path, "audit")) == [10, 11, 12]


@pytest.mark.parametrize("ids", [[], [10, 12], [10, 10], [11, 10]])
def test_bad_inventory_rejected(tmp_path, ids):
    write_pool(tmp_path, ids)
    with pytest.raises(ValueError, match="contiguous"):
        inventory(tmp_path, "audit")


def test_holdout_and_wrong_source_rejected(tmp_path):
    write_pool(tmp_path, [10], role="holdout")
    with pytest.raises(ValueError, match="development"):
        inventory(tmp_path, "audit")
    with pytest.raises(ValueError, match="recording"):
        inventory(tmp_path, "another")


def test_escaping_image_rejected_before_read(tmp_path):
    with pytest.raises(ValueError, match="path/hash"):
        load(tmp_path, {"file": "../escape.png", "sha256": "untrusted"})
