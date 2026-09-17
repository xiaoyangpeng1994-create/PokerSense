import hashlib
import json

import pytest

from poker_engine.desktop.aa_issues import save_issue


def test_issue_saved_only_on_request_with_atomic_frame_and_rules(tmp_path):
    snapshot = {"status": "RUNNING", "generation": 4, "sequence": 9,
                "source_kind": "development-replay", "payload": {"frame": 9}}
    preview = b"test jpeg bytes"
    evidence = (snapshot, preview)
    first = save_issue(tmp_path, evidence, "底池待复核", "amounts", {"revision": "x"})
    second = save_issue(tmp_path, evidence, "另一个问题", "cards", {})
    assert first["issue_id"] != second["issue_id"]
    path = tmp_path / first["issue_id"]
    content = (path / "issue.json").read_bytes()
    result = json.loads(content)
    assert result["observation"] == snapshot
    assert result["preview_sha256"] == hashlib.sha256(preview).hexdigest()
    assert first["issue_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["table_rules"]["revision"] == "x"
    assert not result["training_eligible"]
    assert not result["independent_acceptance"]
    assert (path / "preview.jpg").read_bytes() == preview


@pytest.mark.parametrize("status", ["STOPPED", "ERROR", "STALE", "STARTING"])
def test_cannot_save_obsolete_image_as_current_issue(tmp_path, status):
    with pytest.raises(ValueError, match="新鲜"):
        save_issue(tmp_path, ({"status": status, "payload": {"frame": 1}}, b"old"),
                   "", "other", {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("category", [[], {}, True, None])
def test_malformed_category_is_validation_error(tmp_path, category):
    with pytest.raises(ValueError, match="类别"):
        save_issue(tmp_path, ({}, None), "", category, {})
