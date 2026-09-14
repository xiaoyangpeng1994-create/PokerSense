import hashlib
import json

import pytest

from tools import aa8_action_truth_review as module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture(tmp_path):
    source_path, review_path = tmp_path / "source.json", tmp_path / "review.json"
    samples = tmp_path / "samples"
    frames = samples / "frames"
    frames.mkdir(parents=True)
    sample_rows = []
    for frame in range(100, 137):
        path = frames / f"frame_{frame:06d}.png"
        path.write_text(str(frame), encoding="utf-8")
        sample_rows.append({"global_frame": frame,
                            "file": path.relative_to(samples).as_posix(),
                            "sha256": digest(path)})
    write_json(samples / "samples.json", {"samples": sample_rows})
    write_manifest(samples)
    opportunities, labels = [], []
    for index in range(1, 37):
        frame = 100 + index
        opportunities.append({
            "hand_id": "hand", "action_frame": frame, "actor_slot": 0,
            "glyph": "check", "candidate_board_count": 3,
            "source_sha256": sample_rows[index]["sha256"],
        })
        stale = index > 31
        labels.append({
            "review_id": f"A{index:02d}", "hand_id": "hand",
            "action_frame": frame, "actor_slot": 0, "candidate_glyph": "check",
            "review_status": (module.STALE if stale else module.MATCH),
            "actual_action": None if stale else "check",
            "amount": None if stale else "0",
            "amount_status": None if stale else "ZERO_CHIP_ACTION",
            "duplicate_of": "A01" if stale else None,
            "street": "flop", "legal_actions": None,
            "evidence_frames": [100, frame],
        })
    source = {"opportunities": opportunities, "registered_hands": [{
        "hand_id": "hand", "start_frame": 100, "end_frame": 136}]}
    write_json(source_path, source)
    review = {
        "schema_version": 1,
        "status": "DEVELOPMENT_AUTHOR_REVIEW_NOT_INDEPENDENT",
        "source_report_sha256": digest(source_path),
        "source_samples_sha256": digest(samples / "samples.json"),
        "review_scope": "all_36_complete_hand_glyph_candidates",
        "labels": labels, "constraints": ["development_only"],
    }
    write_json(review_path, review)
    return {"source": source, "source_path": source_path, "review": review,
            "review_path": review_path, "samples": samples}


def refresh(item):
    write_json(item["source_path"], item["source"])
    item["review"]["source_report_sha256"] = digest(item["source_path"])
    write_json(item["review_path"], item["review"])


def run(item):
    return module.analyze(item["source_path"], item["review_path"], item["samples"])


def test_complete_review_retains_matches_and_false_positives(tmp_path):
    item = fixture(tmp_path)
    result = run(item)
    assert result["candidates_reviewed"] == 36
    assert result["matched_visible_actions"] == 31
    assert result["false_positive_candidates"] == 5
    assert result["opponent_matched_visible_actions"] == 31
    assert result["complete_legal_action_menus"] == 0
    assert result["full_decision_truth_rows"] == 0
    assert not result["ready_for_opponent_calibration"]
    assert not result["strategy_eligible"] and not result["advice_emitted"]


def test_missing_candidate_label_is_rejected(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"].pop()
    refresh(item)
    with pytest.raises(ValueError, match="all_36"):
        run(item)


def test_candidate_identity_cannot_be_relabelled(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][0]["actor_slot"] = 1
    refresh(item)
    with pytest.raises(ValueError, match="candidate_identity"):
        run(item)


def test_candidate_source_hash_must_match_action_frame(tmp_path):
    item = fixture(tmp_path)
    item["source"]["opportunities"][0]["source_sha256"] = "f" * 64
    refresh(item)
    with pytest.raises(ValueError, match="source_hash_differs"):
        run(item)


def test_legal_menu_cannot_be_invented_from_visible_glyph(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][0]["legal_actions"] = ["check", "bet"]
    refresh(item)
    with pytest.raises(ValueError, match="legal_menu_must_remain_unknown"):
        run(item)


def test_actual_action_must_agree_with_candidate_glyph(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][0].update(
        actual_action="call", amount="2",
        amount_status="VISUAL_STACK_DELTA_MATCH")
    refresh(item)
    with pytest.raises(ValueError, match="differs_from_visible_glyph"):
        run(item)


def test_zero_chip_action_cannot_claim_positive_amount(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][0]["amount"] = "1"
    refresh(item)
    with pytest.raises(ValueError, match="passive_zero"):
        run(item)


def test_false_positive_cannot_gain_an_amount(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][-1]["amount"] = "1"
    refresh(item)
    with pytest.raises(ValueError, match="cannot_become_action"):
        run(item)


def test_stale_reappearance_requires_prior_matching_action(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][-1]["duplicate_of"] = "A36"
    refresh(item)
    with pytest.raises(ValueError, match="prior_same_action"):
        run(item)


def test_stale_reappearance_requires_same_actor_and_glyph(tmp_path):
    item = fixture(tmp_path)
    item["source"]["opportunities"][0]["actor_slot"] = 1
    item["review"]["labels"][0]["actor_slot"] = 1
    refresh(item)
    with pytest.raises(ValueError, match="prior_same_action"):
        run(item)


def test_stale_reappearance_cannot_reference_another_hand(tmp_path):
    item = fixture(tmp_path)
    item["source"]["registered_hands"] = [
        {"hand_id": "first", "start_frame": 100, "end_frame": 131},
        {"hand_id": "second", "start_frame": 132, "end_frame": 136},
    ]
    for index in range(36):
        hand = "first" if index < 31 else "second"
        item["source"]["opportunities"][index]["hand_id"] = hand
        item["review"]["labels"][index]["hand_id"] = hand
        if hand == "second":
            item["review"]["labels"][index]["evidence_frames"] = (
                [132, 133] if index == 31 else [132, 100 + index + 1])
    refresh(item)
    with pytest.raises(ValueError, match="prior_same_action"):
        run(item)


def test_post_action_display_is_not_a_duplicate_action(tmp_path):
    item = fixture(tmp_path)
    label = item["review"]["labels"][-1]
    label["review_status"] = module.POST_ACTION
    refresh(item)
    with pytest.raises(ValueError, match="not_duplicate_action"):
        run(item)


def test_evidence_must_include_action_frame(tmp_path):
    item = fixture(tmp_path)
    item["review"]["labels"][0]["evidence_frames"] = [100, 102]
    refresh(item)
    with pytest.raises(ValueError, match="action_evidence"):
        run(item)


def test_evidence_hash_change_is_rejected(tmp_path):
    item = fixture(tmp_path)
    path = item["samples"] / "frames" / "frame_000101.png"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_hash_mismatch"):
        run(item)


def test_review_source_hash_is_bound(tmp_path):
    item = fixture(tmp_path)
    item["review"]["source_report_sha256"] = "0" * 64
    write_json(item["review_path"], item["review"])
    with pytest.raises(ValueError, match="source_or_scope"):
        run(item)


def test_shipped_review_has_30_matches_and_six_false_candidates():
    review = json.loads(open(
        "configs/reproduction/aa8_late_action_review_v1.json",
        encoding="utf-8").read())
    statuses = [label["review_status"] for label in review["labels"]]
    assert statuses.count(module.MATCH) == 30
    assert statuses.count(module.POST_ACTION) == 1
    assert statuses.count(module.STALE) == 5
    assert all(label["legal_actions"] is None for label in review["labels"])
