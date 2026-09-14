import copy
import json
from pathlib import Path

import pytest

from tools import aa_glyph_holdout_v1 as h


def config():
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_glyph_holdout_v1.json")
    return h.load(path)


def history(c):
    # Synthetic structural witnesses only. Never reads private history in tests.
    ev = {k: dict.fromkeys(v) for k, v in h.HISTORY_KEYS.items()}
    source = c["sources"][0]
    counts = [m["last"] - m["first"] + 1 for m in source["media"]]
    ev["receipt"].update(schema_version=1, session_id="synthetic-session",
                         segments=[], progress=dict(
                             frame=sum(counts), drop_frames=0, dup_frames=0,
                             out_time="synthetic", progress="end"))
    for i, media in enumerate(source["media"]):
        ev["receipt"]["segments"].append(dict(
            segment_index=i, relative_path=f"segment_{i:04d}.mkv",
            sha256=media["sha256"], csv_start_pts="0", csv_end_pts_exclusive="1",
            csv_gap_seconds=None, size_bytes=1, mtime_ns=1))
    ev["boundaries"].update(schema_version=1, source_session_id="synthetic-session",
                            source_segment_frame_counts=counts,
                            total_frames=sum(counts), hands=[])
    for row in c["candidates"]:
        ev["boundaries"]["hands"].append(dict(
            hand_id=row["id"], start_global_frame=row["first"],
            end_global_frame=row["last"], temporal_complete=True, status="test"))
    selected = []
    for row in c["candidates"]:
        if row["exposure"]["tuning_or_debug"] == "YES":
            selected.append(dict(
                hand_id=row["id"], role="development",
                start_global_frame=row["first"], end_global_frame=row["last"],
                start_pts_seconds="0", end_pts_seconds="1",
                frame_count=row["last"] - row["first"] + 1,
                preflop_only_candidate=False, temporal_complete=True,
                ordinary_mode=True, privacy_usable=True))
    total = sum(x["frame_count"] for x in selected)
    ev["selection"].update(schema_version=1, source_session_id="synthetic-session",
                           selected_hands=selected, selected_frame_count=total,
                           selected_hand_count=len(selected))
    for key in ("v3-first", "v3-final"):
        ev[key].update(status="DEVELOPMENT_REGRESSION_NOT_HOLDOUT",
                       frames=total, reference_frames=[4565, 8680, 11761],
                       training_cases_included=True, events=[dict(
                           hand_id=x["hand_id"], frame=x["start_global_frame"],
                           slot=0, glyph="fold") for x in selected])
    rows = []
    for c_row in c["candidates"]:
        if c_row["exposure"]["manual_action_review"] == "YES":
            rows.append(dict(
                event_id="e" + str(len(rows)), hand_id=c_row["id"],
                frame=c_row["first"], slot=0, glyph="fold",
                review_status="MATCH_VISIBLE_GLYPH_TRANSITION",
                manual_action="fold", manual_street="preflop",
                visible_stack_decrease=None, amount_semantics="synthetic",
                legal_amount=None, evidence_page="test.png"))
    ev["action-review"].update(rows=rows, candidate_count=len(rows),
                               matched_glyph_transitions=len(rows), false_folds=0)
    ev["v3-comparison"].update(training_and_regression_overlap=True,
                               independent_holdout=False, all_event_count=len(selected))
    return ev


def inputs(tmp_path):
    c = config()
    ev = history(c)
    directory = tmp_path / "evidence"
    directory.mkdir()

    def save(key):
        row = next(r for r in c["evidence"] if r["id"] == key)
        path = directory / row["file"]
        path.write_text(json.dumps(ev[key]), encoding="utf-8")
        row["sha256"] = h.digest(path)
        return row["sha256"]
    receipt = save("receipt")
    c["sources"][0]["receipt_sha256"] = receipt
    ev["boundaries"]["source_receipt_sha256"] = receipt
    boundary = save("boundaries")
    ev["selection"].update(source_receipt_sha256=receipt,
                           exact_boundary_registry_sha256=boundary)
    for key in ev:
        save(key)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c), encoding="utf-8")
    return path, directory


def test_negative_only_even_fabricated_clean_candidate(tmp_path):
    cfg, directory = inputs(tmp_path)
    c = h.load(cfg)
    row = c["candidates"][0]
    row.update(clean_proven=True, episode_complete=True,
               exposure={key: "NO" for key in h.FLAGS})
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="POSITIVE_ELIGIBILITY_UNSUPPORTED"):
        h.freeze(cfg, directory, tmp_path / "frozen.json")
    assert not (tmp_path / "frozen.json").exists()


@pytest.mark.parametrize("case", ["gap", "overlap", "outside", "compensating",
                                  "media-gap", "media-overlap", "media-sha",
                                  "unknown-media", "wrong-media", "missing-episode",
                                  "shrunk-episode", "duplicate-template",
                                  "shift-episode",
                                  "bool-frame", "bool-version", "unknown-field"])
def test_closure_attacks_fail(tmp_path, case):
    cfg, directory = inputs(tmp_path)
    c = h.load(cfg)
    if case == "gap":
        c["candidates"][1]["first"] += 1
    elif case == "overlap":
        c["candidates"][1]["first"] -= 1
    elif case == "outside":
        c["candidates"][-1]["last"] += 1
    elif case == "compensating":
        c["candidates"][1]["first"] += 1
        c["candidates"][-1]["last"] += 1
    elif case == "media-gap":
        c["sources"][0]["media"][1]["first"] += 1
    elif case == "media-overlap":
        c["sources"][0]["media"][1]["first"] -= 1
    elif case == "media-sha":
        c["sources"][0]["media"][0]["sha256"] = "b" * 64
    elif case == "unknown-media":
        c["candidates"][0]["media_ids"] = ["absent"]
    elif case == "wrong-media":
        c["candidates"][0]["media_ids"].append("segment-01")
    elif case == "missing-episode":
        c["episodes"] = []
    elif case == "shrunk-episode":
        c["episodes"][0]["first"] += 1
    elif case == "duplicate-template":
        c["episodes"][1]["template_frame"] = c["episodes"][0]["template_frame"]
    elif case == "shift-episode":
        c["episodes"][0]["template_frame"] = 1
    elif case == "bool-frame":
        c["candidates"][0]["first"] = False
    elif case == "bool-version":
        c["version"] = True
    else:
        c["candidates"][0]["extra"] = "unexpected"
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        h.freeze(cfg, directory, tmp_path / "frozen.json")


def test_derived_unknown_not_handwritten_yes(tmp_path):
    cfg, directory = inputs(tmp_path)
    c = h.load(cfg)
    c["candidates"][0]["exposure"]["manual_action_review"] = "YES"
    result = h.eligibility(c, h.verified_evidence(c, directory))
    assert result[0]["derived_exposure"]["manual_action_review"] == "UNKNOWN"
    assert not any(d["eligible"] for d in result)


def test_fixed_temporal_seat_schedule():
    q = h.schedule([dict(id="b", source="s", first=2, last=2, eligible=True),
                    dict(id="a", source="s", first=1, last=1, eligible=True)])
    assert q == [(name, f, s) for name, f in [("a", 1), ("b", 2)] for s in range(8)]


def test_per_slot_negative_census_and_fn_fp():
    q = [("hand-a", f, s) for f in (10, 11) for s in range(8)]
    labels = [{"key": key, "label": "none"} for key in q]
    outputs = copy.deepcopy(labels)
    labels[0]["label"] = "fold"  # false negative seat zero
    outputs[1]["label"] = "all_in"  # false positive seat one
    labels[2]["label"] = outputs[2]["label"] = "muck"
    result = h.confusion(q, labels, outputs)
    assert result["0"]["fold"]["FN"] == 1
    assert result["1"]["all_in"]["FP"] == 1
    assert result["2"]["muck"]["TP"] == 1
    assert result["7"]["fold"] == dict(
        TP=0, FN=0, FP=0, TN=2, opportunities=0, negatives=2)
    for slot in result.values():
        for cls in slot.values():
            assert sum(cls[k] for k in ("TP", "FN", "FP", "TN")) == 2
    with pytest.raises(ValueError):
        h.confusion(q, labels[:-1], outputs)
    with pytest.raises(ValueError):
        h.confusion(q, list(reversed(labels)), outputs)


_DRIVE_PATH = chr(71) + chr(58) + chr(47) + "private" + chr(47) + "x.json"


@pytest.mark.parametrize("text", [_DRIVE_PATH, "../x", "x\\y",
                                  "image.png", "clip.mkv", "12345678901",
                                  "ghp_credential"])
def test_public_config_rejects_private_paths_media_and_identifiers(text):
    with pytest.raises(ValueError):
        h.public_safe({"nested": [text]})


def test_frozen_hash_and_no_eligible_not_run_artifacts(tmp_path):
    cfg, evidence = inputs(tmp_path)
    manifest = tmp_path / "frozen.json"
    value = h.freeze(cfg, evidence, manifest)
    assert value["status"] == "NO_ELIGIBLE_HOLDOUT"
    assert value["frozen_at"].endswith("+08:00")
    expected = h.digest(manifest)
    out = tmp_path / "result"
    result = h.conclude(cfg, evidence, manifest, expected, out)
    assert not result["model_executed"]
    assert all(r["denominator"] == 0 and r["FP"] is None
               for seat in result["metrics"].values() for r in seat.values())
    assert h.load(out / "machine-output.json")["rows"] == []
    with pytest.raises(FileExistsError):
        h.freeze(cfg, evidence, manifest)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        h.conclude(cfg, evidence, manifest, expected, tmp_path / "bad")


def test_changed_evidence_stops_before_report(tmp_path):
    cfg, evidence = inputs(tmp_path)
    manifest = tmp_path / "frozen.json"
    h.freeze(cfg, evidence, manifest)
    (evidence / "v3-first.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence hash"):
        h.conclude(cfg, evidence, manifest, h.digest(manifest), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_committed_inventory_is_all_excluded():
    cfg = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_glyph_holdout_v1.json")
    decisions = h.declarations(h.load(cfg))
    decisions = h.load(cfg)["candidates"]
    assert len(decisions) == 13
    assert sum(d["last"] - d["first"] + 1 for d in decisions) == 18003
    assert all(d["clean_proven"] is False for d in decisions)


@pytest.mark.parametrize("case", ["missing", "duplicate-file", "duplicate-role",
                                  "duplicate-key", "bad-schema", "conflicting-frames"])
def test_evidence_failures_cannot_become_positive(tmp_path, case):
    cfg, directory = inputs(tmp_path)
    c = h.load(cfg)
    path = directory / "v3-first.json"
    if case == "missing":
        path.unlink()
    elif case == "duplicate-file":
        (directory / "extra.json").write_bytes(path.read_bytes())
    elif case == "duplicate-role":
        c["evidence"].append(c["evidence"][0].copy())
        cfg.write_text(json.dumps(c))
    else:
        data = h.load(path)
        if case == "duplicate-key":
            raw = '{"frames": 1, "frames": 2}'
        elif case == "bad-schema":
            raw = '{}'
        else:
            data["frames"] = True
            raw = json.dumps(data)
        path.write_text(raw)
        for entry in c["evidence"]:
            if entry["id"] == "v3-first":
                entry["sha256"] = h.digest(path)
        cfg.write_text(json.dumps(c))
    with pytest.raises((ValueError, FileNotFoundError)):
        h.freeze(cfg, directory, tmp_path / "frozen.json")


def test_symlink_evidence_rejected(tmp_path):
    cfg, directory = inputs(tmp_path)
    path = directory / "v3-first.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit creating symlinks")
    with pytest.raises(ValueError, match="symlink"):
        h.freeze(cfg, directory, tmp_path / "frozen.json")


@pytest.mark.parametrize("case", ["config", "tool", "positive"])
def test_conclude_revalidates_every_binding(tmp_path, case):
    cfg, directory = inputs(tmp_path)
    manifest = tmp_path / "frozen.json"
    h.freeze(cfg, directory, manifest)
    frozen = h.load(manifest)
    if case == "tool":
        frozen["tool_sha256"] = "0" * 64
    else:
        c = h.load(cfg)
        if case == "positive":
            c["candidates"][0].update(clean_proven=True, episode_complete=True,
                                      exposure={k: "NO" for k in h.FLAGS})
        else:
            c["candidates"][0]["scene"] = "changed"
        cfg.write_text(json.dumps(c))
        if case == "positive":
            frozen["config_sha256"] = h.digest(cfg)
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError):
        h.conclude(cfg, directory, manifest, h.digest(manifest), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_code_commit_uses_tool_repo_and_all_null_metrics(tmp_path, monkeypatch):
    import subprocess
    repo = Path(h.__file__).resolve().parents[1]
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    cfg, directory = inputs(tmp_path)
    manifest = tmp_path / "frozen.json"
    h.freeze(cfg, directory, manifest)
    monkeypatch.chdir(tmp_path)
    result = h.conclude(cfg, directory, manifest, h.digest(manifest), tmp_path / "out")
    assert result["code_commit"] == expected
    for metrics in [result["metrics"], *result["metrics_by_hand"].values()]:
        assert set(metrics) == {str(i) for i in range(8)}
        for seat in metrics.values():
            for row in seat.values():
                assert row["denominator"] == 0
                assert all(row[key] is None for key in ("TP", "FN", "FP", "TN"))
