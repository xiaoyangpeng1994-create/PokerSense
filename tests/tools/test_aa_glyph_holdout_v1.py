import copy
import json
from pathlib import Path

import pytest

from tools import aa_glyph_holdout_v1 as h


def config():
    return {"version": 1, "scope": "template-disjoint intra-session holdout",
            "sources": [{"id": "s"}], "evidence": [
                {"id": "history", "file": "history.json", "sha256": "a" * 64}],
            "episodes": [], "candidates": [{
                "id": "hand-a", "source": "s", "first": 10, "last": 11,
                "slots": list(range(8)), "exposure": {k: "NO" for k in h.FLAGS},
                "clean_proven": True, "episode_complete": True,
                "evidence": ["history"]}]}


@pytest.mark.parametrize("flag", sorted(h.FLAGS))
@pytest.mark.parametrize("value", ["YES", "UNKNOWN"])
def test_contamination_and_unknown_provenance_rejected(flag, value):
    c = config()
    c["candidates"][0]["exposure"][flag] = value
    assert not h.eligibility(c)[0]["eligible"]


def test_adjacent_frame_same_episode_rejected_not_just_template_frame():
    c = config()
    c["episodes"] = [{"source": "s", "first": 5, "last": 12,
                      "template_frame": 6}]
    result = h.eligibility(c)[0]
    assert not result["eligible"]
    assert "TEMPLATE_EPISODE_ENCLOSURE_OVERLAP" in result["reasons"]


@pytest.mark.parametrize("key", ["clean_proven", "episode_complete"])
def test_missing_proof_does_not_become_clean(key):
    c = config()
    c["candidates"][0][key] = False
    assert not h.eligibility(c)[0]["eligible"]


def test_queue_is_exhaustive_temporal_then_eight_seats():
    q = h.schedule(h.eligibility(config()))
    assert q == [("hand-a", f, s) for f in (10, 11) for s in range(8)]
    assert len(q) == 16
    c = config()
    c["candidates"][0]["exposure"]["tuning_or_debug"] = "YES"
    assert h.schedule(h.eligibility(c)) == []


def test_per_slot_negative_census_and_fn_fp():
    q = h.schedule(h.eligibility(config()))
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


@pytest.mark.parametrize("text", ["G:/private/x.json", "../x", "x\\y",
                                  "image.png", "clip.mkv", "12345678901",
                                  "ghp_credential"])
def test_public_config_rejects_private_paths_media_and_identifiers(text):
    with pytest.raises(ValueError):
        h.public_safe({"nested": [text]})


def inputs(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "history.json").write_text("{}")
    c = config()
    c["evidence"][0]["sha256"] = h.digest(evidence / "history.json")
    c["candidates"][0]["clean_proven"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    return path, evidence


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
    (evidence / "history.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence hash"):
        h.conclude(cfg, evidence, manifest, h.digest(manifest), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_committed_inventory_is_all_excluded():
    cfg = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_glyph_holdout_v1.json")
    decisions = h.eligibility(h.load(cfg))
    assert len(decisions) == 13
    assert sum(d["last"] - d["first"] + 1 for d in decisions) == 18003
    assert not any(d["eligible"] for d in decisions)
