"""The historical gate verifies receipts without using private evidence in CI."""

import hashlib
import json

from tools import audit_action_likelihood_historical_v1 as module


def _synthetic_manifest(tmp_path):
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    private = tmp_path / "private"
    for index, receipt in enumerate(manifest["source_receipts"]):
        raw = (f"synthetic structured metadata {index}\n").encode()
        path = private / receipt["relative_private_path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        receipt["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path = tmp_path / "audit.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, private


def test_frozen_public_manifest_is_valid_but_not_live_reverified():
    report = module.audit(private_root=None)
    assert report["status"] == "PUBLIC_ONLY_NOT_LIVE_REVERIFIED"
    assert report["manifest_verified"] is True
    assert report["private_receipts_verified"] is False
    assert report["eligible_real_opportunities"] == 0
    assert "private_evidence_not_live_reverified" in report["refusal_reasons"]


def test_synthetic_receipts_verify_without_promoting_candidates(tmp_path):
    manifest, private = _synthetic_manifest(tmp_path)
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["status"] == "VERIFIED_INSUFFICIENT_DATA"
    assert report["private_receipts_verified"] is True
    assert report["eligible_real_opportunities"] == 0
    assert set(report["model_support"].values()) == {"INSUFFICIENT_DATA"}
    assert ("all_opportunity_denominator_including_unmatched_and_unknown"
            in report["refusal_reasons"])


def test_tampered_receipt_refuses_deterministically(tmp_path):
    manifest, private = _synthetic_manifest(tmp_path)
    receipt = module.RECEIPTS[1]
    (private / receipt[1]).write_bytes(b"tampered metadata")
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["status"] == "REFUSED"
    assert report["private_receipts_verified"] is False
    assert (report["refusal_reasons"][0]
            == "private_receipt_sha256_mismatch:" + receipt[0])
    assert report["eligible_real_opportunities"] == 0


def test_missing_receipt_refuses_without_reading_any_media(tmp_path):
    manifest, private = _synthetic_manifest(tmp_path)
    role, relative = module.RECEIPTS[2]
    (private / relative).unlink()
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["status"] == "REFUSED"
    assert report["refusal_reasons"][0] == "private_receipt_missing:" + role


def test_public_manifest_drift_and_schema_change_refuse(tmp_path):
    manifest, private = _synthetic_manifest(tmp_path)
    frozen_report = module.audit(manifest, None)
    assert frozen_report["status"] == "REFUSED"
    assert frozen_report["refusal_reasons"] == [
        "frozen_public_manifest_sha256_mismatch"]

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["opportunity_corpus"]["eligible_training_count"] = 1
    manifest.write_text(json.dumps(document), encoding="utf-8")
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["status"] == "REFUSED"
    assert report["refusal_reasons"] == [
        "public_manifest_field_mismatch:opportunity_corpus"]


def test_schema_rejects_extra_receipt_field_and_wrong_boolean_type(tmp_path):
    manifest, private = _synthetic_manifest(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["source_receipts"][0]["future_showdown_feature"] = "not permitted"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["refusal_reasons"] == ["public_manifest_receipt_schema_mismatch:0"]

    del document["source_receipts"][0]["future_showdown_feature"]
    document["opportunity_corpus"]["complete_denominator_verified"] = 0
    manifest.write_text(json.dumps(document), encoding="utf-8")
    report = module.audit(manifest, private, require_frozen_hash=False)
    assert report["refusal_reasons"] == [
        "public_manifest_field_mismatch:opportunity_corpus"]
