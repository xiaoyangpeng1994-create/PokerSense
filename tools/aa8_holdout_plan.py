"""Freeze checks and boundary-only holdout planning; never decodes media."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp required")
    return stamp


def validate_freeze(freeze, base, prediction_started_at=None, now=None):
    failures = []
    now = now or datetime.now(timezone.utc)
    try:
        stamp = utc(freeze.get("frozen_at_utc", ""))
        if stamp > now:
            failures.append("freeze_timestamp_in_future")
        if prediction_started_at is not None and stamp >= utc(prediction_started_at):
            failures.append("freeze_not_before_prediction")
    except (ValueError, TypeError, AttributeError):
        failures.append("freeze_timestamp_invalid")
    entries = freeze.get("files", [])
    identities = []
    roles = set()
    training = set()
    for entry in entries:
        path = (base / entry.get("path", "")).resolve()
        identities.append(path.as_posix().casefold())
        roles.add(entry.get("kind"))
        if not path.is_file() or sha(path) != entry.get("sha256"):
            failures.append("freeze_file_hash_mismatch:" + entry.get("path", ""))
        if entry.get("kind") == "training":
            training.add(entry.get("sha256"))
    if identities != sorted(set(identities)):
        failures.append("freeze_paths_not_sorted_unique")
    if not {"implementation", "model", "parameters", "training"} <= roles:
        failures.append("freeze_required_file_roles_missing")
    exposures = freeze.get("exposures")
    if not isinstance(exposures, list) or not exposures:
        failures.append("training_exposure_ledger_missing")
    else:
        ledger_hashes = set()
        for row in exposures:
            if row.get("used_for") in ("training", "tuning", "template_selection"):
                ledger_hashes.add(row.get("artifact_sha256"))
                if row.get("role") not in ("development", "calibration"):
                    failures.append("holdout_or_unknown_exposure_used_for_tuning")
                if row.get("artifact_sha256") not in training:
                    failures.append("exposure_not_hash_bound")
            elif row.get("used_for") == "boundary_review":
                if (row.get("role") != "holdout"
                        or row.get("shared_with_tuning") is not False):
                    failures.append("boundary_review_leakage")
            else:
                failures.append("exposure_purpose_unknown")
        if not training <= ledger_hashes:
            failures.append("training_file_missing_exposure_ledger")
    return sorted(set(failures))


def boundary_plan(split, freeze, base, now=None):
    failures = validate_freeze(freeze, base, now=now)
    ranges = [r for r in split.get("ranges", [])
              if r.get("role") == "holdout_candidate"]
    if not ranges:
        failures.append("holdout_candidate_range_missing")
    return {
        "status": "BLOCKED" if failures else "READY_FOR_BOUNDARY_REVIEW_ONLY",
        "failures": sorted(set(failures)), "source_audit_sha256": split.get(
            "source_audit_sha256"), "candidate_ranges": ranges,
        "verified_independent_hands": split.get("verified_independent_hands", 0),
        "media_consumed": False, "prediction_permitted": False,
        "full_visual_acceptance": False,
        "procedure": [
            "Freeze implementation/model/parameters/training before holdout view.",
            "Isolated reviewer sees chronological footage, never model scores.",
            "Coarse chronological sampling proposes boundaries; refine uncertainty.",
            "Require consecutive before/first-post and before/next-post witnesses.",
            "Retain ALL whole hands inside range, never choose by prediction quality.",
            "Log boundary-straddling exclusions; keep ordered candidate census.",
            "Register frame IDs, PTS, witness PNG hashes and reviewer access log.",
            "Share only frozen registry metadata with tuning agents, no images/labels.",
            "Recheck freeze hashes; save prediction before comparison.",
            "Changed models invalidate freeze; exposed hands never become untouched."
        ]}


def audit_existing(split_path, report_paths):
    """Read only JSON metadata, leaving original images and videos untouched."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    failures, reports = [], []
    if split.get("verified_independent_hands", 0) == 0:
        failures.append("split_plan_has_zero_verified_independent_hands")
    for path in report_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": path.as_posix(), "sha256": sha(path)}
        for key in ("independent_holdout", "hand_boundaries_verified",
                    "full_visual_acceptance", "strategy_eligible", "frame_count",
                    "first_frame", "last_frame", "labels_used_for_prediction"):
            if key in value:
                row[key] = value[key]
        if value.get("independent_holdout") is not True:
            failures.append("report_not_independent_holdout:" + path.parent.name)
        if value.get("full_visual_acceptance") is not True:
            failures.append("report_not_full_visual_acceptance:" + path.parent.name)
        reports.append(row)
    return {"status": "PARTIAL" if failures else "NEEDS_EVIDENCE_VALIDATION",
            "full_visual_acceptance": False, "media_consumed": False,
            "split_sha256": sha(split_path), "reports": reports,
            "failures": sorted(set(failures))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.freeze:
        result = boundary_plan(json.loads(args.split.read_text(encoding="utf-8")),
                               json.loads(args.freeze.read_text(encoding="utf-8")),
                               args.freeze.parent)
    else:
        result = audit_existing(args.split, args.report)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "READY_FOR_BOUNDARY_REVIEW_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
