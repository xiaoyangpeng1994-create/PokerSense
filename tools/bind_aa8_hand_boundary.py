"""Bind reviewed AA8 hand ownership to its segment index and before/after pixels."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from tools.sample_aa8_development import permitted
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def validate_interval(review, index, plan):
    if (review.get("role") != "development" or
            review.get("independent_holdout") is not False):
        raise ValueError("this binder only registers development hands")
    start, end, next_start = (review[k] for k in (
        "start_global_frame", "end_global_frame", "next_hand_post_frame"))
    if any(type(v) is not int for v in (start, end, next_start)) or start < 1:
        raise ValueError("source frame IDs required")
    if end < start or next_start != end + 1:
        raise ValueError("inconsistent hand ownership interval")
    expected = {"before_start": start - 1, "first_posts": start,
                "before_next_posts": end, "next_posts": next_start}
    if (len(review["evidence"]) != 4 or
            {e["kind"]: e["frame"] for e in review["evidence"]} != expected):
        raise ValueError("four matching boundary evidence frames required")
    if any(frame not in index for frame in range(start - 1, next_start + 1)):
        raise ValueError("missing boundary/index frames")
    if not permitted(Decimal(index[start - 1]["pts_seconds"]),
                     Decimal(index[next_start]["pts_seconds"]), plan):
        raise ValueError("hand and boundary context must remain in development")


def bind(audit, plan_path, review_path, entry, exit_pool, output):
    for folder in (audit, entry, exit_pool):
        if verify_sha256sums(folder):
            raise ValueError("input integrity failure")
    review = json.loads(review_path.read_text())
    plan = json.loads(plan_path.read_text())
    audit_hash = sha256_file(audit / "report.json")
    if (audit_hash != review["audit_sha256"] or
            audit_hash != plan["source_audit_sha256"]):
        raise ValueError("review/plan references a different recording")
    index_lines = (audit / "frame_index.jsonl").read_text().splitlines()
    index_rows = [json.loads(line) for line in index_lines]
    index = {r["global_frame"]: r for r in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("duplicate source frame identity")
    validate_interval(review, index, plan)
    pools = {"entry": entry, "exit": exit_pool}
    evidence = []
    for item in review["evidence"]:
        folder = pools[item["pool"]]
        manifest = json.loads((folder / "samples.json").read_text())
        if manifest["audit_sha256"] != audit_hash:
            raise ValueError("boundary image belongs to another recording")
        row = next(r for r in manifest["samples"]
                   if r["global_frame"] == item["frame"])
        path = (folder / row["file"]).resolve()
        if (not path.is_relative_to(folder.resolve()) or
                sha256_file(path) != row["sha256"]):
            raise ValueError("boundary image mismatch")
        evidence.append({**item, "png_sha256": row["sha256"],
                         "file": str(path), "source_index": index[item["frame"]]})
    owned = [index[i] for i in range(
        review["start_global_frame"], review["end_global_frame"] + 1)]
    result = {**review, "evidence": evidence, "owned_frame_count": len(owned),
              "first_pts": owned[0]["pts_seconds"],
              "last_pts": owned[-1]["pts_seconds"],
              "source_segments": sorted({r["segment"] for r in owned}),
              "boundary_pixels_bound": True, "review_sha256": sha256_file(review_path),
              "strategy_ready": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "registry.json").write_text(json.dumps(result, indent=2))
    write_sha256sums(output)
    print(json.dumps({"hand_id": result["hand_id"], "owned_frames": len(owned),
                      "first_pts": result["first_pts"],
                      "last_pts": result["last_pts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "plan", "review", "entry", "exit-pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bind(args.audit, args.plan, args.review, args.entry, args.exit_pool, args.output)
