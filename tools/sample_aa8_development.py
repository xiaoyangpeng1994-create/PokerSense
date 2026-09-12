"""Extract development-only review frames using the frozen segmented frame index."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.aa_window_sheet import field_strip
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.aa8_holdout_plan import validate_freeze


def permitted(start, end, plan):
    if start < 0 or end <= start:
        return False
    return any(r["role"] == "development" and Decimal(r["start_inclusive"]) <= start
               and end <= Decimal(r["end_exclusive"]) for r in plan["ranges"])


def boundary_permission(start, end, step, every_frame, plan, freeze_path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if validate_freeze(freeze, freeze_path.parent):
        raise ValueError("invalid or changed freeze before boundary review")
    allowed = any(r["role"] == "holdout_candidate" and
                  Decimal(r["start_inclusive"]) <= start < end <= Decimal(
                      r["end_exclusive"]) for r in plan["ranges"])
    if not allowed:
        raise ValueError("boundary review must remain wholly in reserved holdout")
    if (every_frame or step < 1) and end - start > 3:
        raise ValueError("dense boundary refinement limited to three seconds")
    return {"id": freeze["id"], "sha256": sha256_file(freeze_path)}


def sample(audit, plan_path, normalization_path, output, start=0, end=300, step=5,
           boundary_freeze=None):
    every_frame = step == "frame"
    start, end = (Decimal(str(v)) for v in (start, end))
    step = Decimal("1") if every_frame else Decimal(str(step))
    if not all(v.is_finite() for v in (start, end, step)) or step <= 0:
        raise ValueError("finite positive sampling step required")
    if verify_sha256sums(audit):
        raise ValueError("audit integrity failure")
    plan = json.loads(plan_path.read_text())
    if plan["source_audit_sha256"] != sha256_file(audit / "report.json"):
        raise ValueError("split plan belongs to another audit")
    freeze = None
    role = "development"
    if boundary_freeze is not None:
        freeze = boundary_permission(
            start, end, step, every_frame, plan, boundary_freeze)
        role = "holdout"
    elif not permitted(start, end, plan):
        raise ValueError("requested window is not entirely development")
    source = json.loads((audit / "report.json").read_text())
    if source["state"] != "verified":
        raise ValueError("verified source audit required")
    index_lines = (audit / "frame_index.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in index_lines]
    selected = {}
    next_time = start
    for row in rows:
        pts = Decimal(row["pts_seconds"])
        if start <= pts < end and (every_frame or pts >= next_time):
            selected[(row["segment"], row["local_frame"])] = row
            next_time += step
    normalization = NormalizationConfig.from_json(normalization_path.read_text())
    root = Path(source["recording"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    frames, strips = [], []
    for segment in source["segments"]:
        keys = [key for key in selected if key[0] == segment["file"]]
        if not keys:
            continue
        path = (root / segment["file"]).resolve()
        if path.parent != root:
            raise ValueError("segment escapes source root")
        before = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
                segment["size_bytes"], segment["mtime_ns"]):
            raise ValueError("source attributes differ from frozen audit")
        capture = cv2.VideoCapture(str(path))
        try:
            for local_frame in range(segment["decoded_frames"]):
                ok, raw = capture.read()
                if not ok:
                    raise ValueError("unexpected local decode end")
                key = (segment["file"], local_frame)
                if key not in selected:
                    continue
                row = selected[key]
                canvas = normalize(raw, normalization)
                target = output / "frames" / f"frame_{row['global_frame']:06d}.png"
                ok, encoded = cv2.imencode(".png", canvas)
                if not ok:
                    raise ValueError("PNG encoding failed")
                target.write_bytes(encoded.tobytes())
                frames.append({**row, "file": target.relative_to(output).as_posix(),
                               "sha256": sha256_file(target), "role": role,
                               "hand_id": None, "participation": "UNKNOWN"})
                strip = field_strip(canvas, row["global_frame"])
                cv2.putText(strip, f"PTS {row['pts_seconds']}s", (305, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
                strips.append(strip)
        finally:
            capture.release()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during extraction")
        print(f"Sampled {segment['file']}", flush=True)
    if len(frames) != len(selected):
        raise ValueError("sample inventory mismatch")
    for index in range(0, len(strips), 8):
        ok, encoded = cv2.imencode(".png", np.vstack(strips[index:index + 8]))
        if not ok:
            raise ValueError("sheet encoding failed")
        (output / f"sheet_{index // 8:02d}.png").write_bytes(encoded.tobytes())
    if boundary_freeze is not None and validate_freeze(
            json.loads(boundary_freeze.read_text(encoding="utf-8")),
            boundary_freeze.parent):
        raise ValueError("freeze changed during boundary extraction; retain failure")
    (output / "samples.json").write_text(json.dumps({
        "audit_sha256": sha256_file(audit / "report.json"),
        "plan_sha256": sha256_file(plan_path),
        "normalization_sha256": sha256_file(normalization_path),
        "range_seconds": [str(start), str(end)],
        "step_seconds": None if every_frame else str(step),
        "sampling_mode": "every_frame" if every_frame else "interval",
        "samples": frames, "hand_boundaries_verified": False,
        "boundary_review_only": boundary_freeze is not None, "freeze": freeze,
        "source_mapping": "sequential per-segment decoding and frozen global index"
    }, indent=2))
    write_sha256sums(output)
    print(json.dumps({"samples": len(frames), "range": [str(start), str(end)]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "plan", "normalization", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key, default in (("start", "0"), ("end", "300"), ("step", "5")):
        parser.add_argument("--" + key, default=default)
    parser.add_argument("--boundary-freeze", type=Path)
    args = parser.parse_args()
    sample(args.audit, args.plan, args.normalization, args.output,
           args.start, args.end, args.step, args.boundary_freeze)
