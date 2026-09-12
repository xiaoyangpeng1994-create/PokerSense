"""Hash and fully decode finalized AA recording segments without rewriting them."""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums


def validate_segments(folder, rows):
    if not rows:
        raise ValueError("empty segment list")
    result = []
    previous_end = None
    for index, row in enumerate(rows):
        if len(row) != 3 or row[0] != f"segment_{index:04d}.mkv":
            raise ValueError("unexpected segment name/order")
        path = (folder / row[0]).resolve(strict=True)
        if path.parent != folder.resolve():
            raise ValueError("segment escapes recording folder")
        start, end = map(Decimal, row[1:])
        if not start.is_finite() or not end.is_finite() or start < 0 or end <= start:
            raise ValueError("invalid segment timestamps")
        gap = None if previous_end is None else start - previous_end
        if gap is not None and (gap < 0 or gap > Decimal("0.002")):
            raise ValueError("segment-list overlap or gap exceeds 2ms")
        result.append({"file": row[0], "csv_start": str(start), "csv_end": str(end),
                       "csv_gap_seconds": str(gap) if gap is not None else None})
        previous_end = end
    actual = {p.name for p in folder.glob("segment_*.mkv")}
    if actual != {r["file"] for r in result}:
        raise ValueError("segment list does not match files")
    return result


def run(recording, output):
    recording = recording.resolve(strict=True)
    status_path = recording / "status.json"
    csv_path = recording / "segments.csv"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["state"] != "stopped" or status["exit_code"] != 0:
        raise ValueError("recording must have stopped cleanly")
    inputs = {str(p): sha256_file(p) for p in (status_path, csv_path)}
    csv_rows = list(csv.reader(csv_path.read_text().splitlines()))
    rows = validate_segments(recording, csv_rows)
    output.mkdir(parents=True, exist_ok=False)
    result = {"state": "running", "recording": str(recording),
              "metadata_hashes": inputs,
              "segments": [], "decoded_frames": 0, "strategy_ready": False,
              "independent_hand_validation": False, "large_frame_gaps": []}
    previous_pts = None
    try:
        with (output / "frame_index.jsonl").open("x", encoding="utf-8") as frame_index:
            for row in rows:
                path = recording / row["file"]
                before = path.stat()
                digest = sha256_file(path)
                print(f"Hash/decode {row['file']}", flush=True)
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-err_detect", "explode", "-threads", "2",
                    "-select_streams", "v:0", "-show_frames", "-show_entries",
                    "frame=best_effort_timestamp_time,pkt_duration_time,width,height",
                    "-of", "json", str(path)], capture_output=True, text=True,
                    encoding="utf-8", timeout=180)
                (output / (row["file"] + ".decode.log")).write_text(
                    probe.stderr, encoding="utf-8")
                if probe.returncode or probe.stderr.strip():
                    raise ValueError(f"strict decoder reported an error: {row['file']}")
                frames = json.loads(probe.stdout).get("frames", [])
                if not frames:
                    raise ValueError("no decoded frames")
                first_pts = None
                for local_index, frame in enumerate(frames):
                    if (frame.get("width"), frame.get("height")) != (1920, 1080):
                        raise ValueError("unexpected decoded frame dimensions")
                    pts = Decimal(frame["best_effort_timestamp_time"])
                    if not pts.is_finite() or pts < 0:
                        raise ValueError("invalid decoded PTS")
                    if previous_pts is not None:
                        gap = pts - previous_pts
                        if gap <= 0:
                            raise ValueError("non-increasing decoded timeline")
                        if gap > Decimal("0.067"):
                            result["large_frame_gaps"].append({
                                "global_frame": result["decoded_frames"],
                                "gap": str(gap)})
                    if first_pts is None:
                        first_pts = pts
                    frame_index.write(json.dumps({
                        "segment": row["file"], "local_frame": local_index,
                        "global_frame": result["decoded_frames"],
                        "pts_seconds": str(pts)
                    }) + "\n")
                    previous_pts = pts
                    result["decoded_frames"] += 1
                if abs(first_pts - Decimal(row["csv_start"])) > Decimal("0.002"):
                    raise ValueError("first frame disagrees with segment-list start")
                if previous_pts >= Decimal(row["csv_end"]):
                    raise ValueError("last frame outside segment interval")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                        after.st_size, after.st_mtime_ns):
                    raise ValueError("source changed during verification")
                result["segments"].append({
                    **row, "sha256": digest,
                    "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                    "decoded_frames": len(frames), "first_pts": str(first_pts),
                    "last_pts": str(previous_pts)})
        if any(sha256_file(Path(p)) != h for p, h in inputs.items()):
            raise ValueError("recording metadata changed")
        progress_path = recording / "progress.log"
        reported = re.findall(r"^frame=(\d+)$", progress_path.read_text(), re.MULTILINE)
        result["recorder_reported_frames"] = int(reported[-1]) if reported else None
        result["frame_count_matches_recorder"] = (
            result["decoded_frames"] == result["recorder_reported_frames"])
        result["total_bytes"] = sum(s["size_bytes"] for s in result["segments"])
        result["duration_seconds"] = rows[-1]["csv_end"]
        matched = result["frame_count_matches_recorder"]
        result["state"] = "verified" if matched else "partial"
    except Exception as error:
        result.update(state="failed", failure=str(error))
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        write_sha256sums(output)
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("segments", "metadata_hashes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recording, args.output)
