"""Replay frozen exploration samples against AA candidate layout, locally."""

import argparse
import json
from pathlib import Path

import cv2

from tools.aa_visual_candidate import AASceneCandidate, render_geometry, table_map
from tools.capture_card_calibration.hashing import verify_sha256sums, write_sha256sums
from tools.wpk_video_dataset import read_image


def run(exploration, profile_path, output):
    if verify_sha256sums(exploration):
        raise ValueError("exploration integrity failure")
    queue = json.loads((exploration / "review_queue.json").read_text(encoding="utf-8"))
    if (queue["source_sha256"] !=
            "2638c3ea894fa6a342947b9c4a06b5d7746a6512360f4ef7d79a387fe89da82f"
            or [r["source_frame_index"] for r in queue["frames"]]
            != list(range(0, 41339, 900))):
        raise ValueError("scene review applies only to the frozen AA exploration batch")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mapping = table_map(profile)
    reference = read_image(exploration / "frame_000000.png")
    model = AASceneCandidate(reference, profile)
    output.mkdir(parents=True, exist_ok=False)
    (output / "layout.snapshot.json").write_bytes(profile_path.read_bytes())
    (output / "table_map.candidate.json").write_text(
        mapping.to_json(), encoding="utf-8")
    results = []
    for row in queue["frames"]:
        path = (exploration / row["candidate_file"]).resolve()
        if not path.is_relative_to(exploration.resolve()):
            raise ValueError("frame path escapes exploration")
        image = read_image(path)
        frame = row["source_frame_index"]
        # Coarse scene truth reviewed from all four contact sheets before inference.
        # No card/mode/participation labels are inferred from this review.
        truth = "NON_TABLE" if frame in (10800, 40500) else "TABLE"
        prediction = model.recognize(image)
        results.append({"source_frame": frame, "reviewed_scene": truth,
                        "exposure": "development_including_template_source",
                        **prediction})
        annotated = render_geometry(image, profile)
        ok, encoded = cv2.imencode(".png", annotated)
        if not ok:
            raise ValueError("PNG encoding failed")
        (output / path.name).write_bytes(encoded.tobytes())
    report = {"source_sha256": queue["source_sha256"], "rows": results,
              "table_accepts": sum(r["reviewed_scene"] == "TABLE"
                                   and r["scene"] == "AA_TABLE_CANDIDATE"
                                   for r in results),
              "non_table_false_accepts": sum(r["reviewed_scene"] == "NON_TABLE"
                                             and r["scene"] == "AA_TABLE_CANDIDATE"
                                             for r in results),
              "vision_ready_for_strategy": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploration", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.exploration, args.profile, args.output)
