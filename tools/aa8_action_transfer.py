"""Frozen V5 glyph inference on disjoint development frames, without labels."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.aa8_action_reader import AA8GlyphReader, GlyphTransitions, TEMPLATES, compare
from tools.wpk_video_dataset import read_image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(pool, audit):
    manifest = json.loads((pool / "samples.json").read_text())
    if manifest["audit_sha256"] != audit:
        raise ValueError("wrong recording")
    rows = manifest["samples"]
    ids = [r["global_frame"] for r in rows]
    if not ids or ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError("contiguous unique ordered frames required")
    if any(r["role"] != "development" for r in rows):
        raise ValueError("development frames only")
    return {r["global_frame"]: r for r in rows}


def load(pool, row):
    path = (pool / row["file"]).resolve()
    if not path.is_relative_to(pool.resolve()) or sha(path) != row["sha256"]:
        raise ValueError("frame path/hash mismatch")
    return read_image(path)


def run(baseline, source, target, profile, output):
    frozen = json.loads(baseline.read_text())
    if (frozen["floor"] != .90 or frozen["stable_frames"] != 2
            or frozen["clear_frames"] != 5 or sha(profile) != frozen["layout_sha256"]):
        raise ValueError("requires frozen V5 parameters and layout")
    source_rows = inventory(source, frozen["audit_sha256"])
    target_rows = inventory(target, frozen["audit_sha256"])
    if set(source_rows) & set(target_rows):
        raise ValueError("transfer frames overlap template development hand")
    images = {}
    for key, (frame, _) in TEMPLATES.items():
        if source_rows[frame] != frozen["template_sources"][key]:
            raise ValueError("template source changed")
        images[key] = load(source, source_rows[frame])
    reader = AA8GlyphReader(json.loads(profile.read_text()), images,
                            load(source, source_rows[1320]), floor=frozen["floor"])
    if reader.text_floor != frozen.get("aggressive_text_floor"):
        raise ValueError("baseline lacks matching aggressive text gate")
    masks = {k: [hashlib.sha256(v.tobytes()).hexdigest() for v in bank]
             for k, bank in reader.templates.items()}
    if masks != frozen["template_mask_sha256"]:
        raise ValueError("template masks changed")
    tracker = GlyphTransitions(frozen["stable_frames"], frozen["clear_frames"])
    predictions, events = [], []
    for frame, row in target_rows.items():
        values = reader.recognize(load(target, row))
        predictions.append({"frame": frame, "glyphs": values})
        events.extend(tracker.observe(frame, values))
    report = {"baseline_sha256": sha(baseline), "layout_sha256": sha(profile),
              "reader_sha256": sha(Path(__file__).with_name("aa8_action_reader.py")),
              "target_manifest_sha256": sha(target / "samples.json"),
              "source_manifest_sha256": sha(source / "samples.json"),
              "scene_reference_sha256": source_rows[1320]["sha256"],
              "template_masks_match_v5": True, "template_mask_sha256": masks,
              "floor": frozen["floor"], "stable_frames": tracker.required,
              "aggressive_text_floor": reader.text_floor,
              "clear_frames": tracker.clear_required, "frame_count": len(predictions),
              "first_frame": min(target_rows), "last_frame": max(target_rows),
              "events": events, "labels_used_for_prediction": False,
              "independent_holdout": False, "hand_boundaries_verified": False,
              "full_visual_acceptance": False, "strategy_eligible": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "predictions.jsonl").write_text("\n".join(
        json.dumps(r) for r in predictions) + "\n")
    print(json.dumps({"frames": len(predictions), "events": events}))


def review_predictions(report_path, review_path, pool, output):
    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["role"] != "development" or sha(pool / "samples.json") != (
            report["target_manifest_sha256"]):
        raise ValueError("review target mismatch")
    rows = inventory(pool, review["audit_sha256"])
    frames = set()
    for street in review["streets"]:
        for action in street["actions"]:
            frames.update(action["window"])
    for action in review["unmarked_action_evidence"]:
        frames.update(action["window"])
    frames.update(r["frame"] for r in review["negative_observations"])
    for frame in frames:
        load(pool, rows[frame])
    result = {**compare(report["events"], review),
              "prediction_report_sha256": sha(report_path),
              "review_sha256": sha(review_path),
              "evidence": {str(f): rows[f] for f in sorted(frames)},
              "unmarked_action_evidence": review["unmarked_action_evidence"],
              "independent_holdout": False, "full_visual_acceptance": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"matched": len(result["matched"]), "missed": result["missed"],
                      "extra": result["unmatched_proposals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "source", "target", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.source, args.target, args.profile, args.output)
