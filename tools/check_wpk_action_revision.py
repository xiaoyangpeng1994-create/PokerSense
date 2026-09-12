"""Compare the opt-in action revision on earlier, separately reviewed points."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.probe_wpk_field_candidate import candidate_score
from tools.wpk_combined_vision import from_artifacts
from tools.wpk_field_candidate import crop
from tools.wpk_video_dataset import pixels_digest, read_image


def run(corpus, output):
    if output.exists():
        raise ValueError("preserve previous checks")
    repo = Path(__file__).resolve().parents[1]
    spec_path = repo / (
        "tests/fixtures/wpk_reference_hands/check_variant_source001_v1.json")
    spec = json.loads(spec_path.read_text())
    folder = corpus.parent / "capture_card_calibration_20260903/normalized/frames"
    template_path = folder / spec["source_frame_file"]
    if sha256_file(template_path) != spec["image_sha256"]:
        raise ValueError("check template changed")
    bank = corpus / "gray_amount/session001_proposals_v1"
    before = from_artifacts(corpus, bank).actions
    patch = crop(read_image(template_path), spec["rect"])
    after = from_artifacts(corpus, bank, check_variant=patch,
                           badge_scale_tolerance=True).actions
    bound = {}
    for name in ("features_v2_final", "features_v2_transfer_final"):
        index_path = corpus / "field_candidates" / name / "bound-samples.json"
        for sample in json.loads(index_path.read_text()):
            bound[sample["source_frame"]] = sample
    results, details = {}, []
    groups = ("aq_observation_v1", "field_followup_v1", "field_features_v2_followup")
    for name in groups:
        path = repo / "tests/fixtures/wpk_reference_hands" / f"{name}.json"
        truth = json.loads(path.read_text())
        counts = {"before": Counter(), "after": Counter()}
        for point in truth["checkpoints"]:
            sample = bound[point["source_frame"]]
            image_path = Path(sample["path"])
            image = read_image(image_path)
            if (sha256_file(image_path) != sample["image_sha256"]
                    or pixels_digest(image) != sample["pixel_sha256"]):
                raise ValueError("reviewed frame changed")
            old, new = before.recognize(image), after.recognize(image)
            for seat, expected in enumerate(point["actions"]):
                old_result = candidate_score(expected, old[str(seat)])
                new_result = candidate_score(expected, new[str(seat)])
                counts["before"][old_result] += 1
                counts["after"][new_result] += 1
                details.append({"group": name, "frame": point["source_frame"],
                                "seat": seat, "expected": expected,
                                "before": old_result,
                                "after": new_result, "after_read": new[str(seat)]})
        results[name] = counts
    errors = [r for r in details if r["after"] in ("wrong", "false_accept")]
    lost = [r for r in details if r["before"] == "correct" and r["after"] != "correct"]
    report = {"groups": results, "checks": details, "independent_holdout": False,
              "new_errors": errors, "lost_correct_accepts": lost}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    sources = (Path(__file__), spec_path, template_path,
               repo / "tools/wpk_field_features_v2.py",
               repo / "tools/wpk_combined_vision.py")
    for path in sources:
        shutil.copy2(path, output / path.name)
    write_sha256sums(output)
    return {"groups": results, "errors": len(report["new_errors"]),
            "lost_correct": len(report["lost_correct_accepts"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.output), indent=2))
