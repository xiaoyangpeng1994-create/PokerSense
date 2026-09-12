"""Append one explicitly reviewed DEVELOPMENT wager style, never overwrite a bank."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_money_bank_v2 import labelled_glyphs
from tools.aa8_money_center_bank_v2 import CaptureBank, append_unique
from tools.aa8_special_modes import AUDIT
from tools.aa8_unmarked_money import pot_patch
from tools.aa8_wagergeometry_v2 import AA8WagerReaderV2, WAGERS_V2
from tools.aa_amount_candidate import stack_patch
from tools.aa_pot_candidate import colon_x, pot_mask, region
from tools.aa_pot_evidence import AACenterAmountCandidate
from tools.aa_visual_candidate import AASceneCandidate


REVIEW_SHA = "2893d33a48c59895ac5c530280af621f3cfe6eb9020dd9376aa6ac0e8dc385c2"


def validate_review(row):
    if (row.get("role") != "development" or row.get("global_frame") != 2503
            or row.get("sha256") != REVIEW_SHA):
        raise ValueError("explicit reviewed development frame2503 hash required")


def build(base, first, second, profile_path, output):
    old = json.loads((base / "report.json").read_text())
    if (old.get("role") != "development" or old["floor"] != .90
            or old["margin"] != .05 or old["bank_sha256"] != sha(base / "bank.npz")):
        raise ValueError("intact reviewed bank with unchanged gates required")
    indices = {"first": inventory(first, AUDIT), "second": inventory(second, AUDIT)}
    pools = {"first": first, "second": second}
    manifests = {k: sha(p / "samples.json") for k, p in pools.items()}
    if manifests != old["source_manifests"]:
        raise ValueError("changed development source pools")
    if sha(profile_path) != old["profile_sha256"]:
        raise ValueError("changed geometry")
    validate_review(indices["first"][2503])
    images = {}

    def image(pool, frame):
        key = pool, frame
        if key not in images:
            images[key] = load(pools[pool], indices[pool][frame])
        return images[key]

    capture = CaptureBank()
    cropper = AA8WagerReaderV2(capture, image("first", 1500), coin_smoothing=True)
    x, y, w, h = WAGERS_V2[5]
    diagnostic = cropper.read_label(image("first", 2503)[y:y + h, x:x + w], 5)
    glyphs, reason = labelled_glyphs("58", capture.captured)
    if reason or len(glyphs) != 2:
        raise ValueError("reviewed58 did not yield exactly two unchanged reader glyphs")
    with np.load(base / "bank.npz", allow_pickle=False) as data:
        features = list(data["features"])
        labels = list(map(str, data["labels"]))
    origins = list(old["glyph_provenance"])
    before_count = len(features)
    for position, (label, (feature, box)) in enumerate(glyphs):
        append_unique(features, labels, origins, feature, label, {
            "pool": "first", "frame": 2503, "field": "wager_5", "value": "58",
            "role": "development", "source_sha256": REVIEW_SHA,
            "review": "root independently read full PNG before bank construction",
            "position": position, "digit": label, "glyph_box": box})
    bank = GrayAmountRecognizer(np.array(features), np.array(labels),
                                floor=.90, margin=.05, augment=True)
    profile = json.loads(profile_path.read_text())
    scene = AASceneCandidate(image("first", 1320), profile)
    center_capture = CaptureBank()
    center = AACenterAmountCandidate(
        center_capture, scene, reference=image("first", 1320))
    binary = pot_mask(region(image("first", 1500)))
    colon = colon_x(binary)
    prefix = binary[:, colon - 34:colon + 2]
    checks = []
    for item in old["holdin_development_checks"]:
        im = image(item["pool"], item["frame"])
        if item["field"] == "center":
            center_capture.captured = None
            center.recognize(im)
            patch = center_capture.captured
        elif item["field"] == "pot":
            patch = pot_patch(im, prefix)
        else:
            slot = int(item["field"].split("_")[1])
            patch = stack_patch(im, profile["slots"][slot]["stack"])
        read = asdict(bank.diagnose(patch))
        checks.append({**item, "previous_recognition": item["recognition"],
                       "recognition": read})
    reader = AA8WagerReaderV2(bank, image("first", 1500), coin_smoothing=True)
    probes = [{"frame": f, "pool": p, "source_sha256": indices[p][f]["sha256"],
               "read": reader.read(image(p, f), scene_supported=True,
                                   unobstructed=True)}
              for p, f in (("first", 1320), ("first", 1500), ("first", 2070),
                           ("first", 2400), ("first", 2490), ("first", 2503),
                           ("first", 2550), ("first", 2700), ("second", 3390),
                           ("second", 4755), ("second", 4890))]
    regressions = [v for v in checks if v["previous_recognition"]["value"] is not None
                   and v["recognition"]["value"] != v["previous_recognition"]["value"]]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "bank.npz", features=np.array(features),
                        labels=np.array(labels))
    report = {"schema": "aa8_money_v2_wager58", "role": "development", "floor": .90,
              "margin": .05, "augment": True, "bank_sha256": sha(output / "bank.npz"),
              "base_bank_sha256": old["bank_sha256"],
              "base_report_sha256": sha(base / "report.json"),
              "glyph_provenance": origins, "source_manifests": old["source_manifests"],
              "profile_sha256": sha(profile_path),
              "training_glyph_count": len(features),
              "added_glyphs": len(features) - before_count,
              "capture_diagnostic": diagnostic,
              "holdin_development_checks": checks, "regressions": regressions,
              "wager_probes": probes, "independent_accuracy": False,
              "implementation_sha256": sha(Path(__file__)),
              "crop_implementation_sha256": sha(Path(__file__).with_name(
                  "aa8_wagergeometry_v2.py")),
              "full_visual_acceptance": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"added": report["added_glyphs"], "existing_checks": len(checks),
                      "regressions": len(regressions), "probe_slot5": [
                          [r["frame"], r["read"]["wagers"]["5"]] for r in probes]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "first", "second", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.first, args.second, args.profile, args.output)
