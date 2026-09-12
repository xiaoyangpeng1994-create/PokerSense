"""Add independently read DEVELOPMENT center-number styles to reviewed V2 bank."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import (
    GrayAmountRecognizer, GrayRead,
)
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_money_bank_v2 import labelled_glyphs
from tools.aa8_special_modes import AUDIT
from tools.aa8_unmarked_money import pot_patch
from tools.aa_amount_candidate import stack_patch
from tools.aa_pot_candidate import colon_x, pot_mask, region
from tools.aa_pot_evidence import AACenterAmountCandidate
from tools.aa_visual_candidate import AASceneCandidate


# Entire predeclared frame list read directly, NOT inferred from title or predictions.
CENTER_REVIEWS = {1320: "14", 1500: "14", 1980: "14", 2070: "115",
                  2400: "115", 2700: "115", 2850: "289", 3150: "289",
                  3390: "16", 4402: "114", 4755: "190"}


class CaptureBank:
    """Exercise production center crop unchanged; never supplies a prediction."""
    def __init__(self):
        self.captured = None

    def diagnose(self, value):
        self.captured = value.copy()
        return GrayRead(None, None, "capture_only", ())


def append_unique(features, labels, origins, feature, label, origin):
    digest = hashlib.sha256(feature.tobytes()).hexdigest()
    matches = [i for i, f in enumerate(features)
               if hashlib.sha256(f.tobytes()).hexdigest() == digest]
    if any(labels[i] != label for i in matches):
        raise ValueError("identical feature has conflicting digit labels")
    if matches:
        return False
    features.append(feature)
    labels.append(label)
    origins.append({**origin, "feature_sha256": digest})
    return True


def build(base, first, second, profile_path, output):
    old = json.loads((base / "report.json").read_text())
    if (old.get("role") != "development" or old["floor"] != .90
            or old["margin"] != .05 or sha(base / "bank.npz") != old["bank_sha256"]):
        raise ValueError("intact reviewed development V2 bank required")
    pools = {"first": first, "second": second}
    indices = {k: inventory(p, AUDIT) for k, p in pools.items()}
    manifests = {k: sha(p / "samples.json") for k, p in pools.items()}
    if manifests != old["source_manifests"]:
        raise ValueError("changed development source pools")
    if sha(profile_path) != old["profile_sha256"]:
        raise ValueError("changed layout")
    profile = json.loads(profile_path.read_text())
    reference = load(first, indices["first"][1320])
    scene = AASceneCandidate(reference, profile)
    capture = CaptureBank()
    reader = AACenterAmountCandidate(capture, scene, reference=reference)
    arrays = np.load(base / "bank.npz", allow_pickle=False)
    if len(arrays["features"]) != len(old["glyph_provenance"]):
        raise ValueError("base glyph provenance count mismatch")
    features, labels, origins = [], [], []
    for feature, label, origin in zip(
            arrays["features"], arrays["labels"], old["glyph_provenance"]):
        append_unique(features, labels, origins, feature, str(label), origin)
    retained_base = len(features)
    diagnostics, center_patches = [], []
    for frame, value in CENTER_REVIEWS.items():
        key = "first" if frame < 3320 else "second"
        source = indices[key][frame]
        image = load(pools[key], source)
        capture.captured = None
        crop_result = reader.recognize(image)
        glyphs, reason = labelled_glyphs(value, capture.captured)
        row = {"pool": key, "frame": frame, "value": value, "field": "center",
               "source_sha256": source["sha256"], "crop_reason": crop_result["reason"],
               "exclusion_reason": reason, "glyph_count": len(glyphs)}
        added = 0
        for position, (label, (feature, box)) in enumerate(glyphs):
            added += append_unique(features, labels, origins, feature, label,
                                   {**row, "position": position, "digit": label,
                                    "glyph_box": box})
        diagnostics.append({**row, "new_unique_glyphs": added})
        center_patches.append((row, capture.captured))
    bank = GrayAmountRecognizer(np.array(features), np.array(labels),
                                floor=.90, margin=.05, augment=True)
    checks = [{**row, "recognition": asdict(bank.diagnose(image_patch))}
              for row, image_patch in center_patches]
    pot_reference = load(first, indices["first"][1500])
    binary = pot_mask(region(pot_reference))
    colon = colon_x(binary)
    if colon is None or colon < 34:
        raise ValueError("missing reviewed title prefix")
    prefix = binary[:, colon - 34:colon + 2]
    for row in old["holdin_development_checks"]:
        source = indices[row["pool"]][row["frame"]]
        image = load(pools[row["pool"]], source)
        if row["field"] == "pot":
            image_patch = pot_patch(image, prefix)
        else:
            slot = int(row["field"].split("_")[1])
            image_patch = stack_patch(image, profile["slots"][slot]["stack"])
        checks.append({**row, "recognition": asdict(bank.diagnose(image_patch))})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "bank.npz", features=np.array(features),
                        labels=np.array(labels))
    report = {"schema": "aa8_money_v2_center", "role": "development",
              "base_report_sha256": sha(base / "report.json"),
              "base_bank_sha256": old["bank_sha256"],
              "bank_sha256": sha(output / "bank.npz"), "floor": .90, "margin": .05,
              "augment": True, "retained_unique_base_glyphs": retained_base,
              "training_glyph_count": len(features), "glyph_provenance": origins,
              "center_field_diagnostics": diagnostics,
              "base_field_diagnostics": old["field_diagnostics"],
              "holdin_development_checks": checks,
              "independent_accuracy": False, "arithmetic_correction": False,
              "source_manifests": old["source_manifests"],
              "profile_sha256": sha(profile_path),
              "implementation_sha256": sha(Path(__file__)),
              "center_crop_implementation_sha256": sha(
                  Path(__file__).with_name("aa_pot_evidence.py")),
              "holdin_summary": {
                  "fields": len(checks),
                  "matched": sum(r["recognition"]["value"] == r["value"]
                                 for r in checks),
                  "unknown": sum(r["recognition"]["value"] is None for r in checks),
                  "wrong": sum(r["recognition"]["value"] not in (None, r["value"])
                               for r in checks)}}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["holdin_summary"], "center": diagnostics,
                      "glyphs": len(features)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "first", "second", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.first, args.second, args.profile, args.output)
