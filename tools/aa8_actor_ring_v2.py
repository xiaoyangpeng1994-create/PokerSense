"""AA8 inference-only 4/8-connected suffix union, original source template retained."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_continuous_state import (
    CandidateReader, actor_ring, suffix_components, timer_mask,
)
from tools.aa_visual_candidate import canvas_ok


def union_suffix_components(binary):
    values = suffix_components(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not (5 <= w <= 13 and 8 <= h <= 17 and area >= 20):
            continue
        glyph = (labels[y:y + h, x:x + w] == label).astype(np.float32)
        glyph = cv2.resize(glyph, (24, 24), interpolation=cv2.INTER_AREA)
        yy, xx = np.indices(glyph.shape)
        mass = float(glyph.sum())
        dx = 11.5 - float((glyph * xx).sum()) / mass
        dy = 11.5 - float((glyph * yy).sum()) / mass
        glyph = cv2.warpAffine(glyph, np.float32([[1, 0, dx], [0, 1, dy]]), (24, 24))
        values.append(cv2.GaussianBlur(glyph, (5, 5), 1.))
    # Exact normalized glyph identity deduplication, not parameter selection.
    unique = {hashlib.sha256(v.tobytes()).hexdigest(): v for v in values}
    return list(unique.values())


def actor_ring_v2(image, profile, timer):
    if not canvas_ok(image) or timer is None:
        return {"actor": None,
                "reason": "missing_canvas_or_original_timer_template",
                "timer_suffix_verified": False, "timer_text_verified": False}
    base = actor_ring(image, profile, None)
    scores = [None] * 8
    candidates = [s for s, score in enumerate(base["ring_scores"])
                  if score >= .07]
    for s in candidates:
        x, y, w, h = profile["slots"][s]["avatar"]
        patch = timer_mask(image[y + 20:y + h - 15, x + 15:x + w - 9])
        values = union_suffix_components(patch)
        matches = [float(2 * np.sum(g * timer) / (
            np.sum(g * g) + np.sum(timer * timer)))
                   for g in values]
        scores[s] = max(matches, default=0.)
    accepted = [s for s in candidates if scores[s] >= .85]
    return {**base, "actor": accepted[0] if len(accepted) == 1 else None,
            "reason": "unique_ring_and_suffix_union_candidate" if len(accepted) == 1
            else "ambiguous_or_missing_ring_suffix", "timer_suffix_scores": scores,
            "timer_suffix_verified": len(accepted) == 1, "timer_text_verified": False,
            "inference_connectivity": [4, 8], "suffix_floor": .85,
            "original_template_unchanged": True}


class ContextReaderV2(CandidateReader):
    def read(self, image):
        result = super().read(image)
        actor = actor_ring_v2(image, self.profile, self.timer)
        return {**result, "actor": actor["actor"], "actor_evidence": actor}


def run(first, second, countdown, profile_path, output):
    audit = json.loads((first / "samples.json").read_text())["audit_sha256"]
    first_rows, second_rows = inventory(first, audit), inventory(second, audit)
    countdown_rows = inventory(countdown, audit)
    profile = json.loads(profile_path.read_text())
    source = load(second, second_rows[4755])
    templates = suffix_components(timer_mask(source[181:197, 254:265]))
    if len(templates) != 1:
        raise ValueError("original four-connected source template must remain unique")
    timer = templates[0]
    groups = [(first, first_rows, (1500, 1650, 1950, 2400, 2550, 2700,
                                   2850, 3150, 3180, 3195)),
              (second, second_rows, (4755, 4822, 4823, 4824, 4890)),
              (countdown, countdown_rows, tuple(countdown_rows))]
    rows = []
    for pool, inventory_rows, selected in groups:
        for frame in selected:
            sample = inventory_rows[frame]
            image = load(pool, sample)
            rows.append({"frame": frame, "sha256": sample["sha256"],
                         "pts_seconds": sample["pts_seconds"],
                         "previous": actor_ring(image, profile, timer),
                         "candidate": actor_ring_v2(image, profile, timer)})
    output.mkdir(parents=True, exist_ok=False)
    report = {"observations": rows, "source_template": second_rows[4755],
              "source_template_mask_sha256": hashlib.sha256(
                  timer.tobytes()).hexdigest(),
              "input_manifest_sha256": {str(p): sha(p / "samples.json")
                                        for p in (first, second, countdown)},
              "implementation_sha256": sha(Path(__file__)),
              "profile_sha256": sha(profile_path), "source_connectivity": 4,
              "inference_connectivity": [4, 8], "suffix_floor": .85,
              "independent_holdout": False, "full_visual_acceptance": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames": len(rows), "changed": [r["frame"] for r in rows
                      if r["previous"]["actor"] != r["candidate"]["actor"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "countdown", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.first, args.second, args.countdown, args.profile, args.output)
