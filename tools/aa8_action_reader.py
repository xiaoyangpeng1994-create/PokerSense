"""Offline AA8 glyph development experiment; no legal actions or live advice."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa_action_candidate import text_mask, text_patch
from tools.aa_visual_candidate import AASceneCandidate, canvas_ok, validate_layout
from tools.verify_aa8_actions import bind
from tools.wpk_video_dataset import read_image


LABELS = ("check", "call", "aggressive", "fold", "all_in")
TEMPLATES = {"check": (2100, 1), "call": (1560, 4),
             "aggressive": (1470, 3), "fold": (1830, 0),
             "all_in": (3168, 4), "all_in_opponent": (3120, 1)}


def badge_text(image, avatar):
    """White badge lettering, excluding the orange pill and player name."""
    x, y, width, _ = avatar
    center = x + width // 2
    patch = image[y - 39:y - 18, center - 30:center + 30]
    return cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                       (0, 0, 175), (179, 95, 255))


def mask(image, avatar, label):
    if label == "fold":
        return text_mask(text_patch(image, avatar, label), label)
    if label in ("fold", "all_in"):
        glyph = text_mask(text_patch(image, avatar, label), label)
        return normalize_glyph(glyph, label)
    x, y, width, _ = avatar
    # Completed badge ABOVE the name; never the large Hero action control.
    patch = image[y - 39:y - 18, x + width // 2 - 34:x + width // 2 + 34]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    limits = {"check": ((40, 100, 180), (85, 255, 255)),
              "call": ((85, 100, 180), (125, 255, 255)),
              "aggressive": ((5, 100, 180), (35, 255, 255))}
    return normalize_glyph(cv2.inRange(hsv, *limits[label]), label)


def normalize_glyph(value, label):
    """Remove border animation/islands and align glyph support, not avatar origin."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(value)
    if count <= 1:
        return np.zeros((28, 72), np.uint8)
    if label in ("check", "call", "aggressive"):
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        value = np.where(components == component, 255, 0).astype(np.uint8)
    elif label == "all_in":
        border = set(np.unique(np.concatenate((components[0], components[-1],
                                               components[:, 0], components[:, -1]))))
        value = np.where(np.isin(components, list(border)), 0, value).astype(np.uint8)
    points = cv2.findNonZero(value)
    if points is None or len(points) < 25:
        return np.zeros((28, 72), np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    return np.pad(cv2.resize(
        value[y:y + h, x:x + w], (64, 20), interpolation=cv2.INTER_LINEAR), 4)


def similarity(value, template):
    first = cv2.GaussianBlur(value.astype(np.float32), (7, 7), 1.0)
    second = cv2.GaussianBlur(template.astype(np.float32), (7, 7), 1.0)
    if min(float(first.std()), float(second.std())) < 1:
        return 0.
    return max(0., float(cv2.matchTemplate(
        np.pad(first, 2), second, cv2.TM_CCOEFF_NORMED).max()))


class AA8GlyphReader:
    def __init__(self, profile, images, scene_reference, floor=.90):
        validate_layout(profile)
        if len(profile["slots"]) != 8 or not 0 < floor <= 1:
            raise ValueError("explicit eight-seat geometry and valid floor required")
        self.profile, self.floor = profile, floor
        self.scene = AASceneCandidate(scene_reference, profile)
        self.aggressive_text = badge_text(
            images["aggressive"], profile["slots"][3]["avatar"])
        self.text_floor = .80
        self.templates = {}
        for key, (_, slot) in TEMPLATES.items():
            label = "all_in" if key == "all_in_opponent" else key
            if not canvas_ok(images[key]):
                raise ValueError("invalid template canvas")
            glyph = mask(images[key], profile["slots"][slot]["avatar"], label)
            if np.count_nonzero(glyph) < 25:
                raise ValueError("missing glyph template")
            self.templates.setdefault(label, []).append(glyph)

    def recognize(self, image):
        if not canvas_ok(image) or self.scene.recognize(image)["scene"] != (
                "AA_TABLE_CANDIDATE"):
            return {str(i): None for i in range(8)}
        result = {}
        for row in self.profile["slots"]:
            accepted = []
            for label in LABELS:
                value = mask(image, row["avatar"], label)
                if max(similarity(value, template) for template in (
                        self.templates[label])) >= self.floor:
                    accepted.append(label)
            result[str(row["slot"])] = accepted[0] if len(accepted) == 1 else None
            if result[str(row["slot"])] == "aggressive" and similarity(
                    badge_text(image, row["avatar"]), self.aggressive_text) < (
                        self.text_floor):
                result[str(row["slot"])] = None
        return result


class GlyphTransitions:
    """Stable visible changes, not poker events. Unknown does not prove a fold."""

    def __init__(self, stable_frames=2, clear_frames=5):
        if type(stable_frames) is not int or stable_frames < 1:
            raise ValueError("positive stability required")
        if type(clear_frames) is not int or clear_frames < stable_frames:
            raise ValueError("clear stability must be at least confirmation stability")
        self.required = stable_frames
        self.clear_required = clear_frames
        self.last_frame = None
        self.state = {}

    def observe(self, frame, values):
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("duplicate/backwards frame")
        if self.last_frame is None or frame != self.last_frame + 1:
            self.state.clear()
        self.last_frame = frame
        events = []
        for slot, value in values.items():
            previous, count, emitted = self.state.get(slot, (None, 0, None))
            count = count + 1 if value == previous else 1
            threshold = self.clear_required if value is None else self.required
            if count >= threshold and value != emitted:
                emitted = value
                if value is not None:
                    events.append({"frame": frame, "slot": int(slot), "glyph": value})
            self.state[slot] = (value, count, emitted)
        return events


def compare(events, review, latency=2):
    expected = []
    for street in review["streets"]:
        for action in street["actions"]:
            glyph = ("all_in" if action.get("all_in") else "aggressive"
                     if action["kind"] in ("bet", "raise") else action["kind"])
            expected.append({"slot": action["slot"], "glyph": glyph,
                             "window": action["window"], "street": street["street"]})
    unused = set(range(len(events)))
    matched, missed = [], []
    for item in expected:
        found = [i for i in sorted(unused) if events[i]["slot"] == item["slot"]
                 and events[i]["glyph"] == item["glyph"]
                 and item["window"][0] < events[i]["frame"] <= (
                     item["window"][1] + latency)]
        if found:
            unused.remove(found[0])
            matched.append({**item, "prediction": events[found[0]]})
        else:
            missed.append(item)
    return {"matched": matched, "missed": missed,
            "unmatched_proposals": [events[i] for i in sorted(unused)],
            "matching_latency_frames": latency}


def run(pool, profile_path, review_path, registry, output):
    implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    bind(review, pool, registry)
    manifest = json.loads((pool / "samples.json").read_text(encoding="utf-8"))
    rows = {r["global_frame"]: r for r in manifest["samples"]}
    ownership = json.loads(registry.read_text(encoding="utf-8"))
    expected_ids = list(range(ownership["start_global_frame"],
                              ownership["end_global_frame"] + 1))
    if sorted(rows) != expected_ids:
        raise ValueError("full contiguous owned hand required")

    def load(frame):
        row = rows[frame]
        path = (pool / row["file"]).resolve()
        if (row["role"] != "development" or not path.is_relative_to(pool.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("non-development or changed frame")
        return read_image(path)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reader = AA8GlyphReader(profile, {k: load(v[0]) for k, v in TEMPLATES.items()},
                            load(review["seed_frame"]))
    tracker, predictions, events = GlyphTransitions(), [], []
    for frame in expected_ids:
        values = reader.recognize(load(frame))
        predictions.append({"frame": frame, "glyphs": values})
        events.extend(tracker.observe(frame, values))
    report = {**compare(events, review), "frame_count": len(predictions),
              "proposal_count": len(events), "template_sources": {
                  k: rows[v[0]] for k, v in TEMPLATES.items()},
              "template_and_evaluation_same_development_hand": True,
              "independent_holdout": False, "strategy_eligible": False,
              "full_visual_acceptance": False, "current_actor": "UNKNOWN",
              "floor": reader.floor, "review_sha256": hashlib.sha256(
                  review_path.read_bytes()).hexdigest(),
              "implementation_sha256": implementation_sha,
              "badge_hsv_min_value": 180, "stable_frames": tracker.required,
              "aggressive_text_floor": reader.text_floor,
              "clear_frames": tracker.clear_required,
              "mask_normalization": "badge_component_bbox; all_in_border_removal",
              "matcher_blur": {"kernel": 7, "sigma": 1.0},
              "template_mask_sha256": {k: [hashlib.sha256(v.tobytes()).hexdigest()
                                           for v in bank]
                                       for k, bank in reader.templates.items()},
              "layout_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
              "audit_sha256": review["audit_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.jsonl").write_text("\n".join(
        json.dumps(r) for r in predictions) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(predictions), "matched": len(report["matched"]),
                      "missed": report["missed"],
                      "unmatched": report["unmatched_proposals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pool", "profile", "review", "registry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.pool, args.profile, args.review, args.registry, args.output)
