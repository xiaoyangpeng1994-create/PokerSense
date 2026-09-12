"""AA8 offline current-frame cards with source-aware temporal invalidation.

The existing WPK source heads are reused without training or AA8 calibration.
Outputs remain candidates. In particular a folded/dim card can stay UNKNOWN.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_card_preflight import transform_card
from tools.aa_card_visibility import face_card_support, locate_face_card


class AA8CardReader:
    def __init__(self, heads, *, preprocessing=None):
        if preprocessing not in (None, "gaussian_050"):
            raise ValueError("only explicit fixed gaussian_050 preprocessing supported")
        self.preprocessing = preprocessing
        if isinstance(heads, (str, Path)):
            heads = load_card_heads(Path(heads))
        self.cards = FusedCardRecognizerAdapter(FusedCardRecognizer(
            heads, rank_floor=.5, suit_floor=.3))
        self.last_frame = self.last_pts = self.source = None
        self.regions = {}
        self.values = {}

    def clear(self):
        self.cards.reset()
        self.regions.clear()
        self.values.clear()

    def _confirm(self, key, value, frame):
        old = self.values.get(key)
        if value is None:
            self.values.pop(key, None)
            return None
        self.values[key] = (value, frame)
        return value if old == (value, frame - 1) else None

    def read(self, image, frame, pts, source):
        """pts is source seconds. Never use wall clock or duplicate frame samples."""
        if (type(frame) is not int or frame < 0 or not isinstance(source, str)
                or not source or isinstance(pts, bool)
                or not isinstance(pts, (int, float))
                or not np.isfinite(pts) or pts < 0):
            self.clear()
            raise ValueError("valid frame/source/time required")
        if (source == self.source and self.last_frame is not None
                and frame <= self.last_frame):
            self.clear()
            raise ValueError("duplicate/backwards source frame")
        gap = (self.last_frame is None or source != self.source
               or frame != self.last_frame + 1 or pts <= self.last_pts
               or pts - self.last_pts > 1.)
        if gap:
            self.clear()
        self.last_frame, self.last_pts, self.source = frame, pts, source
        result = {"frame": frame, "pts_seconds": pts, "source": source,
                  "gap_reset": gap, "hero": None, "board_slots": [None] * 5,
                  "raw_hero": [None] * 2, "raw_board_slots": [None] * 5,
                  "evidence": {}, "strategy_eligible": False,
                  "model_calibrated_for_aa8": False,
                  "preprocessing": self.preprocessing}
        if (not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3)
                or image.dtype != np.uint8):
            self.clear()
            result["reason"] = "unsupported_canvas"
            return result
        timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=pts)
        self.cards.begin_frame(frame, timestamp, (source, "aa8_498x1080"),
                               {"hero": (192, 936, 114, 83),
                                "board": (109, 452, 280, 121)})
        groups = (("hero", (193, 250), 938),
                  ("board", (110, 166, 223, 279, 335), 473))
        confirmed = {}
        for group, xs, y in groups:
            raw, supported = [], []
            for slot, x in enumerate(xs):
                key = (group, slot)
                rect = (x, y, 53, 78)
                if group == "board":
                    rect, reason = locate_face_card(image, rect)
                else:
                    good, reason = face_card_support(image, rect)
                    rect = rect if good else None
                if rect != self.regions.get(key):
                    self.cards.reset(key)
                    self.values.pop(key, None)
                self.regions[key] = rect
                evidence = {"rect": rect, "face_support": reason, "score": None}
                value = None
                if rect is not None:
                    rx, ry, rw, rh = rect
                    crop = image[ry:ry + rh, rx:rx + rw]
                    if self.preprocessing is not None:
                        crop = transform_card(crop, self.preprocessing)
                    read = self.cards.recognize(crop, key)
                    value = str(read.value[0]) if read.value else None
                    evidence["score"] = read.raw_score
                else:
                    self.cards.reset(key)
                raw.append(value)
                supported.append(self._confirm(key, value, frame))
                result["evidence"][f"{group}_{slot}"] = evidence
            result["raw_hero" if group == "hero" else "raw_board_slots"] = raw
            confirmed[group] = supported
        known = [v for values in confirmed.values() for v in values if v is not None]
        if len(set(known)) != len(known):
            self.clear()
            result["reason"] = "duplicate_card_identity_abstention"
            return result
        hero = confirmed["hero"]
        result["hero"] = hero if all(hero) else None
        result["board_slots"] = confirmed["board"]
        result["reason"] = "current_frame_candidates_not_aa8_calibrated"
        return result


def run(source, target, heads, output):
    audit = json.loads((source / "samples.json").read_text())["audit_sha256"]
    pools = [(source, inventory(source, audit),
              (1470, 1980, 2070, 2400, 2820, 3060, 3180)),
             (target, inventory(target, audit), (4890, 4950))]
    reader = AA8CardReader(heads)
    observations = []
    for pool, rows, anchors in pools:
        for anchor in anchors:
            for frame in range(anchor, anchor + 5):
                row = rows[frame]
                result = reader.read(load(pool, row), frame, float(row["pts_seconds"]),
                                     str(pool.resolve()))
                observations.append({**result, "sha256": row["sha256"],
                                     "anchor": anchor})
    output.mkdir(parents=True, exist_ok=False)
    report = {"observations": observations, "frames": len(observations),
              "heads_sha256": sha(heads), "implementation_sha256": sha(Path(__file__)),
              "source_manifest_sha256": sha(source / "samples.json"),
              "target_manifest_sha256": sha(target / "samples.json"),
              "independent_holdout": False, "full_visual_acceptance": False,
              "heads_retrained": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{k: r[k] for k in ("frame", "hero", "board_slots", "reason")}
                      for r in observations if r["frame"] == r["anchor"] + 4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "heads", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.target, args.heads, args.output)
