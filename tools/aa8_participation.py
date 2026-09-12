"""AA8 positive participation cues and conservative temporal roster candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_action_transfer import inventory, load
from tools.aa_seat_candidate import avatar_patch, plus_mask


def mask(patch, kind):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if kind == "back":
        return ((cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) > 0)
                | (cv2.inRange(hsv, (165, 80, 60), (179, 255, 255)) > 0))
    return cv2.inRange(hsv, (35, 80, 90), (95, 255, 255)) > 0


def dice(a, b):
    denominator = np.count_nonzero(a) + np.count_nonzero(b)
    return float(2 * np.count_nonzero(a & b) / denominator) if denominator else 0.


def white(patch):
    return cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                       (0, 0, 110), (179, 80, 255)) > 0


def waiting_yellow(patch):
    return cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                       (15, 80, 110), (40, 255, 255)) > 0


def back_patch(image, row):
    x, y, _, height = row["avatar"]
    bx = x + 49 if row["slot"] <= 3 else x + 8
    return image[y + height - 24:y + height - 2, bx:bx + 22]


class ParticipationReader:
    """Fixed source templates are development candidates, not calibrated rates."""
    def __init__(self, profile, opening, waiting, floor=.90, waiting_next=None):
        if (profile.get("canvas") != [498, 1080] or profile.get("hero_slot") != 4
                or [row["slot"] for row in profile["slots"]] != list(range(8))):
            raise ValueError("AA eight-slot geometry required")
        self.profile, self.floor = profile, floor
        self.empty = plus_mask(avatar_patch(opening, profile["slots"][6]["avatar"])) > 0
        self.back = mask(opening[198:220, 262:284], "back")
        self.back_white = white(opening[198:220, 262:284])
        self.back_bank = {}
        for row in profile["slots"]:
            if row["slot"] == 4:
                continue
            source_row = profile["slots"][5] if row["slot"] == 6 else row
            patch = back_patch(opening, source_row)
            self.back_bank[row["slot"]] = mask(patch, "back"), white(patch)
        self.waiting = mask(avatar_patch(waiting, profile["slots"][6]["stack"]), "wait")
        self.waiting_next = None
        self.waiting_next_pixel_sha256 = None
        if waiting_next is not None:
            patch = avatar_patch(waiting_next, profile["slots"][6]["stack"])
            self.waiting_next = waiting_yellow(patch)
            if np.count_nonzero(self.waiting_next) < 15:
                raise ValueError("positive waiting-next yellow text template required")
            self.waiting_next_pixel_sha256 = hashlib.sha256(
                waiting_next.tobytes()).hexdigest()
        if min(np.count_nonzero(v) for v in (self.empty, self.back, self.waiting)) < 15:
            raise ValueError("positive empty/back/wait templates required")

    def recognize(self, image):
        if not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3):
            return {str(slot): {"cue": "UNKNOWN"} for slot in range(8)}
        result = {}
        for row in self.profile["slots"]:
            slot = row["slot"]
            avatar = avatar_patch(image, row["avatar"])
            patch = back_patch(image, row)
            back = 0.
            if slot != 4:
                red_template, white_template = self.back_bank[slot]
                back = min(dice(mask(patch, "back"), red_template),
                           dice(white(patch), white_template))
            # Hero has a different, full-size pair of backs; do not reuse opponent crop.
            if slot == 4:
                parts = [image[942:1009, left:left + 42] for left in (199, 256)]
                back = min(float(mask(part, "back").mean()) for part in parts)
            empty = dice(plus_mask(avatar) > 0, self.empty)
            green = float(mask(avatar, "wait").mean())
            wait_patch = avatar_patch(image, row["stack"])
            wait_mask = cv2.resize(mask(wait_patch, "wait").astype(np.uint8),
                                   (self.waiting.shape[1], self.waiting.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
            wait = dice(wait_mask, self.waiting)
            wait_next = 0.
            if self.waiting_next is not None:
                yellow = cv2.resize(waiting_yellow(wait_patch).astype(np.uint8),
                                    (self.waiting_next.shape[1],
                                     self.waiting_next.shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0
                wait_next = dice(yellow, self.waiting_next)
            cues = []
            if empty >= self.floor and green > .5:
                cues.append("EMPTY")
            if back >= self.floor:
                cues.append("BACK_CARDS")
            if wait >= self.floor:
                cues.append("WAITING_POST_OR_PASS")
            if wait_next >= self.floor:
                cues.append("WAITING_NEXT_HAND")
            result[str(slot)] = {"cue": cues[0] if len(cues) == 1 else "UNKNOWN",
                                 "conflict": len(cues) > 1,
                                 "scores": {"empty": empty, "back": back, "wait": wait,
                                            "waiting_next": wait_next}}
        return result


class RosterCandidates:
    """No glyph/cards absent => fold or observer inference. Boundary stays external."""
    def __init__(self, stable_frames=2):
        if type(stable_frames) is not int or stable_frames < 1:
            raise ValueError("positive confirmation length required")
        self.required, self.last = stable_frames, None
        self.streak, self.history = {}, {}
        self.hand = None

    def begin_hand(self, hand):
        if not isinstance(hand, str) or not hand:
            raise ValueError("explicit hand identity required")
        if hand != self.hand:
            self.hand, self.streak, self.history = hand, {}, {}

    def observe(self, frame, cues, glyphs=None):
        if type(frame) is not int or frame < 0:
            raise ValueError("nonnegative frame required")
        if self.last is not None and frame <= self.last:
            raise ValueError("strictly increasing frames required")
        if self.last is not None and frame != self.last + 1:
            self.streak = {}
        self.last = frame
        rows = {}
        for slot in map(str, range(8)):
            cue = cues.get(slot, {}).get("cue", "UNKNOWN")
            if (glyphs or {}).get(slot) == "fold":
                cue = "EXPLICIT_FOLD"
            prior, count = self.streak.get(slot, (None, 0))
            count = count + 1 if prior == cue else 1
            self.streak[slot] = cue, count
            confirmed = cue != "UNKNOWN" and count >= self.required
            status = "UNKNOWN"
            conflict = cues.get(slot, {}).get("conflict", False)
            if confirmed and not conflict:
                mapping = {"BACK_CARDS": "DEALT_IN_CANDIDATE",
                           "EXPLICIT_FOLD": "FOLDED_CANDIDATE",
                           "EMPTY": "EMPTY_CANDIDATE",
                           "WAITING_NEXT_HAND": "WAITING_CANDIDATE",
                           "WAITING_POST_OR_PASS": "WAITING_CANDIDATE"}
                status = mapping.get(cue, "UNKNOWN")
                if (self.history.get(slot) in (
                        "DEALT_IN_CANDIDATE", "FOLDED_CANDIDATE")
                        and status in ("EMPTY_CANDIDATE", "WAITING_CANDIDATE")):
                    conflict, status = True, "UNKNOWN"
                elif (self.history.get(slot) in (
                        "FOLDED_CANDIDATE", "WAITING_CANDIDATE", "EMPTY_CANDIDATE")
                        and status == "DEALT_IN_CANDIDATE"):
                    conflict, status = True, "UNKNOWN"
                elif self.hand is not None:
                    self.history[slot] = status
            rows[slot] = {"current": status,
                          "history": self.history.get(slot, "UNKNOWN"),
                          "conflict": conflict, "in_this_hand": None}
        return {"frame": frame, "hand": self.hand, "slots": rows,
                "boundary_authority": "external_not_inferred",
                "full_visual_acceptance": False, "strategy_eligible": False}


def run(first, target, profile_path, output):
    metadata = json.loads((first / "samples.json").read_text())
    audit = metadata["audit_sha256"]
    source_rows, target_rows = inventory(first, audit), inventory(target, audit)
    reader = ParticipationReader(json.loads(profile_path.read_text()),
                                 load(first, source_rows[1320]),
                                 load(first, source_rows[3319]),
                                 waiting_next=load(first, source_rows[2400]))
    predictions = [{"frame": frame, "slots": reader.recognize(load(target, row))}
                   for frame, row in target_rows.items()]
    counts = {cue: sum(row["cue"] == cue for item in predictions
                       for row in item["slots"].values())
              for cue in ("EMPTY", "BACK_CARDS", "WAITING_POST_OR_PASS",
                          "WAITING_NEXT_HAND", "UNKNOWN")}
    report = {"status": "DEVELOPMENT_CANDIDATES_NOT_ACCEPTANCE", "counts": counts,
              "frame_count": len(predictions), "source_frames": [1320, 2400, 3319],
              "source_sha256": {str(f): source_rows[f]["sha256"]
                                for f in (1320, 2400, 3319)},
              "layout_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
              "independent_holdout": False, "full_visual_acceptance": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.first, args.target, args.profile, args.output)
