"""Automatic AA8 new-post/deal candidate, never an authoritative hand boundary."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.aa8_action_transfer import inventory, load
from tools.aa8_participation import dice, mask, white


class CenterDealCue:
    def __init__(self, source_image, floor=.90):
        patch = source_image[520:560, 235:263]
        self.red, self.white = mask(patch, "back"), white(patch)
        self.floor = floor
        if np.count_nonzero(self.red) < 100 or np.count_nonzero(self.white) < 20:
            raise ValueError("positive center deal-card template required")

    def recognize(self, image):
        if not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3):
            return {"visible": None, "reason": "unsupported_canvas"}
        patch = image[520:560, 235:263]
        score = min(dice(mask(patch, "back"), self.red), dice(white(patch), self.white))
        return {"visible": True if score >= self.floor else None,
                "score": score, "reason": "center_back_card_positive_only"}


def number(value):
    if isinstance(value, dict):
        value = value.get("value")
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    except InvalidOperation:
        return None


def snapshot(row):
    stacks = {slot: number(row.get("stacks", {}).get(slot))
              for slot in map(str, range(8))}
    if (row.get("scene_supported") is not True
            or type(row.get("board_count")) is not int
            or row.get("board_count") != 0
            or number(row.get("pot")) != 0 or any(v is None for v in stacks.values())):
        return None
    return stacks


class HandTransitionCandidates:
    def __init__(self, stable_frames=2, min_posting_seats=3):
        if type(stable_frames) is not int or stable_frames < 2:
            raise ValueError("at least two stable frames required")
        if type(min_posting_seats) is not int or not 3 <= min_posting_seats <= 8:
            raise ValueError("three or more simultaneous debits required")
        self.required, self.minimum = stable_frames, min_posting_seats
        self.last, self.value, self.count, self.pending = None, None, 0, None

    def observe(self, row, deal):
        frame = row.get("frame")
        if (type(frame) is not int or frame < 0
                or self.last is not None and frame <= self.last):
            raise ValueError("strictly increasing nonnegative frames required")
        if self.last is not None and frame != self.last + 1:
            self.value, self.count, self.pending = None, 0, None
        current = snapshot(row)
        visible = deal.get("visible") is True
        event = None
        if self.pending:
            pending = self.pending
            if current == pending["new"] and visible:
                pending["count"] += 1
                if pending["count"] >= self.required:
                    event = {key: value for key, value in pending.items()
                             if key not in ("new", "count")}
                    event.update(confirmed_at_frame=frame, status="NEW_HAND_CANDIDATE",
                                 authoritative_boundary=False, strategy_eligible=False)
                    self.pending = None
            else:
                self.pending = None
        if (current is not None and self.value is not None and current != self.value
                and self.count >= self.required and visible):
            debits = {slot: str(self.value[slot] - current[slot]) for slot in current
                      if current[slot] < self.value[slot]}
            credits = {slot: str(current[slot] - self.value[slot]) for slot in current
                       if current[slot] > self.value[slot]}
            if len(debits) >= self.minimum:
                self.pending = {
                    "first_candidate_frame": frame, "before_frame": self.last,
                    "posting_debits": debits, "unallocated_credits": credits,
                    "credit_semantics": "POSSIBLE_REFILL_NOT_PROFIT",
                    "new": current, "count": 1,
                    "evidence": "stable_zero_board_pot_multi_debit_center_deal"}
        unchanged = current is not None and current == self.value
        self.count = self.count + 1 if unchanged else 1
        self.value, self.last = current, frame
        if current is None:
            self.count = 0
        return {"frame": frame, "candidate": event, "center_deal": deal,
                "full_visual_acceptance": False, "strategy_eligible": False}


def run(observations, first, pool, output):
    source_meta = json.loads((first / "samples.json").read_text())
    audit = source_meta["audit_sha256"]
    first_rows, rows = inventory(first, audit), inventory(pool, audit)
    reader = CenterDealCue(load(first, first_rows[1263]))
    tracker, results = HandTransitionCandidates(), []
    for line in observations.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = rows[row["frame"]]
        if row.get("source_sha256") != source["sha256"]:
            raise ValueError("observation source hash mismatch")
        results.append(tracker.observe(row, reader.recognize(load(pool, source))))
    report = {"status": "DEVELOPMENT_CANDIDATES_NOT_BOUNDARY_ACCEPTANCE",
              "frame_count": len(results), "events": [r["candidate"] for r in results
                                                      if r["candidate"] is not None],
              "template_source_frame": 1263,
              "template_source_sha256": first_rows[1263]["sha256"],
              "observations_sha256": hashlib.sha256(
                  observations.read_bytes()).hexdigest(),
              "independent_holdout": False, "full_visual_acceptance": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(results), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.observations, args.first, args.pool, args.output)
