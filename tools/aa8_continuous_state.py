"""Offline AA8 actor-ring/wager candidates with fail-closed temporal evidence.

This is a development probe, not a legal action or participation recognizer.
Unobserved amounts and absent rings remain UNKNOWN, never guessed from order.
"""

import argparse
from collections import deque
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_unmarked_money import pot_patch
from tools.aa_amount_candidate import stack_patch
from tools.aa_pot_candidate import colon_x, pot_mask, region
from tools.aa_wager_candidate import AAWagerCandidate


# Eight-seat canvas only; deliberately separate from the nine-slot reader.
WAGERS = ((210, 249, 79, 27), (335, 281, 85, 33),
          (335, 439, 85, 33), (335, 596, 85, 33),
          (285, 808, 85, 30), (79, 596, 85, 33),
          (79, 439, 85, 33), (79, 281, 85, 33))


def timer_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 170), (180, 95, 255))


def suffix_components(binary):
    """Isolate strokes so avatar pixels around the countdown cannot dominate."""
    # AA side-avatar antialiasing can bridge a suffix diagonally to a white
    # portrait background. Four-connectivity preserves the isolated strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    values = []
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
    return values


def actor_ring(image, profile, timer=None):
    """Unique bright green avatar perimeter is a candidate, not timer OCR."""
    scores = []
    for item in profile["slots"]:
        x, y, w, h = item["avatar"]
        hsv = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, (25, 80, 190), (80, 255, 255)) > 0
        ring = np.ones((h, w), bool)
        ring[8:-8, 8:-8] = False
        scores.append(float(bright[ring].mean()))
    candidates = [s for s, score in enumerate(scores) if score >= .07]
    timer_scores = [None] * 8
    if timer is not None:
        for s in candidates:
            x, y, w, h = profile["slots"][s]["avatar"]
            patch = timer_mask(image[y + 20:y + h - 15, x + 15:x + w - 9])
            glyphs = suffix_components(patch)
            scores_glyph = [float(2 * np.sum(glyph * timer) / (
                np.sum(glyph * glyph) + np.sum(timer * timer))) for glyph in glyphs]
            timer_scores[s] = max(scores_glyph, default=0.)
        candidates = [s for s in candidates if timer_scores[s] >= .85]
    return {"actor": candidates[0] if len(candidates) == 1 else None,
            "reason": "unique_bright_ring_candidate" if len(candidates) == 1
            else "ambiguous_or_missing_ring", "ring_scores": scores,
            "timer_suffix_scores": timer_scores,
            "timer_suffix_verified": timer is not None and len(candidates) == 1,
            "timer_text_verified": False}


def board_count(image):
    """Positive large pale card rectangles; does not recognize ranks or suits."""
    counts = []
    for x in (110, 166, 223, 279, 335):
        hsv = cv2.cvtColor(image[475:545, x:x + 49], cv2.COLOR_BGR2HSV)
        counts.append(float(((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 175)).mean()))
    visible = [v >= .45 for v in counts]
    n = sum(visible)
    valid = n in (0, 3, 4, 5) and visible == [True] * n + [False] * (5 - n)
    return n if valid else None


class CandidateReader:
    def __init__(self, profile, bank, reference, timer_reference=None):
        if profile.get("hero_slot") != 4 or len(profile.get("slots", [])) != 8:
            raise ValueError("eight-seat layout required")
        self.profile, self.bank = profile, bank
        self.timer = None
        if timer_reference is not None:
            crop = timer_mask(timer_reference[181:197, 254:265])
            components = suffix_components(crop)
            if len(components) != 1:
                raise ValueError("ambiguous timer suffix source")
            self.timer = components[0]
        self.wager = AAWagerCandidate.__new__(AAWagerCandidate)
        self.wager.bank = bank
        # Coin at the first-hand slot3 wager; source reviewed frame1500.
        self.wager.coin = cv2.cvtColor(reference[604:621, 392:409], cv2.COLOR_BGR2GRAY)
        mask = pot_mask(region(reference))
        colon = colon_x(mask)
        if colon is None or colon < 34:
            raise ValueError("missing reference pot prefix")
        self.prefix = mask[:, colon - 34:colon + 2]

    def read(self, image):
        if image is None or image.shape != (1080, 498, 3):
            raise ValueError("normalized AA8 canvas required")
        wagers = {}
        for s, (x, y, w, h) in enumerate(WAGERS):
            patch = image[y:y + h, x:x + w]
            # Legacy helper's right-alignment IDs are translated explicitly.
            value = self.wager.read_label(patch, 2 if s in (1, 2, 3) else 0)
            wagers[str(s)] = value
        actor = actor_ring(image, self.profile, self.timer)
        return {"actor": actor["actor"], "actor_evidence": actor,
                "board_count": board_count(image),
                "wagers": {s: v.get("value") for s, v in wagers.items()},
                "wager_diagnostics": wagers,
                "stacks": {str(s): self.bank.diagnose(stack_patch(
                    image, self.profile["slots"][s]["stack"])).value for s in range(8)},
                "pot": self.bank.diagnose(pot_patch(image, self.prefix)).value,
                "strategy_eligible": False}


class ContinuousEvidence:
    """No stale actor, no inferred check, no amount carry across board changes."""
    def __init__(self):
        self.previous = None
        self.context = None
        self.history = deque(maxlen=4)

    def observe(self, frame, observation):
        self.history.append({"frame": frame, **observation})
        result = self._observe(frame, observation)
        result["unmarked_action_candidate"] = short_all_in_candidate(list(self.history))
        return result

    def _observe(self, frame, observation):
        row = {"frame": frame, **observation}
        previous = self.previous
        self.previous = row
        result = {"frame": frame, "action": None, "status": "UNKNOWN",
                  "strategy_eligible": False, "context_automated": True}
        if previous is None or frame != previous["frame"] + 1:
            self.context = None
            return result
        if (row["board_count"] is None or
                row["board_count"] != previous["board_count"]):
            self.context = None
            result["reason"] = "board_change_or_unknown_reset"
            return result
        actor = row["actor"]
        if actor != previous["actor"]:
            self.context = None
        if actor is not None and actor == previous["actor"]:
            own = row["wagers"].get(str(actor))
            known = [v for v in row["wagers"].values() if v is not None]
            if own is not None and known and row["wagers"] == previous["wagers"]:
                try:
                    numbers = [Decimal(v) for v in known if isinstance(v, str)]
                    if len(numbers) != len(known) or any(
                            not v.is_finite() or v < 0 for v in numbers):
                        raise ValueError("invalid amount")
                except (InvalidOperation, ValueError):
                    self.context = None
                    return result
                self.context = {"actor": actor, "own": own,
                                "price_lower_bound": str(max(numbers)),
                                "frame": frame}
            else:
                self.context = None
        # This layer intentionally returns candidates, not inferred legal calls:
        # missing other wagers cannot establish the exact current street price.
        result["context"] = self.context if actor is not None else None
        if result["context"] is not None:
            result["status"] = "AUTOMATED_VISUAL_CONTEXT_CANDIDATE"
        if actor is None:
            self.context = None
        return result


def short_all_in_candidate(rows):
    """Only a four-frame cash-supported short-call candidate, not legal truth.

    A known wager greater than own wager plus all remaining cash proves a short
    amount even if another seat's wager is unknown. No exact-price claim needed.
    """
    unknown = {"action": None, "status": "UNKNOWN", "strategy_eligible": False}
    if len(rows) != 4:
        return unknown
    a, b, c, d = rows
    try:
        ids = [r["frame"] for r in rows]
        if ids != list(range(ids[0], ids[0] + 4)):
            return unknown
        if a["board_count"] not in (0, 3, 4, 5) or any(
                r["board_count"] != a["board_count"] for r in rows):
            return unknown
        actor = a["actor"]
        if actor not in range(8) or b["actor"] != actor:
            return unknown
        if not all(r["actor_evidence"].get("timer_suffix_verified") for r in (a, b)):
            return unknown
        if any(r["actor"] is not None for r in (c, d)):
            return unknown
        if a["wagers"] != b["wagers"]:
            return unknown

        def money(value):
            if not isinstance(value, str):
                raise ValueError("unknown money")
            amount = Decimal(value)
            if not amount.is_finite() or amount < 0:
                raise ValueError("invalid money")
            return amount

        stacks = [{s: money(v) for s, v in r["stacks"].items()} for r in rows]
        if any(set(s) != set(map(str, range(8))) for s in stacks):
            return unknown
        pots = [money(r["pot"]) for r in rows]
        if stacks[0] != stacks[1] or stacks[2] != stacks[3] or (
                pots[0] != pots[1] or pots[2] != pots[3]):
            return unknown
        slot = str(actor)
        changed = [s for s in stacks[1] if stacks[1][s] != stacks[2][s]]
        if changed != [slot] or stacks[2][slot] != 0 or stacks[1][slot] <= 0:
            return unknown
        debit = stacks[1][slot]
        if pots[2] - pots[1] != debit:
            return unknown
        own = money(b["wagers"][slot])
        others = [money(v) for s, v in b["wagers"].items()
                  if s != slot and v is not None]
        if not others or max(others) <= own + debit:
            return unknown
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return unknown
    return {"action": "call", "all_in": True, "actor": actor, "debit": str(debit),
            "status": "CASH_SUPPORTED_SHORT_CALL_CANDIDATE", "strategy_eligible": False,
            "context_automated": True, "visual_action_glyph_inferred": False,
            "legal_action_verified": False, "single_action_interval_verified": False,
            "exact_street_price_known": False, "price_lower_bound": str(max(others)),
            "evidence_frames": ids}


def run(source, target, bank_path, profile_path, output, start=None, end=None):
    manifest = json.loads((source / "samples.json").read_text())
    source_rows = inventory(source, manifest["audit_sha256"])
    target_rows = inventory(target, manifest["audit_sha256"])
    arrays = np.load(bank_path)
    bank = GrayAmountRecognizer(arrays["features"], arrays["labels"], augment=True)
    timer_reference = load(target, target_rows[4755])
    reader = CandidateReader(json.loads(profile_path.read_text()), bank,
                             load(source, source_rows[1500]), timer_reference)
    tracker = ContinuousEvidence()
    output.mkdir(parents=True, exist_ok=False)
    selected = [f for f in target_rows if (start is None or f >= start)
                and (end is None or f <= end)]
    if not selected:
        raise ValueError("empty development selection")
    contexts, actors, observations = [], {}, []
    with (output / "observations.jsonl").open("w") as stream:
        for frame in selected:
            observation = reader.read(load(target, target_rows[frame]))
            event = tracker.observe(frame, observation)
            row = {"frame": frame, "sha256": target_rows[frame]["sha256"],
                   **observation, "temporal": event}
            stream.write(json.dumps(row) + "\n")
            key = str(observation["actor"])
            actors[key] = actors.get(key, 0) + 1
            if event.get("context") is not None:
                contexts.append(frame)
            if frame in (4755, 4822, 4823, 4824, 4825):
                observations.append(row)
    report = {"frames": len(selected), "first": selected[0], "last": selected[-1],
              "actor_candidate_counts": actors, "context_frames": len(contexts),
              "checkpoints": observations,
              "source_manifest_sha256": sha(source / "samples.json"),
              "target_manifest_sha256": sha(target / "samples.json"),
              "bank_sha256": sha(bank_path), "layout_sha256": sha(profile_path),
              "timer_template_frame": target_rows[4755],
              "implementation_sha256": sha(Path(__file__)),
              "observations_sha256": sha(output / "observations.jsonl"),
              "independent_holdout": False, "full_visual_acceptance": False,
              "limitations": ["timer suffix is not numeric countdown OCR",
                              "wager OCR absence is UNKNOWN, not zero",
                              "street price is only a lower bound",
                              "no automatic hand identity or dealt-in roster",
                              "no complete action reconstruction"]}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoints"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "bank", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    run(args.source, args.target, args.bank, args.profile,
        args.output, args.start, args.end)
