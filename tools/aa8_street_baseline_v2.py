"""Conditional eight-seat visible wager partition, not a canonical betting ledger."""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_continuous_state import CandidateReader
from tools.aa8_wager_visibility import AA8WagerVisibility
from tools.aa_pot_evidence import AACenterAmountCandidate
from tools.aa_visual_candidate import AASceneCandidate


class PartitionReader:
    def __init__(self, profile, bank, reference1500, timer4755, *, wager_reader=None):
        self.scene = AASceneCandidate(reference1500, profile)
        self.center = AACenterAmountCandidate(bank, self.scene, reference=reference1500)
        self.fields = CandidateReader(profile, bank, reference1500, timer4755)
        self.visibility = AA8WagerVisibility(reference1500)
        self.wager_reader = wager_reader

    def read(self, image, *, unobstructed=False):
        scene = self.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE"
        fields = self.fields.read(image)
        wagers = fields["wagers"]
        visibility = self.visibility.read(
            image, scene_supported=scene, unobstructed=unobstructed)
        if self.wager_reader is not None:
            extra = self.wager_reader.read(
                image, scene_supported=scene, unobstructed=unobstructed)
            wagers, visibility = extra["wagers"], extra["visibility"]
        return {"title": fields["pot"], "center": self.center.recognize(image),
                "wagers": wagers, "actor": fields["actor"],
                "actor_evidence": fields["actor_evidence"],
                "visibility": visibility,
                "scene_supported": scene, "unobstructed": unobstructed}


def partition_evidence(observation, context):
    result = {"status": "BASELINE_UNKNOWN", "wagers": None, "street_price": None,
              "canonical_verified": False, "strategy_eligible": False,
              "context_automated": context.get("automated") is True}
    if (observation.get("scene_supported") is not True
            or observation.get("unobstructed") is not True
            or context.get("epoch") is None
            or context.get("new_street_observed") is not True
            or context.get("no_earlier_actions") is not True
            or context.get("collection_or_modal") is not False
            or type(observation.get("actor")) is not int
            or not 0 <= observation["actor"] < 8):
        return {**result, "reason": "incomplete_causal_scene_actor_context"}
    evidence = observation.get("actor_evidence", {})
    if (evidence.get("timer_suffix_verified") is not True
            and evidence.get("hero_turn") is not True):
        return {**result, "reason": "no_fresh_actor_evidence"}

    def money(value):
        if not isinstance(value, str):
            raise ValueError("unknown amount")
        number = Decimal(value)
        if not number.is_finite() or number < 0:
            raise ValueError("invalid amount")
        return number

    wagers, supports = {}, {}
    try:
        title = money(observation.get("title"))
        center = money(observation.get("center", {}).get("value"))
        for s in map(str, range(8)):
            value = observation.get("wagers", {}).get(s)
            visible = observation.get("visibility", {}).get(s, {}).get("status")
            absent = context.get("not_applicable", {}).get(s) in ("empty", "waiting")
            if absent:
                if value is not None or visible != "VISIBLE_EMPTY_CANDIDATE":
                    return {**result, "reason": "absent_seat_has_conflicting_display"}
                wagers[s] = {"status": "NOT_APPLICABLE"}
                supports[s] = "explicit_absent_seat_and_empty_rectangle"
            elif value is not None and visible == "VISIBLE_COIN_CANDIDATE":
                wagers[s] = str(money(value))
                supports[s] = "current_coin_prefixed_amount"
            elif value is None and visible == "VISIBLE_EMPTY_CANDIDATE":
                wagers[s] = "0"
                supports[s] = "conditional_positive_empty_before_any_action"
            else:
                return {**result, "reason": "unknown_or_conflicting_wager_rectangle"}
        total = center + sum(money(v) for v in wagers.values() if isinstance(v, str))
    except (ValueError, InvalidOperation):
        return {**result, "reason": "unknown_or_invalid_money"}
    if total != title:
        return {**result, "reason": "title_center_wager_partition_disagrees",
                "center_plus_visible": str(total), "title": str(title)}
    return {**result, "status": "PARTITION_CONSISTENT_SINGLE_FRAME",
            "epoch": context["epoch"], "actor": observation["actor"],
            "wagers": wagers, "supports": supports,
            "street_price": str(max(money(v) for v in wagers.values()
                                    if isinstance(v, str))),
            "title": str(title), "center": str(center),
            "reason": "conditional_visual_partition_not_room_accounting_proof"}


class StreetBaselineCandidate:
    def __init__(self):
        self.previous = None

    def observe(self, frame, observation, context):
        if type(frame) is not int or frame < 0:
            raise ValueError("nonnegative source frame required")
        previous = self.previous
        current = partition_evidence(observation, context)
        self.previous = (frame, current)
        if previous and frame <= previous[0]:
            self.previous = None
            raise ValueError("duplicate/backwards frames")
        if current["status"] != "PARTITION_CONSISTENT_SINGLE_FRAME":
            return current
        if not previous or frame != previous[0] + 1 or current != previous[1]:
            return {"status": "BASELINE_UNKNOWN",
                    "reason": "two_stable_frames_required",
                    "wagers": None, "street_price": None, "canonical_verified": False,
                    "strategy_eligible": False}
        return {**current, "status": "CONDITIONAL_BASELINE_CANDIDATE",
                "evidence_frames": [previous[0], frame]}


def run(first, second, bank_path, profile_path, observations_path, output,
        wager_v2=False):
    metadata = json.loads((first / "samples.json").read_text())
    sources = inventory(first, metadata["audit_sha256"])
    context_rows = inventory(second, metadata["audit_sha256"])
    with np.load(bank_path, allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    override = None
    if wager_v2:
        from tools.aa8_wagergeometry_v2 import AA8WagerReaderV2
        override = AA8WagerReaderV2(bank, load(first, sources[1500]))
    reader = PartitionReader(
        json.loads(profile_path.read_text()), bank,
        load(first, sources[1500]), load(second, context_rows[4755]),
        wager_reader=override)
    rows = {r["frame"]: r for r in map(
        json.loads, observations_path.read_text().splitlines())}
    tracker, result = StreetBaselineCandidate(), []
    for frame in (1320, 1321, 1500, 1501, 2070, 2071, 2400, 2401):
        if rows[frame]["source_sha256"] != sources[frame]["sha256"]:
            raise ValueError("development observation hash mismatch")
        observation = reader.read(load(first, sources[frame]), unobstructed=True)
        # These gate annotations are diagnostics, explicitly NOT an automated history.
        early = frame in (1320, 1321, 2070, 2071)
        context = {"epoch": "preflop_probe" if frame < 2000 else "flop_probe",
                   "new_street_observed": early, "no_earlier_actions": early,
                   "collection_or_modal": False, "automated": False,
                   "not_applicable": {"6": "empty" if frame < 2000 else "waiting"}}
        result.append({"frame": frame, "sha256": sources[frame]["sha256"],
                       "read": observation, "context": context,
                       "baseline": tracker.observe(frame, observation, context)})
    report = {"observations": result, "bank_sha256": sha(bank_path),
              "implementation_sha256": sha(Path(__file__)),
              "development_manifest_sha256": sha(first / "samples.json"),
              "context_manifest_sha256": sha(second / "samples.json"),
              "layout_sha256": sha(profile_path), "context_automated": False,
              "wager_v2_opt_in": wager_v2,
              "wager_geometry_sha256": sha(Path(__file__).with_name(
                  "aa8_wagergeometry_v2.py")) if wager_v2 else None,
              "full_visual_acceptance": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{
        "frame": r["frame"], "title": r["read"]["title"],
        "center": r["read"]["center"], "actor": r["read"]["actor"],
        "baseline": r["baseline"]} for r in result]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "bank", "profile", "observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wager-v2", action="store_true")
    args = parser.parse_args()
    run(args.first, args.second, args.bank, args.profile,
        args.observations, args.output, args.wager_v2)
