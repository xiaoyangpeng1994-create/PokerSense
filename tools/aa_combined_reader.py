"""Offline AA current-frame composition. Missing fields never enable strategy."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa_action_candidate import AAActionCandidate
from tools.aa_amount_candidate import AAAmountCandidate
from tools.aa_card_visibility import face_card_support, locate_face_card
from tools.aa_visual_candidate import AASceneCandidate
from tools.aa_pot_candidate import AAPotCandidate
from tools.aa_seat_candidate import AASeatCandidate
from tools.aa_pot_evidence import AACenterAmountCandidate, reconcile_pot_displays
from tools.aa_hero_turn import hero_turn_candidate
from tools.aa_wager_candidate import AAWagerCandidate
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import pixels_digest, read_image


class CurrentFrameConsensus:
    def __init__(self):
        self.values = {}

    def clear(self):
        self.values.clear()

    def observe(self, key, value, frame):
        old = self.values.get(key)
        if value is None:
            self.values.pop(key, None)
            return None
        self.values[key] = (value, frame)
        return value if old == (value, frame - 1) else None


class AACombinedReader:
    def __init__(self, profile, reference, action_reference, bank, heads,
                 *, prefix_reference=None, center_bank=None):
        if len(profile["slots"]) != 9:
            raise ValueError("combined reader has not been validated for this layout")
        self.profile = profile
        self.scene = AASceneCandidate(reference, profile)
        self.amount = AAAmountCandidate(bank, profile, self.scene)
        self.action = AAActionCandidate(action_reference, profile, self.scene)
        self.pot = AAPotCandidate(reference, bank, self.scene,
                                  prefix_reference=prefix_reference)
        self.seats = AASeatCandidate(reference, profile, bank, self.scene)
        self.center_amount = AACenterAmountCandidate(
            bank if center_bank is None else center_bank, self.scene,
            reference=reference)
        self.wagers = AAWagerCandidate(bank if center_bank is None else center_bank,
                                       self.scene, reference)
        self.cards = FusedCardRecognizerAdapter(FusedCardRecognizer(
            heads, rank_floor=.5, suit_floor=.3))
        self.consensus = CurrentFrameConsensus()
        self.last_frame = None
        self.last_time = None
        self.context = None
        self.regions = {}

    def process(self, image, frame, milliseconds, source):
        if (type(frame) is not int or frame < 0
                or not np.isfinite(milliseconds) or milliseconds < 0):
            raise ValueError("valid source identity and time required")
        if (self.last_frame is not None and frame <= self.last_frame
                and source == self.context):
            raise ValueError("duplicate/backwards frame")
        gap = (self.last_frame is None or source != self.context
               or frame != self.last_frame + 1 or milliseconds <= self.last_time
               or milliseconds - self.last_time > 1000)
        if gap:
            self.consensus.clear()
            self.cards.reset()
            self.regions.clear()
        self.last_frame, self.last_time, self.context = frame, milliseconds, source
        scene = self.scene.recognize(image)
        result = {"source_frame": frame, "opencv_pos_msec": milliseconds,
                  "source": source, "gap_reset": gap, "scene": scene["scene"],
                  "hero": None, "board_slots": [None] * 5,
                  "stacks": {str(i): None for i in range(9)},
                  "actions": {str(i): None for i in range(9)},
                  "seat_presence": {str(i): None for i in range(9)},
                  "seat_presence_semantics": "physical_occupancy_not_hand_roster",
                  "pot": None, "hero_participation": "UNKNOWN",
                  "pot_title": None, "pot_center": None,
                  "visible_wagers": [None] * 9,
                  "pot_evidence": reconcile_pot_displays(None, None),
                  "current_actor": None,
                  "current_actor_semantics": "visual_slot_candidate_not_legal_actor",
                  "special_modes": {name: {"enabled": None, "triggered": None}
                                    for name in ("critical_hit", "mushroom", "squid",
                                                 "insurance")},
                  "strategy_eligible": False,
                  "incomplete_fields": ["seat_presence", "pot", "full_actions",
                                        "participation", "special_modes", "acceptance"]}
        if scene["scene"] != "AA_TABLE_CANDIDATE":
            self.consensus.clear()
            self.cards.reset()
            self.regions.clear()
            return result
        timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            milliseconds=milliseconds)
        self.cards.begin_frame(frame, timestamp, (source, self.profile["layout_id"]),
                               {"hero": (192, 936, 114, 83),
                                "board": (109, 452, 280, 121)})
        for group, xs, y in (("hero", (193, 250), 938),
                             ("board", (110, 166, 223, 279, 335), 473)):
            values = []
            for slot, x in enumerate(xs):
                key = (group, slot)
                rect = (x, y, 53, 78)
                if group == "board":
                    rect, _ = locate_face_card(image, rect)
                elif not face_card_support(image, rect)[0]:
                    rect = None
                if rect != self.regions.get(key):
                    self.cards.reset(key)
                self.regions[key] = rect
                if rect is None:
                    self.cards.reset(key)
                    values.append(None)
                    continue
                rx, ry, rw, rh = rect
                read = self.cards.recognize(image[ry:ry + rh, rx:rx + rw], key)
                values.append(str(read.value[0]) if read.value else None)
            if group == "hero":
                pair = tuple(values) if all(values) and values[0] != values[1] else None
                result["hero"] = self.consensus.observe("hero", pair, frame)
            else:
                result["board_slots"] = [self.consensus.observe(("board", i), v, frame)
                                         for i, v in enumerate(values)]
        amounts = self.amount.recognize(image)["stacks"]
        result["pot_title"] = self.consensus.observe(
            "pot_title", self.pot.recognize(image)["value"], frame)
        result["pot_center"] = self.consensus.observe(
            "pot_center", self.center_amount.recognize(image)["value"], frame)
        wager_reads = self.wagers.recognize(image)
        result["visible_wagers"] = [self.consensus.observe(
            ("wager", slot), r["value"], frame) for slot, r in enumerate(wager_reads)]
        result["pot_evidence"] = reconcile_pot_displays(
            result["pot_title"], result["pot_center"],
            visible_wagers=result["visible_wagers"])
        result["pot"] = result["pot_evidence"]["value"]
        if result["pot"] is not None:
            result["incomplete_fields"].remove("pot")
        actions = self.action.recognize(image)
        seats = self.seats.recognize(image)
        for slot in map(str, range(9)):
            result["seat_presence"][slot] = self.consensus.observe(
                ("presence", slot), seats[slot]["presence"], frame)
            result["stacks"][slot] = self.consensus.observe(
                ("stack", slot), amounts[slot]["value"], frame)
            result["actions"][slot] = self.consensus.observe(
                ("action", slot), actions[slot]["value"], frame)
        if all(v is not None for v in result["seat_presence"].values()):
            result["incomplete_fields"].remove("seat_presence")
        hero = str(self.profile["hero_slot"])
        if result["actions"][hero] == "fold":
            result["hero_participation"] = "VISIBLE_FOLD_CANDIDATE"
        turn = self.consensus.observe(
            "hero_turn", hero_turn_candidate(image)["hero_turn"], frame)
        if (turn is True and result["hero"] is not None
                and result["actions"][hero] != "fold"):
            result["current_actor"] = self.profile["hero_slot"]
        return result


def run(audit, exploration, bank_dir, profile_path, heads_path, output, end,
        *, prefix_reference_path=None, center_bank_dir=None):
    for folder in (audit, exploration, bank_dir):
        if verify_sha256sums(folder):
            raise ValueError("input manifest failure")
    source_report = json.loads((audit / "report.json").read_text())
    if type(end) is not int or not 0 <= end < source_report["decoded_frames"]:
        raise ValueError("bounded source range required")
    source = Path(source_report["source"])

    def signature():
        stat = source.stat()
        return stat.st_size, stat.st_mtime_ns
    expected = (source_report["source_size_bytes"], source_report["source_mtime_ns"])
    if signature() != expected:
        raise ValueError("original source changed")
    profile = json.loads(profile_path.read_text())
    bank_report = json.loads((bank_dir / "report.json").read_text())
    if (bank_report["source_sha256"] != source_report["source_sha256"]
            or bank_report["profile_sha256"] != sha256_file(profile_path)):
        raise ValueError("bank is not bound to this source and geometry")
    input_hashes = {str(path.resolve()): sha256_file(path)
                    for path in (heads_path, profile_path, bank_dir / "bank.npz")}
    if prefix_reference_path is not None:
        input_hashes[str(prefix_reference_path.resolve())] = sha256_file(
            prefix_reference_path)
    config = NormalizationConfig.from_json(
        (exploration / "normalization.candidate.json").read_text())
    with np.load(bank_dir / "bank.npz", allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    center_bank = None
    if center_bank_dir is not None:
        if verify_sha256sums(center_bank_dir):
            raise ValueError("centre bank integrity failure")
        meta = json.loads((center_bank_dir / "report.json").read_text())
        if (meta["source_sha256"] != source_report["source_sha256"] or
                meta["profile_sha256"] != sha256_file(profile_path)):
            raise ValueError("centre bank source/geometry mismatch")
        path = center_bank_dir / "bank.npz"
        input_hashes[str(path.resolve())] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            center_bank = GrayAmountRecognizer(
                data["features"], data["labels"], augment=True)
    reader = AACombinedReader(profile, read_image(exploration / "frame_000000.png"),
                              read_image(exploration / "frame_029700.png"),
                              bank, load_card_heads(heads_path), prefix_reference=(
                                  read_image(prefix_reference_path)
                                  if prefix_reference_path is not None else None),
                              center_bank=center_bank)
    anchors = {r["source_frame_index"]: r for r in source_report["samples"]}
    index = [json.loads(line) for line in
             (audit / "decode_index.jsonl").read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(source))
    start_time = time.perf_counter()
    hero_count = 0
    changes, previous = [], None
    try:
        with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for frame in range(end + 1):
                ok, raw = capture.read()
                if not ok:
                    raise ValueError("early decode termination")
                stamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if abs(stamp - index[frame]["opencv_pos_msec"]) > .001:
                    raise ValueError("source time differs from audit")
                if (frame in anchors and
                        pixels_digest(raw) != anchors[frame]["pixels_sha256"]):
                    raise ValueError("anchor differs from audit")
                result = reader.process(normalize(raw, config), frame, stamp,
                                        source_report["source_sha256"])
                stream.write(json.dumps(result) + "\n")
                hero_count += result["hero"] is not None
                if result["hero"] != previous:
                    changes.append({"frame": frame, "hero": result["hero"]})
                    previous = result["hero"]
                if frame % 300 == 0:
                    print(f"AA combined: {frame}/{end}", flush=True)
    finally:
        capture.release()
    if signature() != expected:
        raise ValueError("source changed during replay")
    if any(sha256_file(Path(p)) != h for p, h in input_hashes.items()):
        raise ValueError("model or geometry changed during replay")
    report = {"frames": end + 1, "hero_presented_frames": hero_count,
              "hero_presentation_changes": changes,
              "elapsed_seconds": time.perf_counter() - start_time,
              "source_sha256": source_report["source_sha256"],
              "heads_sha256": sha256_file(heads_path),
              "bank_sha256": sha256_file(bank_dir / "bank.npz"),
              "profile_sha256": sha256_file(profile_path),
              "input_hashes": input_hashes,
              "independent_holdout": False, "strategy_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "exploration", "bank", "profile", "heads", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--end", type=int, default=1800)
    parser.add_argument("--prefix-reference", type=Path)
    parser.add_argument("--center-bank", type=Path)
    args = parser.parse_args()
    run(args.audit, args.exploration, args.bank, args.profile, args.heads,
        args.output, args.end, prefix_reference_path=args.prefix_reference,
        center_bank_dir=args.center_bank)
