"""V2 development integration, leaving the independently tested V1 unchanged."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_holdout_predict import FrozenPredictionState
from tools.aa8_participation_v2 import ParticipationReaderV2
from tools.aa8_state_adapter_v2 import AA8StateAdapterV2
from tools.aa8_wagergeometry_v2 import AA8WagerReaderV2
from tools.aa8_live_wagers_v2 import CausalWagersV2
from tools.aa8_dealer_v2 import AA8DealerReader, StableDealerEvidence
from tools.aa8_strategy_state_v2 import AA8HandLedgerCandidate, candidate_action_line
from tools.aa8_base_scope import scope_receipt
from tools.aa_pot_evidence import AACenterAmountCandidate
from tools.aa8_continuous_state import actor_ring, board_count
from tools.aa_visual_candidate import canvas_ok
from tools.aa_bomb_candidate import AABombTitleCandidate
from tools.aa_data_separation import assert_training_frames
from tools.wpk_video_dataset import read_image


class BombGuard:
    def __init__(self, base, reference):
        self.base = base
        self.bomb = AABombTitleCandidate(reference)
        self.last = None
        self.result = None

    def recognize(self, image):
        result = self.base.recognize(image)
        current = hashlib.sha256(image[350:700, 10:488].tobytes()).hexdigest()
        if current != self.last:
            self.last, self.result = current, self.bomb.recognize(image)
        result["critical_hit_title"] = self.result.copy()
        if self.result["critical_hit_animation"] is True:
            result["block_state_updates"] = True
            result["blocking_overlay"] = "LUCKY_BOMB_TRANSITION"
        return result


def load_bomb(pool, reservations_path):
    manifest = json.loads((pool / "samples.json").read_text())
    reservations = json.loads(reservations_path.read_text())
    if manifest["source_sha256"] != reservations["source_sha256"]:
        raise ValueError("legacy bomb source mismatch")
    assert_training_frames([8640], reservations)
    sample = next(r for r in manifest["samples"] if r["source_frame"] == 8640)
    path = (pool / sample["file"]).resolve()
    if (sample["split"] != "development" or not path.is_relative_to(pool.resolve())
            or sha(path) != sample["sha256"]):
        raise ValueError("untrusted bomb development witness")
    return read_image(path), sample


class CandidateStateV2(FrozenPredictionState):
    def __init__(self, source, context_source, late_source, bank_path, profile_path,
                 heads_path, audit, bomb_reference):
        super().__init__(source, context_source, late_source, bank_path,
                         profile_path, heads_path, audit)
        source_rows = inventory(source, audit)
        self.seats = ParticipationReaderV2(
            self.profile, load(source, source_rows[1320]),
            load(source, source_rows[3319]),
            waiting_next=load(source, source_rows[2400]))
        self.modes = BombGuard(self.modes, bomb_reference)
        bank = self.context_reader.bank
        reference = load(source, source_rows[1500])
        self.center = AACenterAmountCandidate(bank, self.reader.scene,
                                              reference=reference)
        self.context_reader = ContextReaderV2(
            self.context_reader, self.reader.scene, self.modes,
            AA8WagerReaderV2(bank, reference, coin_smoothing=True))
        self.adapter = AA8StateAdapterV2()
        self.causal_wagers = CausalWagersV2()
        self.dealer_reader = AA8DealerReader()
        self.dealer_evidence = StableDealerEvidence()
        self.hand_ledger = AA8HandLedgerCandidate()

    def read(self, image, frame, sample):
        row = super().read(image, frame, sample)
        row["wager_visibility_v2"] = self.context_reader.last_visibility if (
            row["scene_supported"]) else {}
        previous = len(self.adapter.actions)
        row["observed_state_v2"] = self.adapter.observe(row)
        row["observed_actions_v2"] = self.adapter.actions[previous:]
        partition = {"title": row.get("pot", {}).get("value"),
                     "center": self.center.recognize(image),
                     "wagers": row.get("street_wagers") or {},
                     "visibility": row["wager_visibility_v2"]}
        row["observed_center_v2"] = partition["center"]
        row["causal_street_wagers_v2"] = self.causal_wagers.observe(
            row, self.adapter, partition)
        row["visual_scope"] = scope_receipt(row)
        detected = self.dealer_reader.read(image)
        modes = row.get("special_modes") or {}
        dealer = self.dealer_evidence.observe(
            frame, detected["dealer_seat"],
            epoch=row["observed_state_v2"].get("observed_epoch"),
            blocked=bool(modes.get("block_state_updates")
                         or modes.get("insurance") == "VISIBLE"),
        )
        row["dealer_observation_v2"] = detected
        row["dealer_evidence_v2"] = dealer
        row["dealer_seat"] = dealer["dealer_seat"]
        row["dealer_seat_canonical_verified"] = False
        ledger = self.hand_ledger.observe(row, self.adapter)
        row["hand_ledger_v2"] = ledger
        row["hand_commitments"] = ledger["hand_commitments"]
        row["hand_commitments_canonical_verified"] = False
        action_line = candidate_action_line(
            self.adapter.actions,
            row["observed_state_v2"].get("observed_epoch"),
            row["observed_state_v2"].get("street_candidate"),
        )
        row["action_line_v2"] = action_line
        row["action_line"] = action_line["value"]
        row["actions_complete_and_canonical_verified"] = False
        return row


class ContextReaderV2:
    """Avoid duplicate money OCR; parent supplies cached current stacks and pot."""
    def __init__(self, original, scene, modes, wagers):
        self.profile, self.timer = original.profile, original.timer
        self.scene, self.modes, self.wagers = scene, modes, wagers
        self.last_visibility = {}

    def read(self, image):
        supported = (canvas_ok(image) and self.scene.recognize(image)["scene"] == (
            "AA_TABLE_CANDIDATE") and not self.modes.recognize(image)[
                "block_state_updates"])
        if not supported:
            self.last_visibility = {}
            return {"actor": None, "actor_evidence": {}, "board_count": None,
                    "wagers": dict.fromkeys(map(str, range(8))),
                    "stacks": {}, "pot": None}
        observed = self.wagers.read(image, scene_supported=True, unobstructed=True)
        self.last_visibility = observed["visibility"]
        actor = actor_ring(image, self.profile, self.timer)
        return {"actor": actor["actor"], "actor_evidence": actor,
                "board_count": board_count(image), "wagers": observed["wagers"],
                "wager_diagnostics": observed["diagnostics"], "stacks": {}, "pot": None}


def create_candidate(spec):
    """Explicit frozen path configuration; no evaluation labels or dynamic imports."""
    reference, _ = load_bomb(Path(spec["bomb_pool"]), Path(spec["reservations"]))
    return CandidateStateV2(
        Path(spec["source"]), Path(spec["context_source"]), Path(spec["late_source"]),
        Path(spec["bank_path"]), Path(spec["profile_path"]), Path(spec["heads_path"]),
        spec["audit"], reference)


def run(source, target, late, bank, profile, heads, bomb_pool, reservations, output,
        preroll=None):
    audit = json.loads((source / "samples.json").read_text())["audit_sha256"]
    bomb_reference, bomb_provenance = load_bomb(bomb_pool, reservations)
    reader = CandidateStateV2(source, target, late, bank, profile, heads, audit,
                              bomb_reference)
    inputs = {str(p.resolve()): sha(p) for p in (
        Path(__file__), Path(__file__).with_name("aa8_state_adapter_v2.py"),
        Path(__file__).with_name("aa8_participation_v2.py"), bank, profile, heads,
        Path(__file__).with_name("aa8_wagergeometry_v2.py"),
        Path(__file__).with_name("aa8_live_wagers_v2.py"),
        Path(__file__).with_name("aa8_street_baseline_v2.py"),
        Path(__file__).with_name("aa8_base_scope.py"),
        Path(__file__).with_name("aa8_dealer_v2.py"),
        Path(__file__).with_name("aa8_strategy_state_v2.py"),
        bomb_pool / "samples.json", reservations)}
    output.mkdir(parents=True, exist_ok=False)
    total = 0
    preroll_frames = []
    if preroll is not None:
        rows = inventory(preroll, audit)
        first = min(inventory(source, audit))
        selected = [f for f in rows if first - 3 <= f < first]
        if selected != list(range(first - 3, first)):
            raise ValueError("requires three contiguous preceding context frames")
        manifest = preroll / "samples.json"
        inputs[str(manifest.resolve())] = sha(manifest)
        for frame in selected:
            reader.read(load(preroll, rows[frame]), frame, rows[frame])
            preroll_frames.append({"frame": frame, "sha256": rows[frame]["sha256"]})
    with (output / "observations.jsonl").open("x") as stream:
        for pool in (source, target):
            inputs[str((pool / "samples.json").resolve())] = sha(pool / "samples.json")
            for frame, row in inventory(pool, audit).items():
                result = reader.read(load(pool, row), frame, row)
                stream.write(json.dumps(result) + "\n")
                total += 1
                if total % 1000 == 0:
                    print(f"V2 observed {total} development frames", flush=True)
    if any(sha(Path(p)) != v for p, v in inputs.items()):
        raise ValueError("V2 inputs changed during run")
    report = {"frames": total, "input_hashes": inputs,
              "preroll_context_not_scored": preroll_frames,
              "legacy_bomb_training_source": bomb_provenance,
              "actions": reader.adapter.actions, "epochs": reader.adapter.epoch_events,
              "observations_sha256": sha(output / "observations.jsonl"),
              "independent_holdout": False, "full_visual_acceptance": False,
              "strategy_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames": total, "actions": len(reader.adapter.actions),
                      "amount_known": sum(r["amount"] is not None
                                          for r in reader.adapter.actions)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "target", "late", "bank", "profile", "heads", "bomb-pool",
                "reservations", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--preroll", type=Path)
    args = parser.parse_args()
    run(args.source, args.target, args.late, args.bank, args.profile, args.heads,
        args.bomb_pool, args.reservations, args.output, args.preroll)
