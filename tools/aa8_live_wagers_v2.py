"""Causal development wager ledger candidates; never live game advice."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_state_adapter_v2 import AA8StateAdapterV2, amount, current_cue
from tools.aa8_street_baseline_v2 import PartitionReader, StreetBaselineCandidate
from tools.aa8_wagergeometry_v2 import AA8WagerReaderV2


class CausalWagersV2:
    """Consume adapter AFTER its current frame, without caller-supplied truth flags."""
    def __init__(self):
        self.last = None
        self.key = None
        self.start = None
        self.initializer = StreetBaselineCandidate()
        self.baseline = None
        self.totals = None
        self.used = set()
        self.previous_signature = None
        self.unallocated_credits = []
        self.seen_credits = set()
        self.conflict_start = None

    def _invalidate(self):
        self.baseline = self.totals = None
        self.used.clear()
        self.initializer = StreetBaselineCandidate()
        self.previous_signature = None
        self.conflict_start = None

    def observe(self, row, adapter, partition):
        frame = row["frame"]
        if adapter.last != frame:
            raise ValueError("state adapter must consume this current frame first")
        if self.last is not None and frame <= self.last:
            raise ValueError("strictly increasing frames required")
        gap = self.last is not None and frame != self.last + 1
        self.last = frame
        observed = adapter.snapshot(frame)
        stage = observed["street_candidate"]
        geometry = observed.get("positive_board_geometry")
        # Buffer a first positive geometry frame, but two stable frames are still
        # mandatory before the initializer can publish anything. No rank backfill.
        if geometry and geometry["count"] in (3, 4, 5):
            proposed = {3: "flop", 4: "turn", 5: "river"}[geometry["count"]]
            order = {None: 0, "preflop": 0, "flop": 3, "turn": 4, "river": 5}
            historical_stage = getattr(adapter, "street", stage)
            if geometry["count"] >= max(order[stage], order[historical_stage]):
                stage = proposed
        key = (observed["observed_epoch"], stage)
        if key != self.key:
            self._invalidate()
            self.key, self.start = key, frame
        credit_rows = list(getattr(adapter, "credits", []))
        for epoch in adapter.epoch_events:
            credits = (epoch.get("posting_comparison") or {}).get("credits", {})
            for s, value in credits.items():
                first = epoch.get("first_frame", epoch["frame"])
                credit_rows.append({"first_frame": first,
                                    "confirmed_frame": epoch["frame"], "seat": int(s),
                                    "amount": value, "source": "new_post_comparison"})
        for credit in credit_rows:
            s = str(credit["seat"])
            identity = (credit["first_frame"], s, credit["amount"])
            if identity not in self.seen_credits:
                self.seen_credits.add(identity)
                self.unallocated_credits.append({
                    **credit, "frame": credit["confirmed_frame"], "slot": s,
                    "semantics": "UNALLOCATED_NOT_PROFIT"})
        result = {"frame": frame, "status": "WAGERS_UNKNOWN", "wagers": None,
                  "street_price": None, "context_automated": True,
                  "canonical_verified": False, "strategy_eligible": False,
                  "unallocated_credits": list(self.unallocated_credits)}
        modes = row.get("special_modes", {})
        blocked = (not row.get("scene_supported") or modes.get("block_state_updates")
                   or modes.get("insurance") == "VISIBLE")
        if gap or blocked or stage is None or key[0] is None:
            self._invalidate()
            return {**result, "reason": "gap_modal_or_no_positive_street_context"}
        epochs = [e for e in adapter.epoch_events if e["epoch"] == key[0]]
        anchored = bool(epochs and epochs[-1]["status"] == "MULTI_POST_DEAL_CANDIDATE")
        if stage == "preflop" and not anchored:
            return {**result, "reason": "initial_deal_context_unanchored"}
        actions = [a for a in adapter.actions if a.get("epoch") == key[0]
                   and a.get("street") == stage]
        pending = [a for a in adapter.pending if a.get("epoch") == key[0]
                   and a.get("street") == stage]
        cash_changes = [a for a in adapter.cash if a.get("street") == stage]
        context = {"epoch": ":".join(key), "new_street_observed": True,
                   "no_earlier_actions": not actions and not pending
                   and not cash_changes,
                   "collection_or_modal": False, "automated": True,
                   "not_applicable": {str(s): "empty" if current_cue(row, s) == (
                       "EMPTY_CANDIDATE") else "waiting" for s in range(8)
                       if current_cue(row, s) in (
                           "EMPTY_CANDIDATE", "WAITING_CANDIDATE")}}
        current = {**partition, "actor": row.get("current_actor"),
                   "actor_evidence": row.get("actor_evidence") or {},
                   "scene_supported": row.get("scene_supported") is True,
                   "unobstructed": not blocked}
        if self.baseline is None:
            baseline = self.initializer.observe(frame, current, context)
            if baseline["status"] != "CONDITIONAL_BASELINE_CANDIDATE":
                return {**result, "reason": baseline.get("reason"),
                        "baseline_diagnostic": baseline}
            self.baseline = baseline
            self.totals = dict(baseline["wagers"])
        center = amount(current.get("center"))
        if center is None:
            # A missing reading is not positive evidence of chip collection.
            # Keep only private candidate history; never publish the old ledger
            # until fresh visible center/title/wagers agree again. Gap, modal,
            # street change and actual center changes still invalidate it.
            self.previous_signature = None
            self.conflict_start = None
            return {**result, "reason": "center_unreadable_current_frame"}
        if center != amount(self.baseline["center"]):
            self._invalidate()
            return {**result, "reason": "collection_or_center_change_invalidates_epoch"}
        for action in actions:
            identity = (action["frame"], action["slot"], action["kind"])
            if identity in self.used:
                continue
            debit = amount(action.get("amount"))
            s = str(action["slot"])
            old = amount(self.totals.get(s))
            if debit is None or old is None:
                self._invalidate()
                return {**result, "reason": "unresolved_action_or_na_actor"}
            # Absolute UI labels never add chips; only each observed debit adds once.
            self.totals[s] = str(old + debit)
            self.used.add(identity)
        expected = center + sum(amount(v) for v in self.totals.values()
                                if isinstance(v, str))
        title = amount(current.get("title"))
        contradictions = [s for s, v in current["wagers"].items()
                          if v is not None and amount(v) != amount(self.totals.get(s))]
        unknown_regions = [s for s, v in current["wagers"].items() if v is None
                           and current["visibility"].get(s, {}).get("status") != (
                               "VISIBLE_EMPTY_CANDIDATE")]
        signature = (tuple(current["wagers"].items()), title, center)
        stable = signature == self.previous_signature
        self.previous_signature = signature
        if contradictions or unknown_regions or title != expected:
            # UI cash can settle before the two-frame glyph confirmation. During
            # this bounded wait expose UNKNOWN, never the retained internal totals.
            positive_conflict = bool(contradictions) or (
                title is not None and title != expected)
            if not positive_conflict:
                self.conflict_start = None
            elif self.conflict_start is None:
                self.conflict_start = frame
            if (positive_conflict and stable and not pending
                    and frame - self.conflict_start > 12):
                self._invalidate()
            return {**result,
                    "reason": "display_ledger_conflict_or_pending_confirmation",
                    "conflicting_slots": contradictions,
                    "unknown_regions": unknown_regions}
        self.conflict_start = None
        if pending:
            return {**result, "reason": "pending_action_not_yet_accounted"}
        return {**result, "status": "OBSERVED_STREET_WAGERS_CANDIDATE",
                "wagers": dict(self.totals), "street_price": str(max(
                    amount(v) for v in self.totals.values() if isinstance(v, str))),
                "baseline_evidence_frames": self.baseline["evidence_frames"],
                "observed_debits_applied": len(self.used),
                "title_center_ledger_reconciled": True}


def run(first, second, bank_path, profile_path, observations, output):
    audit = json.loads((first / "samples.json").read_text())["audit_sha256"]
    first_rows, second_rows = inventory(first, audit), inventory(second, audit)
    with np.load(bank_path, allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    reference = load(first, first_rows[1500])
    partition = PartitionReader(json.loads(profile_path.read_text()), bank, reference,
                                load(second, second_rows[4755]),
                                wager_reader=AA8WagerReaderV2(
                                    bank, reference, coin_smoothing=True))
    adapter, wagers = AA8StateAdapterV2(), CausalWagersV2()
    counts, baselines, changes = Counter(), {}, []
    output.mkdir(parents=True, exist_ok=False)
    with (output / "wagers.jsonl").open("x") as stream:
        for line in observations.read_text().splitlines():
            row = json.loads(line)
            frame = row["frame"]
            pool, rows = ((first, first_rows) if frame in first_rows
                          else (second, second_rows))
            if row["source_sha256"] != rows[frame]["sha256"]:
                raise ValueError("development observation source mismatch")
            adapter.observe(row)
            image = load(pool, rows[frame])
            # Reuse prepared title/actor observations rather than OCR all stacks again.
            visible = partition.wager_reader.read(
                image, scene_supported=row["scene_supported"],
                unobstructed=row["scene_supported"])
            values = {"title": row["pot"]["value"],
                      "center": partition.center.recognize(image), **visible}
            value = wagers.observe(row, adapter, values)
            counts[value["status"]] += 1
            if value["status"] == "OBSERVED_STREET_WAGERS_CANDIDATE":
                key = tuple(value["baseline_evidence_frames"])
                baselines.setdefault(str(key), value)
                if not changes or changes[-1]["wagers"] != value["wagers"]:
                    changes.append(value)
            stream.write(json.dumps({
                **value, "source_sha256": rows[frame]["sha256"]}) + "\n")
            if frame % 1000 == 0:
                print(f"Observed wager evidence through {frame}", flush=True)
    report = {"counts": dict(counts), "baselines": baselines, "ledger_changes": changes,
              "observed_actions": adapter.actions,
              "coin_smoothing_opt_in": True,
              "unallocated_credits": wagers.unallocated_credits,
              "inputs": {str(p): sha(p) for p in (
                  bank_path, profile_path, observations)},
              "implementation_sha256": sha(Path(__file__)),
              "context_automated": True, "canonical_verified": False,
              "full_visual_acceptance": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"counts": dict(counts), "baseline_count": len(baselines),
                      "ledger_changes": len(changes)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "bank", "profile", "observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.first, args.second, args.bank, args.profile,
        args.observations, args.output)
