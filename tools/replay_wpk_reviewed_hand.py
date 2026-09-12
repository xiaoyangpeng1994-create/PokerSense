"""Source-bind a reviewed WPK betting ledger; does not claim OCR integration."""

import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
import shutil

from poker_engine.core.serialization import deserialize, serialize
from poker_engine.state_engine.reviewed_replay import (
    audit_observed_settlement, replay_reviewed_hand, terminal_call_projection,
    verify_reviewed_checkpoints,
)
from poker_engine.state_engine.reviewed_completion import complete_reviewed_hand
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.verify_wpk_hand_trace import DEFAULT_TRACE, verify_source_binding
from tools.wpk_video_dataset import pixels_digest, read_image


def json_value(value):
    if isinstance(value, (Decimal, Fraction)):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value)}")


def verify_completion_sources(review, window, cash_window):
    summary = json.loads((window / "summary.json").read_text(encoding="utf-8"))
    cash = json.loads((cash_window / "summary.json").read_text(encoding="utf-8"))
    if summary["source_sha256"] != cash["source_sha256"]:
        raise ValueError("cash window belongs to another video")
    for item in review["evidence"].values():
        if item["window"] not in ("hand", "cash"):
            raise ValueError("unsupported evidence window")
        folder = window if item["window"] == "hand" else cash_window
        path = folder / "frames" / f'{item["source_frame"]:06d}.png'
        if sha256_file(path) != item["image_sha256"]:
            raise ValueError("completion image changed")
        if pixels_digest(read_image(path)) != item["pixel_sha256"]:
            raise ValueError("completion pixel identity changed")
    return len(review["evidence"])


def run(trace_path: Path, window: Path, output: Path,
        completion_path: Path | None = None, cash_window: Path | None = None):
    if output.exists():
        raise ValueError("preserve prior reviewed ledger outputs")
    trace_hash = sha256_file(trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    bound = verify_source_binding(trace, window)
    completed, completion_review, completion_hash = None, None, None
    completion_bound = 0
    if completion_path is not None:
        if cash_window is None:
            raise ValueError("completion requires a cash evidence window")
        completion_hash = sha256_file(completion_path)
        completion_review = json.loads(completion_path.read_text(encoding="utf-8"))
        completion_bound = verify_completion_sources(
            completion_review, window, cash_window)
        completed = complete_reviewed_hand(trace, completion_review)
        steps = completed.betting_steps
    else:
        steps = replay_reviewed_hand(trace)
    records, projections = [], []
    for step in steps:
        state_data = serialize(step.state)
        if deserialize(type(step.state), state_data) != step.state:
            raise ValueError("core state serialization changed state")
        record = {"kind": step.kind, "known_by_source_frame": step.known_by_frame,
                  "evidence": step.evidence, "source_kind": step.source_kind,
                  "action_order": step.action_order, "state": state_data,
                  "action_event": serialize(step.action_event)
                  if step.action_event else None,
                  "returned": [{"seat": seat, "amount": str(amount)}
                               for seat, amount in step.returned]}
        records.append(record)
        projection = terminal_call_projection(step.state)
        if projection:
            projections.append({"after_action_order": step.action_order,
                                "known_by_frame": step.known_by_frame,
                                "conditional_call_not_advice": True,
                                **asdict(projection)})
    settlement = (audit_observed_settlement(trace, steps[-1].state)
                  if completed is None else None)
    completion_records = []
    if completed is not None:
        for step in completed.completion_steps:
            state_data = serialize(step.state)
            if deserialize(type(step.state), state_data) != step.state:
                raise ValueError("completion state serialization changed state")
            completion_records.append({
                "kind": step.kind, "state": state_data,
                "known_by_source_frame": step.known_by_frame,
                "evidence": step.evidence, "source_kind": step.source_kind,
                "revealed_holes": {str(seat): [str(c) for c in cards]
                                   for seat, cards in step.revealed_holes},
                "gross_awards": dict(step.gross_awards),
                "visible_balances": dict(step.visible_balances),
                "cash_observed_seats": step.cash_observed_seats,
                "unallocated_outflows": dict(step.unallocated_outflows),
                "insurance_notice": step.insurance_notice,
            })
        settlement = {
            "status": "PARTIAL_UNALLOCATED_DIFFERENCE",
            "gross_awards": dict(completed.gross_awards),
            "unallocated_outflows": dict(completed.unallocated_outflows),
            "all_cash_balances_observed": completed.all_cash_balances_observed,
            "pokerkit_hand_ended": completed.pokerkit_hand_ended,
            "fee_components_verified": completed.fee_components_verified,
        }
    if sha256_file(trace_path) != trace_hash:
        raise ValueError("review changed during replay")
    if completion_path is not None and sha256_file(completion_path) != completion_hash:
        raise ValueError("completion review changed during replay")
    report = {"hand_id": trace["hand_id"], "source_evidence_frames_verified": bound,
              "review_sha256": trace_hash, "pokerkit_version": version("pokerkit"),
              "core_action_events_checked": sum(
                  s.action_event is not None for s in steps),
              "state_count": len(steps) + len(completion_records),
              "betting_state_count": len(steps), "betting_ledger_passed": True,
              "completion_evidence_refs_verified": completion_bound,
              "completion_review_sha256": completion_hash,
              "money_checkpoints_verified": verify_reviewed_checkpoints(trace, steps),
              "money_checkpoint_semantics": "reviewed logical stage targets",
              "same_frame_displayed_money_parity_verified": False,
              "core_state_serialization_roundtrip": True,
              "capture_to_state_verified": False, "strategy_golden_eligible": False,
              "exact_action_timing_verified": False,
              "river_showdown_award_replayed": completed is not None,
              "event_timestamps": "logical replay order only, not actual action times",
              "steps": records, "completion_steps": completion_records,
              "terminal_call_projections": projections,
              "settlement": settlement}
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_value),
        encoding="utf-8")
    shutil.copy2(trace_path, output / "reviewed-input.json")
    if completion_path is not None:
        shutil.copy2(completion_path, output / "completion-input.json")
    repo = Path(__file__).resolve().parents[1]
    files = [Path(__file__),
             repo / "src/poker_engine/state_engine/reviewed_replay.py",
             repo / "src/poker_engine/state_engine/reviewed_completion.py",
             repo / "src/poker_engine/equity/evaluator.py",
             repo / "src/poker_engine/state_engine/action_reconstruction.py",
             repo / "src/poker_engine/strategy/state.py"]
    hashes = {}
    for path in files:
        target = output / "source-snapshot" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[path.relative_to(repo).as_posix()] = sha256_file(path)
    (output / "source-hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return {"state_count": report["state_count"],
            "source_evidence_frames_verified": bound,
            "core_actions": report["core_action_events_checked"],
            "projections": projections, "settlement": settlement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-review", type=Path)
    parser.add_argument("--cash-window", type=Path)
    args = parser.parse_args()
    result = run(args.trace, args.window, args.output,
                 args.completion_review, args.cash_window)
    print(json.dumps(result, indent=2, default=json_value))
