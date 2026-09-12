"""Convert saved AA8 observations into a validated no-advice shadow WAL."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

from poker_engine.core.value_objects import ChipAmount
import poker_engine.strategy.aa8_shadow as aa8_shadow
import poker_engine.strategy.shadow_log as shadow_log
from poker_engine.strategy.aa8_shadow import assess_aa8_shadow_input
from poker_engine.strategy.contracts import GameConfig, GameType
from poker_engine.strategy.shadow_log import (
    ShadowWalWriter,
    analyze_shadow_wal,
    summarize_aa8_observation,
)
from tools.aa8_action_transfer import inventory, sha


def run(observations, prediction_report, pools, player_count, data_root, session_id):
    if player_count not in (6, 7, 8):
        raise ValueError("player_count must be 6, 7 or 8")
    report = json.loads(prediction_report.read_text(encoding="utf-8"))
    if sha(observations) != report.get("observations_sha256"):
        raise ValueError("observation/report hash mismatch")
    sources, audit = {}, None
    manifest_hashes = {}
    for pool in pools:
        manifest = json.loads((pool / "samples.json").read_text(encoding="utf-8"))
        if audit is not None and manifest["audit_sha256"] != audit:
            raise ValueError("mixed recording audit identities")
        audit = manifest["audit_sha256"]
        rows = inventory(pool, audit)
        if set(sources) & set(rows):
            raise ValueError("overlapping source pools")
        sources.update({frame: (pool, row) for frame, row in rows.items()})
        manifest_hashes[str((pool / "samples.json").resolve())] = sha(
            pool / "samples.json"
        )
    structural = GameConfig(
        "NLHE", GameType.CASH, 8, player_count,
        ChipAmount("1"), ChipAmount("2"), ante=ChipAmount("0"),
        rake_percent=Decimal("0"), rake_cap=ChipAmount("0"),
        minimum_chip=ChipAmount("1"),
    )
    writer = ShadowWalWriter(data_root, session_id, {
        "source_observations": str(observations.resolve()),
        "source_observations_sha256": sha(observations),
        "prediction_report": str(prediction_report.resolve()),
        "prediction_report_sha256": sha(prediction_report),
        "source_manifest_sha256": manifest_hashes,
        "source_audit_sha256": audit,
        "player_count_parameter": player_count,
        "table_rules_authorized_for_strategy": False,
        "provider_or_equity_enabled": False,
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(Path(aa8_shadow.__file__).resolve()): sha(
                Path(aa8_shadow.__file__)
            ),
            str(Path(shadow_log.__file__).resolve()): sha(
                Path(shadow_log.__file__)
            ),
        },
    })
    previous = None
    count = 0
    try:
        with observations.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                frame = row["frame"]
                if frame not in sources:
                    raise ValueError("observation frame outside bound pools")
                pool, source = sources[frame]
                if row["source_sha256"] != source["sha256"]:
                    raise ValueError("observation/source hash mismatch")
                frame_path = (pool / source["file"]).resolve()
                if (not frame_path.is_relative_to(pool.resolve())
                        or sha(frame_path) != source["sha256"]):
                    raise ValueError("source frame path/hash mismatch")
                started = perf_counter()
                blockers = assess_aa8_shadow_input(row, structural)
                elapsed = (perf_counter() - started) * 1000.0
                status = "ABSTAIN" if blockers else "READY"
                if any(reason.startswith("deferred_special_mode:")
                       for reason in blockers):
                    status = "DEFERRED_SPECIAL_MODE"
                state = row.get("observed_state_v2") or {}
                change = {
                    "epoch_changed": previous is not None and state.get(
                        "observed_epoch") != previous.get("observed_epoch"),
                    "street_changed": previous is not None and state.get(
                        "street_candidate") != previous.get("street_candidate"),
                    "current_epoch": state.get("observed_epoch"),
                    "current_street": state.get("street_candidate"),
                }
                writer.append(
                    source_frame_seq=frame,
                    source_sha256=source["sha256"],
                    source_ref=str(frame_path),
                    pts_seconds=str(row.get("pts_seconds")) if row.get(
                        "pts_seconds") is not None else None,
                    vision=summarize_aa8_observation(row),
                    strategy_status=status,
                    blockers=blockers,
                    stage_timings_ms={"strategy_input_gate": elapsed},
                    state_change=change,
                )
                previous = state
                count += 1
        if count != report.get("frames"):
            raise ValueError("observation frame count does not match report")
        receipt = writer.close()
    except Exception:
        writer.abort()
        raise
    analysis = analyze_shadow_wal(writer.wal_path)
    analysis.update({
        "receipt_wal_sha256_match": receipt.wal_sha256 == analysis["wal_sha256"],
        "receipt_last_hash_match": (
            receipt.last_record_sha256 == analysis["last_record_sha256"]
        ),
        "strategy_eligible": False,
        "full_visual_acceptance": False,
    })
    (writer.session_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "session": session_id,
        "records": receipt.records,
        "statuses": analysis["statuses"],
        "optimization_flags": analysis["optimization_flags"],
        "wal_sha256": receipt.wal_sha256,
    }))
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--player-count", type=int, choices=(6, 7, 8), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    run(args.observations, args.prediction_report, args.pool, args.player_count,
        args.data_root, args.session_id)
