"""Tamper-evident append-only telemetry for offline shadow strategy work."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STATUSES = {"ABSTAIN", "WAITING", "READY", "DEFERRED_SPECIAL_MODE"}
_SHADOW_RESULT_FIELDS = {
    "status", "method", "gross_expected_chips", "configured_net_expected_chips",
    "rake", "confidence_low", "confidence_high", "numerical_confidence",
    "rule_fingerprint", "asset_binding_id", "rejection_reasons",
}
_FLAG_PRIORITY = {
    "dealer_evidence_missing": "P0",
    "hand_commitment_ledger_missing": "P0",
    "complete_action_line_missing": "P0",
    "actor_visible_but_strategy_blocked": "P0",
    "street_wager_unknown": "P1",
    "stack_vector_incomplete": "P1",
    "hero_cards_unknown": "P1",
    "street_unknown": "P1",
    "deferred_special_mode_not_base_tuning": "DEFERRED",
    "dealer_candidate_unknown": "P0",
    "dealer_candidate_requires_independent_validation": "P0",
    "hand_ledger_unknown": "P0",
    "hand_ledger_requires_reconciliation_and_validation": "P0",
    "action_line_candidate_unknown": "P0",
    "preflop_action_line_candidate_unknown": "P0",
    "action_line_requires_completeness_validation": "P0",
    "postflop_strategy_provider_not_released": "P0",
    "action_line_waiting_for_street": "P1",
}
_LATENCY_BUCKETS_MS = (
    .01, .02, .05, .1, .25, .5, 1., 2., 5., 10., 25., 50., 100., 250.,
    500., 1000., float("inf"),
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def summarize_aa8_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep decision-relevant evidence without copying image pixels or names."""
    state = row.get("observed_state_v2") or {}
    cards = row.get("cards") or {}
    stacks = row.get("stacks") or {}
    wagers = row.get("causal_street_wagers_v2") or {}
    scope = row.get("visual_scope") or {}
    actor = row.get("actor_evidence") or {}
    dealer = row.get("dealer_evidence_v2") or {}
    ledger = row.get("hand_ledger_v2") or {}
    action_line = row.get("action_line_v2") or {}
    known_stacks = sum(
        isinstance(value, Mapping) and value.get("value") is not None
        for value in stacks.values()
    )
    participants = state.get("participants") or {}
    current_cues = (row.get("participation") or {}).get("slots") or {}
    resolved_stacks = 0
    for seat in map(str, range(8)):
        value = stacks.get(seat) or {}
        history = (participants.get(seat) or {}).get("state")
        cue = (current_cues.get(seat) or {}).get("current")
        if (value.get("value") is not None or history in ("empty", "waiting")
                or cue in ("EMPTY_CANDIDATE", "WAITING_CANDIDATE")):
            resolved_stacks += 1
    return {
        "scene_supported": row.get("scene_supported") is True,
        "current_actor": row.get("current_actor"),
        "actor_evidence": {
            key: actor.get(key) for key in (
                "reason", "timer_suffix_verified", "hero_turn", "control_layout",
            ) if key in actor
        },
        "hero_cards": cards.get("hero"),
        "board_cards": state.get("board_candidate"),
        "street": state.get("street_candidate"),
        "observed_epoch": state.get("observed_epoch"),
        "pot": (row.get("pot") or {}).get("value"),
        "known_stack_count": known_stacks,
        "known_or_not_applicable_stack_count": resolved_stacks,
        "stack_vector_complete": resolved_stacks == 8,
        "wager_status": wagers.get("status"),
        "wager_reason": wagers.get("reason"),
        "wagers": wagers.get("wagers"),
        "new_action_count": len(row.get("observed_actions_v2") or ()),
        "pending_action_count": state.get("pending_actions"),
        "unallocated_positive_cash_count": len(
            state.get("unallocated_positive_cash") or ()
        ),
        "scope_status": scope.get("status"),
        "deferred_modes": scope.get("observed_deferred_modes") or [],
        "authoritative_hand_boundary": (
            state.get("authoritative_hand_boundary") is True
        ),
        "complete_legal_state": state.get("complete_legal_state") is True,
        "dealer_seat_candidate": dealer.get("dealer_seat"),
        "dealer_candidate_reason": dealer.get("reason"),
        "dealer_evidence_frames": dealer.get("evidence_frames") or [],
        "hand_ledger_status": ledger.get("status"),
        "hand_commitments_candidate": ledger.get("hand_commitments"),
        "hand_ledger_unallocated_difference": ledger.get(
            "unallocated_difference"
        ),
        "action_line_candidate": action_line.get("value"),
        "action_line_candidate_reason": action_line.get("reason"),
    }


def optimization_flags(
    vision: Mapping[str, Any], status: str, blockers: tuple[str, ...],
) -> tuple[str, ...]:
    flags = []
    if status == "DEFERRED_SPECIAL_MODE":
        return ("deferred_special_mode_not_base_tuning",)
    elif status == "ABSTAIN" and type(vision.get("current_actor")) is int:
        flags.append("actor_visible_but_strategy_blocked")
    if vision.get("wager_status") == "WAGERS_UNKNOWN":
        flags.append("street_wager_unknown")
    if vision.get("stack_vector_complete") is not True:
        flags.append("stack_vector_incomplete")
    if vision.get("hero_cards") is None:
        flags.append("hero_cards_unknown")
    if vision.get("street") not in ("preflop", "flop", "turn", "river"):
        flags.append("street_unknown")
    if "dealer_not_canonical" in blockers:
        flags.append(
            "dealer_candidate_requires_independent_validation"
            if type(vision.get("dealer_seat_candidate")) is int
            else "dealer_candidate_unknown"
        )
    if "hand_commitments_not_canonical" in blockers:
        flags.append(
            "hand_ledger_requires_reconciliation_and_validation"
            if isinstance(vision.get("hand_commitments_candidate"), Mapping)
            else "hand_ledger_unknown"
        )
    if "actions_not_complete_or_canonical" in blockers:
        street = vision.get("street")
        if street == "preflop":
            flags.append(
                "action_line_requires_completeness_validation"
                if isinstance(vision.get("action_line_candidate"), str)
                else "preflop_action_line_candidate_unknown"
            )
        elif street in ("flop", "turn", "river"):
            flags.append("postflop_strategy_provider_not_released")
        else:
            flags.append("action_line_waiting_for_street")
    return tuple(dict.fromkeys(flags))


@dataclass(frozen=True)
class ShadowWalReceipt:
    session_id: str
    records: int
    last_record_sha256: str | None
    wal_sha256: str


class ShadowWalWriter:
    """Create one new session. Any write failure permanently closes the writer."""

    def __init__(
        self,
        data_root: Path,
        session_id: str,
        manifest: Mapping[str, Any],
        *,
        fsync_every: int = 1,
    ) -> None:
        if not isinstance(data_root, Path):
            raise TypeError("data_root must be Path")
        if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
            raise ValueError("invalid shadow session_id")
        if not isinstance(fsync_every, int) or isinstance(fsync_every, bool):
            raise TypeError("fsync_every must be an int")
        if fsync_every <= 0:
            raise ValueError("fsync_every must be > 0")
        root = data_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.session_dir = (root / session_id).resolve()
        if self.session_dir.parent != root:
            raise ValueError("session escaped data root")
        self.session_dir.mkdir(exist_ok=False)
        self.wal_path = self.session_dir / "wal.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"
        self.receipt_path = self.session_dir / "receipt.json"
        created = datetime.now(timezone.utc).isoformat()
        document = {
            **dict(manifest),
            "schema_version": 1,
            "session_id": session_id,
            "created_at_utc": created,
            "mode": "offline_shadow_no_advice",
            "image_pixels_stored": False,
        }
        self.manifest_path.write_bytes(_canonical(document) + b"\n")
        self._stream = self.wal_path.open("xb", buffering=0)
        self._session_id = session_id
        self._fsync_every = fsync_every
        self._seq = 0
        self._last_frame = None
        self._last_hash = None
        self._closed = False
        self._failed = False

    def append(
        self,
        *,
        source_frame_seq: int,
        source_sha256: str,
        source_ref: str,
        pts_seconds: str | None,
        vision: Mapping[str, Any],
        strategy_status: str,
        blockers: tuple[str, ...],
        stage_timings_ms: Mapping[str, float],
        state_change: Mapping[str, Any] | None = None,
        provider_executed: bool = False,
        equity_executed: bool = False,
        shadow_result: Mapping[str, Any] | None = None,
    ) -> str:
        if self._closed or self._failed:
            raise RuntimeError("shadow WAL is closed or failed")
        if (type(source_frame_seq) is not int or source_frame_seq < 0
                or self._last_frame is not None
                and source_frame_seq <= self._last_frame):
            self._failed = True
            raise ValueError("source frames must be strictly increasing")
        if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
            self._failed = True
            raise ValueError("source_sha256 must be lowercase SHA-256")
        if not isinstance(source_ref, str) or not source_ref:
            self._failed = True
            raise ValueError("source_ref must be non-empty")
        if strategy_status not in _STATUSES:
            self._failed = True
            raise ValueError("invalid strategy status")
        if type(provider_executed) is not bool or type(equity_executed) is not bool:
            self._failed = True
            raise TypeError("execution flags must be bool")
        shadow = dict(shadow_result or {})
        if set(shadow) - _SHADOW_RESULT_FIELDS:
            self._failed = True
            raise ValueError("shadow result contains actionable or unknown fields")
        blockers = tuple(blockers)
        if not all(isinstance(reason, str) and reason for reason in blockers):
            self._failed = True
            raise TypeError("blockers must contain non-empty strings")
        timings = dict(stage_timings_ms)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 for value in timings.values()
        ):
            self._failed = True
            raise ValueError("stage timings must be finite and non-negative")
        self._seq += 1
        trace_id = hashlib.sha256(
            f"{self._session_id}:{self._seq}:{source_sha256}".encode("utf-8")
        ).hexdigest()[:32]
        body = {
            "schema_version": 1,
            "seq": self._seq,
            "trace_id": trace_id,
            "source_frame_seq": source_frame_seq,
            "source_sha256": source_sha256,
            "source_ref": source_ref,
            "pts_seconds": pts_seconds,
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "vision": dict(vision),
            "state_change": dict(state_change or {}),
            "strategy": {
                "status": strategy_status,
                "blockers": list(blockers),
                "provider_executed": provider_executed,
                "equity_executed": equity_executed,
                "advice_emitted": False,
                "shadow_result": shadow,
            },
            "optimization_flags": list(
                optimization_flags(vision, strategy_status, blockers)
            ),
            "stage_timings_ms": timings,
            "previous_record_sha256": self._last_hash,
        }
        record = {**body, "record_sha256": _digest(body)}
        try:
            self._stream.write(_canonical(record) + b"\n")
            if self._seq % self._fsync_every == 0:
                os.fsync(self._stream.fileno())
        except Exception:
            self._failed = True
            raise
        self._last_frame = source_frame_seq
        self._last_hash = record["record_sha256"]
        return trace_id

    def close(self) -> ShadowWalReceipt:
        if self._closed:
            raise RuntimeError("shadow WAL already closed")
        if self._failed:
            self._stream.close()
            self._closed = True
            raise RuntimeError("failed shadow WAL cannot produce receipt")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True
        wal_sha = hashlib.sha256(self.wal_path.read_bytes()).hexdigest()
        receipt = ShadowWalReceipt(
            self._session_id, self._seq, self._last_hash, wal_sha
        )
        payload = {
            "schema_version": 1,
            "session_id": receipt.session_id,
            "records": receipt.records,
            "last_record_sha256": receipt.last_record_sha256,
            "wal_sha256": receipt.wal_sha256,
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with self.receipt_path.open("xb") as stream:
            stream.write(_canonical(payload) + b"\n")
        return receipt

    def abort(self) -> None:
        """Close a failed/incomplete session without writing a success receipt."""
        if not self._closed:
            self._stream.close()
            self._closed = True
        self._failed = True


def _validated_record(
    value: Any, expected_seq: int, previous_frame: int | None,
    previous_hash: str | None,
) -> tuple[dict[str, Any], str, int]:
    if not isinstance(value, dict):
        raise ValueError("shadow WAL record must be an object")
    row = dict(value)
    if row.get("seq") != expected_seq:
        raise ValueError("non-contiguous WAL sequence")
    frame = row.get("source_frame_seq")
    if (type(frame) is not int or previous_frame is not None
            and frame <= previous_frame):
        raise ValueError("non-monotonic source frame")
    claimed = row.pop("record_sha256", None)
    if (claimed != _digest(row)
            or row.get("previous_record_sha256") != previous_hash):
        raise ValueError("shadow WAL hash chain mismatch")
    strategy = row.get("strategy") or {}
    if (type(strategy.get("provider_executed")) is not bool
            or type(strategy.get("equity_executed")) is not bool
            or strategy.get("advice_emitted") is not False):
        raise ValueError("shadow WAL execution flags or advice state invalid")
    shadow = strategy.get("shadow_result") or {}
    if not isinstance(shadow, dict) or set(shadow) - _SHADOW_RESULT_FIELDS:
        raise ValueError("shadow WAL result contains actionable or unknown fields")
    for stage, elapsed in (row.get("stage_timings_ms") or {}).items():
        if (not isinstance(stage, str) or not stage
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool) or not math.isfinite(elapsed)
                or elapsed < 0):
            raise ValueError("invalid WAL stage timing")
    return row, claimed, frame


class _LatencyHistogram:
    def __init__(self) -> None:
        self.counts = [0] * len(_LATENCY_BUCKETS_MS)
        self.total = 0
        self.maximum = 0.0

    def observe(self, value: float) -> None:
        for index, ceiling in enumerate(_LATENCY_BUCKETS_MS):
            if value <= ceiling:
                self.counts[index] += 1
                break
        self.total += 1
        self.maximum = max(self.maximum, value)

    def percentile(self, ratio: float) -> float | None:
        if not self.total:
            return None
        target = math.ceil(self.total * ratio)
        cumulative = 0
        for count, ceiling in zip(self.counts, _LATENCY_BUCKETS_MS):
            cumulative += count
            if cumulative >= target:
                return self.maximum if math.isinf(ceiling) else ceiling
        raise AssertionError("histogram count mismatch")

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.total,
            "p50_upper_bound_ms": self.percentile(.50),
            "p95_upper_bound_ms": self.percentile(.95),
            "p99_upper_bound_ms": self.percentile(.99),
            "max_ms": self.maximum if self.total else None,
        }


class ShadowWalFollower:
    """Incrementally validate a growing WAL with bounded in-memory metrics."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be Path")
        self.path = path
        self._offset = 0
        self._buffer = b""
        self._seq = 0
        self._last_frame = None
        self._last_hash = None
        self._failed = False
        self._statuses = Counter()
        self._provider_executions = 0
        self._equity_executions = 0
        self._shadow_statuses = Counter()
        self._blockers = Counter()
        self._flags = Counter()
        self._examples: dict[str, list[dict[str, Any]]] = {}
        self._timings: dict[str, _LatencyHistogram] = {}

    def poll(self) -> int:
        if self._failed:
            raise RuntimeError("shadow WAL follower has failed")
        try:
            with self.path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size < self._offset:
                    raise ValueError("shadow WAL was truncated")
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
            self._buffer += chunk
            lines = self._buffer.split(b"\n")
            self._buffer = lines.pop()
            added = 0
            for line in lines:
                if not line:
                    continue
                row, claimed, frame = _validated_record(
                    json.loads(line), self._seq + 1, self._last_frame,
                    self._last_hash,
                )
                strategy = row["strategy"]
                self._increment(self._statuses, strategy["status"])
                self._provider_executions += int(strategy["provider_executed"])
                self._equity_executions += int(strategy["equity_executed"])
                shadow = strategy.get("shadow_result") or {}
                if shadow.get("status") is not None:
                    self._increment(self._shadow_statuses, str(shadow["status"]))
                for blocker in strategy["blockers"]:
                    self._increment(self._blockers, blocker)
                for flag in row.get("optimization_flags") or ():
                    flag = self._increment(self._flags, flag)
                    examples = self._examples.setdefault(flag, [])
                    if len(examples) < 5:
                        examples.append({
                            "source_frame_seq": frame,
                            "source_sha256": row["source_sha256"],
                            "source_ref": row["source_ref"],
                            "trace_id": row["trace_id"],
                        })
                for stage, value in row.get("stage_timings_ms", {}).items():
                    if stage not in self._timings and len(self._timings) >= 32:
                        stage = "__other__"
                    self._timings.setdefault(stage, _LatencyHistogram()).observe(
                        float(value)
                    )
                self._seq += 1
                self._last_frame = frame
                self._last_hash = claimed
                added += 1
            return added
        except Exception:
            self._failed = True
            raise

    @staticmethod
    def _increment(counter: Counter, key: str, limit: int = 128) -> str:
        if key not in counter and len(counter) >= limit:
            key = "__other__"
        counter[key] += 1
        return key

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": self._seq,
            "last_source_frame_seq": self._last_frame,
            "last_record_sha256": self._last_hash,
            "statuses": dict(self._statuses),
            "provider_executions": self._provider_executions,
            "equity_executions": self._equity_executions,
            "shadow_result_statuses": dict(self._shadow_statuses),
            "blockers": dict(self._blockers),
            "optimization_flags": dict(self._flags),
            "optimization_examples": dict(self._examples),
            "stage_timing_histograms": {
                name: histogram.snapshot()
                for name, histogram in sorted(self._timings.items())
            },
            "partial_line_bytes": len(self._buffer),
            "failed": self._failed,
            "advice_emitted": False,
        }


def analyze_shadow_wal(path: Path) -> dict[str, Any]:
    """Validate sequence/hash chain and rank evidence-backed optimization flags."""
    previous_hash = None
    previous_frame = None
    statuses = Counter()
    blockers = Counter()
    flags = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, list[float]] = {}
    active: dict[str, dict[str, Any]] = {}
    intervals: dict[str, list[dict[str, Any]]] = {}
    records = 0
    provider_executions = 0
    equity_executions = 0
    shadow_statuses = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row, claimed, frame = _validated_record(
                json.loads(line), records + 1, previous_frame, previous_hash,
            )
            strategy = row["strategy"]
            statuses[strategy.get("status")] += 1
            provider_executions += int(strategy["provider_executed"])
            equity_executions += int(strategy["equity_executed"])
            shadow = strategy.get("shadow_result") or {}
            if shadow.get("status") is not None:
                shadow_statuses[str(shadow["status"])] += 1
            blockers.update(strategy.get("blockers") or ())
            current_flags = set(row.get("optimization_flags") or ())
            for stage, elapsed in (row.get("stage_timings_ms") or {}).items():
                timings.setdefault(stage, []).append(float(elapsed))
            for flag in tuple(active):
                if flag not in current_flags:
                    intervals.setdefault(flag, []).append(active.pop(flag))
            for flag in current_flags:
                flags[flag] += 1
                bucket = examples.setdefault(flag, [])
                if len(bucket) < 5:
                    bucket.append({
                        "source_frame_seq": frame,
                        "source_sha256": row["source_sha256"],
                        "source_ref": row["source_ref"],
                        "trace_id": row["trace_id"],
                    })
                run = active.get(flag)
                if run is not None and run["last_frame"] == frame - 1:
                    run["last_frame"] = frame
                    run["frames"] += 1
                else:
                    if run is not None:
                        intervals.setdefault(flag, []).append(run)
                    active[flag] = {
                        "first_frame": frame,
                        "last_frame": frame,
                        "frames": 1,
                        "first_source_ref": row["source_ref"],
                    }
            records += 1
            previous_hash = claimed
            previous_frame = frame
    for flag, run in active.items():
        intervals.setdefault(flag, []).append(run)

    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = max(0, math.ceil(len(ordered) * ratio) - 1)
        return ordered[index]

    timing_summary = {
        stage: {
            "count": len(values),
            "p50_ms": percentile(values, .50),
            "p95_ms": percentile(values, .95),
            "p99_ms": percentile(values, .99),
            "max_ms": max(values),
        }
        for stage, values in sorted(timings.items())
    }
    queue = [
        {
            "priority": _FLAG_PRIORITY.get(flag, "P2"),
            "flag": flag,
            "frames": count,
            "examples": examples.get(flag, []),
            "longest_intervals": sorted(
                intervals.get(flag, []),
                key=lambda value: (-value["frames"], value["first_frame"]),
            )[:5],
        }
        for flag, count in flags.items()
    ]
    order = {"P0": 0, "P1": 1, "P2": 2, "DEFERRED": 3}
    queue.sort(key=lambda value: (
        order[value["priority"]], -value["frames"], value["flag"]
    ))
    return {
        "schema_version": 1,
        "records": records,
        "statuses": dict(statuses),
        "provider_executions": provider_executions,
        "equity_executions": equity_executions,
        "shadow_result_statuses": dict(shadow_statuses),
        "blockers": dict(blockers),
        "optimization_flags": dict(flags),
        "optimization_examples": examples,
        "optimization_queue": queue,
        "stage_timing_summary": timing_summary,
        "last_record_sha256": previous_hash,
        "wal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "hash_chain_valid": True,
        "advice_emitted": False,
    }


__all__ = [
    "ShadowWalReceipt",
    "ShadowWalFollower",
    "ShadowWalWriter",
    "analyze_shadow_wal",
    "optimization_flags",
    "summarize_aa8_observation",
]
