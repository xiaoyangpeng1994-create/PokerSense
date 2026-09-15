"""Append-only, source-bound human observations; no media or training execution."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading


FIELDS = ("seat_presence", "current_bet", "pot", "actor", "action", "special_mode")
DECLARATIONS = {"seat_presence", "pot", "full_actions", "participation",
                "special_modes", "acceptance", "current_bet", "current_actor"}
ALIASES = {"actor": "current_actor", "action": "full_actions",
           "special_mode": "special_modes"}
SEATED = {"seat_presence", "current_bet", "action"}
ZONES = {"development", "reserved", "exploration", "unknown"}
ENTRY_KEYS = {"schema_version", "session_id", "frame", "seat", "field",
              "model_output", "model_reason", "human_value", "source_media_sha256",
              "replay_sha256", "recorded_at_utc", "frame_zone", "training_eligible",
              "exposure", "event", "policy_sha256", "sequence", "previous_sha256",
              "entry_sha256"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def known(value):
    if value is None or value == "UNKNOWN":
        return False
    if isinstance(value, dict):
        return bool(value) and all(known(v) for v in value.values())
    if isinstance(value, list):
        return all(known(v) for v in value)
    return True


def declarations(row):
    missing = row.get("incomplete_fields")
    if (not isinstance(missing, list) or any(type(k) is not str for k in missing)
            or len(set(missing)) != len(missing) or set(missing) - DECLARATIONS):
        raise ValueError("INCOMPLETE_FIELDS_CONTRACT_MISMATCH")
    # The allowlist checks the vocabulary; it never supplies missing fields.
    return missing


def model_value(row, field, seat):
    name = ALIASES.get(field, field)
    value = row.get(name)
    if field in SEATED:
        value = value.get(str(seat)) if isinstance(value, dict) else None
    if field == "action":
        value = row.get("actions", {}).get(str(seat))
    return value


def model_reason(row, field):
    missing = declarations(row)
    name = ALIASES.get(field, field)
    return ("incomplete_fields:" + name if name in missing else
            "source_output_or_unknown;no_numeric_confidence_recorded")


def equal_value(field, model, human):
    if field in {"pot", "current_bet"}:
        if type(model) is int and model >= 0:
            return model == int(human)
        if isinstance(model, str) and re.fullmatch(r"[0-9]+", model):
            return int(model) == int(human)
    return type(model) is type(human) and model == human


def validate_value(field, seat, value):
    if field not in FIELDS:
        raise ValueError("INVALID_FIELD")
    if field in SEATED:
        if type(seat) is not int or not 0 <= seat < 9:
            raise ValueError("INVALID_SEAT")
    elif seat is not None:
        raise ValueError("GLOBAL_FIELD_REQUIRES_NULL_SEAT")
    valid = False
    if field in {"pot", "current_bet"}:
        valid = isinstance(value, str) and bool(re.fullmatch(r"[0-9]{1,12}", value))
    elif field == "seat_presence":
        valid = isinstance(value, str) and value in {
            "participating", "folded", "waiting", "spectating", "empty", "all_in"}
    elif field == "actor":
        valid = type(value) is int and 0 <= value < 9
    elif field == "action":
        valid = isinstance(value, str) and value in {
            "fold", "muck", "all_in", "check", "call", "bet", "raise", "none"}
    elif field == "special_mode":
        valid = (isinstance(value, dict)
                 and set(value) == {"critical_hit", "squid", "insurance"}
                 and all(isinstance(v, dict) and set(v) == {"enabled", "triggered"}
                         and all(type(b) is bool for b in v.values())
                         for v in value.values()))
    if not valid:
        raise ValueError("INVALID_HUMAN_VALUE")


class Ledger:
    def __init__(self, path, rows, replay_sha, policy=None):
        self.path = Path(path)
        self.rows = rows
        self.replay_sha = replay_sha
        self.lock = threading.RLock()
        self.policy = policy or {"session_id": "replay-" + replay_sha[:16],
                                 "replay_sha256": replay_sha, "zones": []}
        self._validate_policy()
        self.policy_sha = digest(self.policy)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._read()

    def _validate_policy(self):
        p = self.policy
        if (set(p) != {"session_id", "replay_sha256", "zones"}
                or p["replay_sha256"] != self.replay_sha
                or not isinstance(p["session_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", p["session_id"])
                or not isinstance(p["zones"], list)):
            raise ValueError("INVALID_ZONE_POLICY")
        end = -1
        for zone in p["zones"]:
            if (set(zone) != {"first", "last", "zone"}
                    or type(zone["first"]) is not int or type(zone["last"]) is not int
                    or not end < zone["first"] <= zone["last"] < len(self.rows)
                    or zone["zone"] not in ZONES):
                raise ValueError("INVALID_ZONE_INTERVAL")
            end = zone["last"]

    def zone(self, frame):
        return next((z["zone"] for z in self.policy["zones"]
                     if z["first"] <= frame <= z["last"]), "unknown")

    @contextmanager
    def _locked(self):
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self.lock, lock_path.open("a+b") as f:
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                f.seek(0)
                if os.name == "nt":
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read(self):
        raw = self.path.read_bytes() if self.path.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise ValueError("LEDGER_TRUNCATED")
        entries, previous, trained = [], None, set()
        for line in raw.splitlines():
            e = json.loads(line)
            if not isinstance(e, dict) or set(e) != ENTRY_KEYS:
                raise ValueError("LEDGER_SCHEMA_MISMATCH")
            if (type(e["schema_version"]) is not int
                    or type(e["sequence"]) is not int
                    or not isinstance(e["recorded_at_utc"], str)
                    or not e["recorded_at_utc"].endswith("+00:00")):
                raise ValueError("LEDGER_SCHEMA_MISMATCH")
            datetime.fromisoformat(e["recorded_at_utc"])
            sha = e.pop("entry_sha256")
            if digest(e) != sha or e["previous_sha256"] != previous:
                raise ValueError("LEDGER_HASH_CHAIN_MISMATCH")
            e["entry_sha256"] = sha
            frame = e["frame"]
            if type(frame) is not int or not 0 <= frame < len(self.rows):
                raise ValueError("LEDGER_FRAME_MISMATCH")
            row = self.rows[frame]
            validate_value(e["field"], e["seat"], e["human_value"])
            if (e["schema_version"] != 1 or e["sequence"] != len(entries) + 1
                    or e["policy_sha256"] != self.policy_sha
                    or e["session_id"] != self.policy["session_id"]
                    or e["replay_sha256"] != self.replay_sha
                    or e["source_media_sha256"] != row["source"]
                    or e["model_output"] != model_value(row, e["field"], e["seat"])
                    or e["model_reason"] != model_reason(row, e["field"])
                    or e["frame_zone"] != self.zone(frame)
                    or e["training_eligible"] is not (self.zone(frame) == "development")
                    or e["exposure"] not in {"never_trained", "trained"}
                    or e["event"] not in {"confirmation", "training_exposure"}):
                raise ValueError("LEDGER_SOURCE_OR_POLICY_MISMATCH")
            key = (e["source_media_sha256"], frame)
            if key in trained and e["exposure"] != "trained":
                raise ValueError("EXPOSURE_CANNOT_REVERT")
            if e["event"] == "training_exposure" and (
                    not e["training_eligible"] or e["exposure"] != "trained"):
                raise ValueError("INVALID_TRAINING_EXPOSURE")
            if e["exposure"] == "trained":
                trained.add(key)
            entries.append(e)
            previous = sha
        return entries, hashlib.sha256(raw).hexdigest()

    def _append(self, entries, frame, field, seat, value, event):
        row = self.rows[frame]
        exposed = event == "training_exposure" or any(
            e["frame"] == frame and e["exposure"] == "trained" for e in entries)
        e = {"schema_version": 1, "session_id": self.policy["session_id"],
             "frame": frame, "seat": seat, "field": field,
             "model_output": model_value(row, field, seat),
             "model_reason": model_reason(row, field), "human_value": value,
             "source_media_sha256": row["source"], "replay_sha256": self.replay_sha,
             "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
             "frame_zone": self.zone(frame),
             "training_eligible": self.zone(frame) == "development",
             "exposure": "trained" if exposed else "never_trained",
             "event": event, "policy_sha256": self.policy_sha,
             "sequence": len(entries) + 1,
             "previous_sha256": entries[-1]["entry_sha256"] if entries else None}
        e["entry_sha256"] = digest(e)
        with self.path.open("ab") as stream:
            stream.write(encoded(e) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return e

    def confirm(self, frame, field, seat, value):
        if type(frame) is not int or not 0 <= frame < len(self.rows):
            raise ValueError("FRAME_OUT_OF_RANGE")
        validate_value(field, seat, value)
        with self._locked():
            entries, _ = self._read()
            return self._append(entries, frame, field, seat, value, "confirmation")

    def snapshot(self):
        with self._locked():
            entries, sha = self._read()
        latest = {}
        for e in entries:
            if e["event"] != "confirmation":
                continue
            latest[(e["frame"], e["field"], e["seat"])] = e
        return entries, list(latest.values()), sha

    def mark_trained(self, entry_sha):
        with self._locked():
            entries, _ = self._read()
            e = next((e for e in entries if e["entry_sha256"] == entry_sha), None)
            if e is None or not e["training_eligible"]:
                raise ValueError("TRAINING_ENTRY_NOT_ELIGIBLE")
            return self._append(entries, e["frame"], e["field"], e["seat"],
                                e["human_value"], "training_exposure")

    def report(self):
        entries, latest, sha = self.snapshot()
        groups = {}
        for field in FIELDS:
            selected = [e for e in latest if e["field"] == field]
            comparable = [e for e in selected if known(e["model_output"])]
            matches = sum(equal_value(field, e["model_output"], e["human_value"])
                          for e in comparable)
            groups[field] = {"confirmed_denominator": len(selected),
                             "comparable_denominator": len(comparable),
                             "consistent": matches,
                             "inconsistent": len(comparable) - matches,
                             "rejected_or_unknown": len(selected) - len(comparable)}
        report = {"schema_version": 1, "ledger_sha256": sha,
                  "confirmation_events": sum(e["event"] == "confirmation"
                                             for e in entries),
                  "latest_confirmations": len(latest), "fields": groups,
                  "scope": "confirmed_fields_only;not_independent_accuracy",
                  "false_positive_rate": None, "miss_rate": None,
                  "rate_reason": "complete_opportunity_truth_not_available",
                  "by_frame_seat_field": latest, "hand": "UNKNOWN",
                  "trained_frames": sorted({e["frame"] for e in entries
                                            if e["exposure"] == "trained"})}
        return {**report, "report_sha256": digest(report)}

    def export(self, kind):
        if kind not in {"all", "money", "glyph"}:
            raise ValueError("INVALID_EXPORT_KIND")
        history, latest, sha = self.snapshot()
        trained = sorted({e["frame"] for e in history if e["exposure"] == "trained"})
        entries = [e for e in latest if e["training_eligible"]]
        if kind == "money":
            entries = [e for e in entries if e["field"] in {"pot", "current_bet"}]
        if kind == "glyph":
            entries = [e for e in entries if e["field"] == "action"
                       and e["human_value"] in {"fold", "muck", "all_in"}]
        # Labels match the existing labelled_glyphs(value, patch) and
        # GlyphSupplementV3 references[label] = (image, slot) inputs.
        inputs = []
        for e in entries:
            item = {"frame": e["frame"],
                    "source_media_sha256": e["source_media_sha256"],
                    "confirmation_sha256": e["entry_sha256"], "field": e["field"],
                    "slot": e["seat"]}
            if e["field"] in {"pot", "current_bet"}:
                item.update(value=e["human_value"],
                            consumer="aa8_money_bank_v2.labelled_glyphs")
            elif e["field"] == "action":
                item.update(label=e["human_value"],
                            consumer="GlyphSupplementV3.references")
            else:
                item.update(value=e["human_value"], consumer="dataset_only")
            inputs.append(item)
        result = {"schema_version": 1, "kind": kind, "ledger_sha256": sha,
                  "entries": entries, "inputs": inputs, "independent_holdout": False,
                  "trained_frames": trained,
                  "media_included": False, "training_executed": False}
        return {**result, "export_sha256": digest(result)}
