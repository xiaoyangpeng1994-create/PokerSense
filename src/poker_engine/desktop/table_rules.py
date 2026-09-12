"""Explicit WPK rule intake: incomplete rules never become strategy defaults."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import hashlib
import os
from pathlib import Path
import tempfile

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import GameConfig, GameType
from .settings import _settings_path


@dataclass(frozen=True)
class TableRulesResult:
    game_config: GameConfig | None
    unavailable_reason: str | None
    pending_fields: tuple[str, ...] = ()


def _amount(data: dict, key: str) -> Decimal | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an exact decimal string or null")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {key}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return number


def load_table_rules(path: Path) -> TableRulesResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_table_rules(data)


def validate_table_rules(data: dict) -> TableRulesResult:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported table rules schema")
    if data.get("mode", "simulation") not in ("simulation", "live"):
        raise ValueError("mode must be simulation or live")
    count = data.get("table_size", 8)
    if isinstance(count, bool) or not isinstance(count, int) or count not in (6, 7, 8):
        raise ValueError("table_size must be 6, 7 or 8")
    effects = data.get("extra_effects", "")
    if not isinstance(effects, str) or len(effects) > 500:
        raise ValueError("extra_effects must be at most 500 characters")
    amounts = {key: _amount(data, key) for key in (
        "small_blind", "big_blind", "rake_percent", "rake_cap_bb", "ante",
        "minimum_chip", "straddle_amount",
    )}
    mode = data.get("straddle_mode", "unconfirmed")
    if mode not in ("none", "mandatory", "optional", "unconfirmed"):
        raise ValueError("invalid straddle_mode")
    for key in ("small_blind", "big_blind", "minimum_chip"):
        if amounts[key] is not None and amounts[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if amounts["rake_percent"] is not None and amounts["rake_percent"] > 1:
        raise ValueError("rake_percent must be a fraction in [0,1]")
    if amounts["small_blind"] is not None and amounts["big_blind"] is not None:
        if amounts["small_blind"] > amounts["big_blind"]:
            raise ValueError("small_blind cannot exceed big_blind")
    pending = [key for key, value in amounts.items()
               if value is None and key != "straddle_amount"]
    if mode == "unconfirmed":
        pending.append("straddle_mode")
    if mode in ("mandatory", "optional") and amounts["straddle_amount"] is None:
        pending.append("straddle_amount")
    if pending:
        return TableRulesResult(None, "table_rules_unverified", tuple(pending))
    if data.get("mode", "simulation") == "simulation":
        return TableRulesResult(None, "simulation_rules_only")
    if effects.strip():
        return TableRulesResult(None, "extra_table_effects_not_supported")
    if mode != "none":
        # The existing state/provider contracts do not encode forced straddle
        # position, action order, or raise reopening. A 2x blind substitution
        # would silently change the game instead of implementing those rules.
        return TableRulesResult(None, "straddle_strategy_not_supported")
    if amounts["straddle_amount"] not in (None, Decimal("0")):
        raise ValueError("straddle_amount conflicts with straddle_mode none")
    return TableRulesResult(GameConfig(
        variant="NLHE", game_type=GameType.CASH, max_seats=8,
        dealt_player_count=count, small_blind=ChipAmount(amounts["small_blind"]),
        big_blind=ChipAmount(amounts["big_blind"]),
        ante=ChipAmount(amounts["ante"]),
        rake_percent=amounts["rake_percent"],
        rake_cap=ChipAmount(amounts["rake_cap_bb"] * amounts["big_blind"]),
        minimum_chip=ChipAmount(amounts["minimum_chip"]),
    ), None)


def _rules_path() -> Path:
    override = os.environ.get("POKERSENSE_TABLE_RULES_PATH")
    if override:
        return Path(override)
    return _settings_path().with_name("table-rules.json")


def table_rules_revision(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_user_table_rules(default_path: Path) -> dict:
    path = _rules_path()
    document = json.loads(
        (path if path.exists() else default_path).read_text(encoding="utf-8")
    )
    validate_table_rules(document)
    return document


def save_user_table_rules(document: dict) -> dict:
    """Validate before replacing only the separate per-user rule document."""
    validate_table_rules(document)
    # Canonical JSON prevents a caller retaining mutable nested references.
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
    path = _rules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix="table-rules-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return json.loads(text)
