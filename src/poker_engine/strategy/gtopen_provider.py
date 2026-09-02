"""Fail-closed local adapter for GTOpen's multiway Preflop Lab API.

GTOpen remains an optional, separately checked-out local service.  PokerSense
does not copy, start, bundle, or redistribute the upstream source.  This
adapter serializes one preflop solve at a time because GTOpen owns one mutable
preflop session per server process.
"""

from __future__ import annotations

import json
import math
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from poker_engine.core._freeze import utc_now
from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Street,
)
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.value_objects import ChipAmount

from .contracts import DecisionContext, GameType
from .provider import (
    ActionOption,
    LookupState,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)


ADAPTER_VERSION = "gtopen-adapter-v1"
PROVIDER_ID = "gtopen-local-preflop"
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_PLAYER_EVENTS = frozenset({
    EventType.FOLD,
    EventType.CHECK,
    EventType.CALL,
    EventType.BET,
    EventType.RAISE,
    EventType.ALL_IN,
})
_POSITION_ORDERS = {
    2: (Position.BTN, Position.BB),
    3: (Position.BTN, Position.SB, Position.BB),
    4: (Position.CO, Position.BTN, Position.SB, Position.BB),
    5: (Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB),
    6: (
        Position.UTG, Position.HJ, Position.CO, Position.BTN,
        Position.SB, Position.BB,
    ),
    7: (
        Position.UTG, Position.LJ, Position.HJ, Position.CO,
        Position.BTN, Position.SB, Position.BB,
    ),
    8: (
        Position.UTG, Position.UTG1, Position.LJ, Position.HJ,
        Position.CO, Position.BTN, Position.SB, Position.BB,
    ),
    9: (
        Position.UTG, Position.UTG1, Position.UTG2, Position.LJ,
        Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB,
    ),
}
_ACTION_LINES = frozenset({
    "unopened", "limp", "multi_limp", "raise", "three_bet",
    "four_bet", "squeeze", "iso_raise", "all_in",
})


class GTOpenError(RuntimeError):
    """One contained transport, protocol, context, or convergence failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@runtime_checkable
class GTOpenTransport(Protocol):
    def get(self, path: str, timeout_seconds: float) -> Mapping[str, Any]: ...

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class GTOpenConfig:
    """Versioned, capability-bounded settings for one local GTOpen service."""

    source_revision: str
    base_url: str = "http://127.0.0.1:3737"
    stack_buckets_bb: tuple[Decimal, ...] = (
        Decimal("20"), Decimal("40"), Decimal("100"),
    )
    ante_values_bb: tuple[Decimal, ...] = (Decimal("0"),)
    rake_percent_values: tuple[Decimal, ...] = (Decimal("0"),)
    open_raises_bb: tuple[Decimal, ...] = (
        Decimal("2"), Decimal("2.5"), Decimal("3"),
    )
    raise_multipliers: tuple[Decimal, ...] = (
        Decimal("2.5"), Decimal("3"),
    )
    max_raises: int = 4
    add_allin: bool = True
    allow_limp: bool = True
    realization: str = "calibrated"
    iterations: int = 500
    check_every: int = 25
    target_gap_bb: Decimal = Decimal("0.01")
    timeout_ms: int = 30_000
    poll_interval_ms: int = 25
    maximum_response_bytes: int = 4_194_304
    model_confidence_cap: float = 0.60

    def __post_init__(self) -> None:
        if not isinstance(self.source_revision, str) or not _REVISION_RE.fullmatch(
            self.source_revision
        ):
            raise ValueError("source_revision must be a 40-character git SHA")
        _validate_local_base_url(self.base_url)
        for name in (
            "stack_buckets_bb", "ante_values_bb", "rake_percent_values",
            "open_raises_bb", "raise_multipliers",
        ):
            values = tuple(getattr(self, name))
            if not values or not all(
                isinstance(value, Decimal)
                and value.is_finite()
                and value >= 0
                for value in values
            ):
                raise ValueError(f"{name} must contain finite non-negative Decimals")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        if self.stack_buckets_bb[0] <= 0:
            raise ValueError("stack_buckets_bb must be positive")
        if self.open_raises_bb[0] <= 0 or self.raise_multipliers[0] <= 1:
            raise ValueError("raise menus must contain meaningful positive values")
        if any(value > 1 for value in self.rake_percent_values):
            raise ValueError("rake_percent_values must be fractions in [0, 1]")
        for name in (
            "max_raises", "iterations", "check_every", "timeout_ms",
            "poll_interval_ms", "maximum_response_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.max_raises > 8:
            raise ValueError("max_raises must be <= 8")
        if self.realization not in {"raw", "static", "calibrated"}:
            raise ValueError("unsupported realization model")
        if (
            not isinstance(self.target_gap_bb, Decimal)
            or not self.target_gap_bb.is_finite()
            or self.target_gap_bb < 0
        ):
            raise ValueError("target_gap_bb must be finite and non-negative")
        if not isinstance(self.model_confidence_cap, (int, float)) or isinstance(
            self.model_confidence_cap, bool
        ):
            raise TypeError("model_confidence_cap must be numeric")
        if not math.isfinite(self.model_confidence_cap) or not (
            0 < self.model_confidence_cap <= 1
        ):
            raise ValueError("model_confidence_cap must be in (0, 1]")


class UrlLibGTOpenTransport:
    """Small JSON client restricted to the configured loopback server."""

    def __init__(self, config: GTOpenConfig) -> None:
        if not isinstance(config, GTOpenConfig):
            raise TypeError("config must be GTOpenConfig")
        self._base_url = config.base_url.rstrip("/")
        self._maximum_response_bytes = config.maximum_response_bytes

    def get(self, path: str, timeout_seconds: float) -> Mapping[str, Any]:
        return self._request("GET", path, None, timeout_seconds)

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return self._request("POST", path, payload, timeout_seconds)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        if not path.startswith("/api/"):
            raise GTOpenError("gtopen_invalid_api_path")
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self._base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=max(timeout_seconds, 0.001)) as response:
                raw = response.read(self._maximum_response_bytes + 1)
        except HTTPError as exc:
            raise GTOpenError(f"gtopen_http_status:{exc.code}") from exc
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise GTOpenError(
                f"gtopen_transport_error:{type(exc).__name__}"
            ) from exc
        if len(raw) > self._maximum_response_bytes:
            raise GTOpenError("gtopen_response_too_large")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GTOpenError("gtopen_invalid_json") from exc
        if not isinstance(value, Mapping):
            raise GTOpenError("gtopen_invalid_response")
        return value


@dataclass(frozen=True)
class _ObservedAction:
    event: StateEvent
    seat_id: int
    position: Position
    kind: str
    total_bb: Decimal | None = None


class GTOpenPreflopProvider:
    """Use a separately running GTOpen service as a bounded Slow Provider."""

    def __init__(
        self,
        config: GTOpenConfig,
        transport: GTOpenTransport | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, GTOpenConfig):
            raise TypeError("config must be GTOpenConfig")
        if transport is not None and not isinstance(transport, GTOpenTransport):
            raise TypeError("transport must implement GTOpenTransport")
        for name, value in (
            ("clock", clock), ("monotonic", monotonic), ("sleep", sleep),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        self._config = config
        self._transport = transport or UrlLibGTOpenTransport(config)
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._session_lock = threading.Lock()
        self._capability = ProviderCapability(
            player_counts=frozenset(range(2, 10)),
            streets=frozenset({Street.PREFLOP}),
            game_types=frozenset({GameType.CASH}),
            stack_buckets_bb=config.stack_buckets_bb,
            ante_values=tuple(ChipAmount(value) for value in config.ante_values_bb),
            rake_percent_values=config.rake_percent_values,
            action_lines=_ACTION_LINES,
            base_match_kind=MatchKind.HEURISTIC,
            ante_values_are_bb=True,
            hero_positions=frozenset(
                position
                for order in _POSITION_ORDERS.values()
                for position in order
            ),
            priority=300,
        )

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    @property
    def source_version(self) -> str:
        return f"{ADAPTER_VERSION}:{self._config.source_revision}"

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    def query(self, context: DecisionContext) -> ProviderResult:
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be DecisionContext")
        match = self.capability.match(context)
        if not match.applicable:
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=match.reasons,
            )
        try:
            deadline = self._deadline(context)
            remaining = deadline - self._monotonic()
            if remaining <= 0 or not self._session_lock.acquire(timeout=remaining):
                raise GTOpenError("gtopen_timeout_waiting_for_session")
            try:
                return self._query_locked(context, deadline)
            finally:
                self._session_lock.release()
        except GTOpenError as exc:
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(exc.reason,),
            )
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(f"gtopen_invalid_response:{type(exc).__name__}",),
            )

    def _query_locked(
        self,
        context: DecisionContext,
        deadline: float,
    ) -> ProviderResult:
        if not context.is_decision_ready:
            raise GTOpenError("gtopen_context_not_ready")
        positions, seat_by_position = _ordered_positions(context)
        observed = _observed_actions(context, seat_by_position)
        _validate_action_line(context, observed)
        request = _build_spot_request(context, positions, observed, self._config)
        built = self._post("/api/preflop/spot", request, deadline)
        nodes = _positive_int(built.get("nodes"), "nodes")
        action_nodes = _positive_int(
            built.get("action_nodes"), "action_nodes"
        )
        arena_mb = _finite_decimal(built.get("arena_mb"), "arena_mb")
        self._post(
            "/api/preflop/solve",
            {
                "iterations": self._config.iterations,
                "check_every": self._config.check_every,
                "target_gap": float(self._config.target_gap_bb),
            },
            deadline,
        )
        try:
            status = self._wait_for_result(deadline)
        except GTOpenError:
            self._stop_best_effort()
            raise
        iteration = _positive_int(status.get("iteration"), "iteration")
        gap = _finite_decimal(status.get("gap_total"), "gap_total")
        if gap < 0:
            raise GTOpenError("gtopen_invalid_gap")
        if gap > self._config.target_gap_bb:
            raise GTOpenError("gtopen_convergence_threshold_not_met")
        path: list[int] = []
        for action in observed:
            node = self._post(
                "/api/preflop/node", {"path": list(path)}, deadline
            )
            path.append(_match_history_action(node, action))
        node = self._post(
            "/api/preflop/node", {"path": list(path)}, deadline
        )
        return self._candidate_from_node(
            context,
            node,
            positions,
            iteration,
            gap,
            nodes,
            action_nodes,
            arena_mb,
        )

    def _candidate_from_node(
        self,
        context: DecisionContext,
        node: Mapping[str, Any],
        positions: Sequence[Position],
        iteration: int,
        gap: Decimal,
        nodes: int,
        action_nodes: int,
        arena_mb: Decimal,
    ) -> ProviderResult:
        if node.get("kind") != "action":
            raise GTOpenError("gtopen_current_node_is_terminal")
        actor = node.get("actor")
        if not isinstance(actor, int) or isinstance(actor, bool):
            raise GTOpenError("gtopen_missing_actor")
        if actor < 0 or actor >= len(positions):
            raise GTOpenError("gtopen_actor_out_of_range")
        hero_position = next(
            seat.position
            for seat in context.seats
            if seat.seat_id == context.hero_seat
        )
        if positions[actor] is not hero_position:
            raise GTOpenError("gtopen_actor_mismatch")
        actions = node.get("actions")
        strategy = node.get("strategy")
        if not isinstance(actions, list) or not actions:
            raise GTOpenError("gtopen_missing_actions")
        if not isinstance(strategy, list) or len(strategy) != len(actions) * 169:
            raise GTOpenError("gtopen_strategy_shape_mismatch")
        hand_index = gtopen_hand_class_index(context.hero_cards)
        raw = [
            _finite_decimal(strategy[index * 169 + hand_index], "strategy")
            for index in range(len(actions))
        ]
        if any(value < 0 for value in raw) or sum(raw, Decimal("0")) <= 0:
            raise GTOpenError("gtopen_invalid_strategy_values")
        probabilities = _normalize(raw)
        options = []
        legal = context.legal_action_types
        big_blind = context.game_config.big_blind.value
        for action_payload, probability in zip(actions, probabilities):
            if not isinstance(action_payload, Mapping):
                raise GTOpenError("gtopen_invalid_action")
            action = _action_type(action_payload.get("kind"))
            if action not in legal:
                raise GTOpenError("gtopen_action_not_legal")
            amount = None
            if action in (ActionType.RAISE, ActionType.ALL_IN):
                to_bb = _finite_decimal(action_payload.get("to"), "action.to")
                amount = ChipAmount(to_bb * big_blind)
            label = action_payload.get("label")
            if not isinstance(label, str) or not label:
                raise GTOpenError("gtopen_action_missing_label")
            options.append(ActionOption(action, probability, amount, label))
        action_probabilities: dict[ActionType, Decimal] = {}
        recommended: dict[ActionType, list[ChipAmount]] = {}
        for option in options:
            action_probabilities[option.action] = (
                action_probabilities.get(option.action, Decimal("0"))
                + option.probability
            )
            if option.amount is not None:
                values = recommended.setdefault(option.action, [])
                if option.amount not in values:
                    values.append(option.amount)
        confidence = min(
            context.input_quality.overall_confidence,
            float(self._config.model_confidence_cap),
        )
        assumptions = [
            "gtopen_separate_local_service",
            "gtopen_server_revision_configured_not_remotely_attested",
            "gtopen_preflop_equity_realization_model",
            "gtopen_equal_starting_stacks_v1",
            "gtopen_result_not_licensed_for_redistribution",
        ]
        if len(positions) >= 3:
            assumptions.extend((
                "gtopen_multiway_product_equity_approximation",
                "multi_player_cfr_equilibrium_not_unique_gto",
            ))
        candidate = StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            match_kind=MatchKind.HEURISTIC,
            state_match_score=1.0,
            action_probabilities=action_probabilities,
            recommended_sizes={
                action: tuple(values) for action, values in recommended.items()
            },
            action_options=tuple(options),
            confidence=confidence,
            evidence=(
                f"gtopen://configured-source/{self._config.source_revision}",
                f"gtopen_adapter:{ADAPTER_VERSION}",
                f"gtopen_realization:{self._config.realization}",
                f"gtopen_iterations:{iteration}",
                f"gtopen_model_gap_bb:{gap}",
                f"gtopen_tree:nodes={nodes};action_nodes={action_nodes};"
                f"arena_mb={arena_mb}",
            ),
            assumptions=tuple(assumptions),
            produced_at=self._clock(),
            expires_at=context.request.expires_at,
        )
        return ProviderResult(
            LookupState.HIT_APPROXIMATE,
            self.provider_id,
            candidate,
        )

    def _deadline(self, context: DecisionContext) -> float:
        seconds = self._config.timeout_ms / 1000
        now = self._clock()
        if context.request.expires_at is not None:
            seconds = min(
                seconds,
                (context.request.expires_at - now).total_seconds(),
            )
        if seconds <= 0:
            raise GTOpenError("gtopen_request_expired")
        return self._monotonic() + seconds

    def _remaining(self, deadline: float) -> float:
        value = deadline - self._monotonic()
        if value <= 0:
            raise GTOpenError("gtopen_timeout")
        return value

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        deadline: float,
    ) -> Mapping[str, Any]:
        return self._transport.post(path, payload, self._remaining(deadline))

    def _wait_for_result(self, deadline: float) -> Mapping[str, Any]:
        while True:
            status = self._transport.get(
                "/api/preflop/status", self._remaining(deadline)
            )
            state = status.get("state")
            if state == "done":
                return status
            if state in {"error", "stopped", "idle"}:
                raise GTOpenError(f"gtopen_solve_state:{state}")
            if state != "running":
                raise GTOpenError("gtopen_unknown_solve_state")
            delay = min(
                self._config.poll_interval_ms / 1000,
                self._remaining(deadline),
            )
            self._sleep(delay)

    def _stop_best_effort(self) -> None:
        try:
            self._transport.post("/api/preflop/stop", {}, 0.25)
        except Exception:
            pass


def _validate_local_base_url(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("base_url must be a non-empty str")
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("base_url must be a plain loopback HTTP origin")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("base_url must include a valid port")


def _ordered_positions(
    context: DecisionContext,
) -> tuple[tuple[Position, ...], dict[Position, int]]:
    count = context.game_config.dealt_player_count
    order = _POSITION_ORDERS[count]
    seats = tuple(seat for seat in context.seats if seat.occupied)
    if len(seats) != count:
        raise GTOpenError("gtopen_dealt_seat_count_mismatch")
    positions = [seat.position for seat in seats]
    if len(set(positions)) != count or set(positions) != set(order):
        raise GTOpenError("gtopen_position_mapping_mismatch")
    seat_by_position = {seat.position: seat.seat_id for seat in seats}
    return order, seat_by_position


def _observed_actions(
    context: DecisionContext,
    seat_by_position: Mapping[Position, int],
) -> tuple[_ObservedAction, ...]:
    position_by_seat = {seat: pos for pos, seat in seat_by_position.items()}
    result = []
    for event in context.action_history:
        if event.event_type not in _PLAYER_EVENTS:
            continue
        if event.hand_id != context.hand_id:
            raise GTOpenError("gtopen_event_hand_mismatch")
        if event.event_type is EventType.BET:
            raise GTOpenError("gtopen_preflop_bet_event_invalid")
        seat_id = event.payload.get("seat_id")
        if not isinstance(seat_id, int) or isinstance(seat_id, bool):
            raise GTOpenError("gtopen_event_missing_seat")
        position = position_by_seat.get(seat_id)
        if position is None:
            raise GTOpenError("gtopen_event_unknown_seat")
        kind = {
            EventType.FOLD: "fold",
            EventType.CHECK: "check",
            EventType.CALL: "call",
            EventType.RAISE: "raise",
            EventType.ALL_IN: "jam",
        }[event.event_type]
        total_bb = None
        if event.event_type in (EventType.RAISE, EventType.ALL_IN):
            amount = _event_total_street(event)
            total_bb = amount / context.game_config.big_blind.value
        result.append(_ObservedAction(event, seat_id, position, kind, total_bb))
    return tuple(result)


def _event_total_street(event: StateEvent) -> Decimal:
    value = event.payload.get("amount_total_street")
    semantics = event.payload.get("amount_semantics")
    semantics = getattr(semantics, "value", semantics)
    if value is None and semantics == "total_street":
        value = event.payload.get("amount")
    if isinstance(value, ChipAmount):
        amount = value.value
    elif isinstance(value, Decimal):
        amount = value
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        amount = Decimal(str(value))
    else:
        raise GTOpenError("gtopen_aggressive_amount_missing")
    if not amount.is_finite() or amount <= 0:
        raise GTOpenError("gtopen_aggressive_amount_invalid")
    return amount


def _build_spot_request(
    context: DecisionContext,
    positions: Sequence[Position],
    observed: Sequence[_ObservedAction],
    config: GTOpenConfig,
) -> Mapping[str, Any]:
    big_blind = context.game_config.big_blind.value
    seat_by_position = {seat.position: seat for seat in context.seats}
    starting = []
    for position in positions:
        seat = seat_by_position[position]
        if seat.status in (PlayerStatus.SITTING_OUT, PlayerStatus.UNKNOWN):
            raise GTOpenError("gtopen_unsupported_seat_status")
        starting.append((seat.stack.value + seat.hand_committed.value) / big_blind)
    if any(value != starting[0] for value in starting[1:]):
        raise GTOpenError("gtopen_requires_equal_starting_stacks")
    if starting[0] not in config.stack_buckets_bb:
        raise GTOpenError("gtopen_unsupported_starting_stack")
    posts = []
    for position in positions:
        if position is Position.BB:
            posts.append(Decimal("1"))
        elif position is Position.SB or (
            len(positions) == 2 and position is Position.BTN
        ):
            posts.append(
                context.game_config.small_blind.value / big_blind
            )
        else:
            posts.append(Decimal("0"))
    opens = set(config.open_raises_bb)
    multipliers = set(config.raise_multipliers)
    current_to = max(posts)
    raises = 0
    for action in observed:
        if action.kind not in {"raise", "jam"}:
            continue
        if action.total_bb is None:
            raise GTOpenError("gtopen_aggressive_amount_missing")
        if raises == 0:
            opens.add(action.total_bb)
        else:
            if current_to <= 0:
                raise GTOpenError("gtopen_invalid_previous_raise")
            multipliers.add(action.total_bb / current_to)
        current_to = action.total_bb
        raises += 1
    if raises > config.max_raises:
        raise GTOpenError("gtopen_too_many_raises")
    return {
        "positions": [position.value for position in positions],
        "stack": float(starting[0]),
        "posts": [float(value) for value in posts],
        "ante": float(context.game_config.ante.value / big_blind),
        "limp": config.allow_limp,
        "open_raises": [float(value) for value in sorted(opens)],
        "raise_mults": [float(value) for value in sorted(multipliers)],
        "max_raises": config.max_raises,
        "add_allin": config.add_allin,
        "allin_threshold": 1.0,
        "rake_pct": float(context.game_config.rake_percent * Decimal("100")),
        "rake_cap": float(context.game_config.rake_cap.value / big_blind),
        "no_flop_no_drop": True,
        "realization": config.realization,
        "call_only_seats": [],
    }


def _validate_action_line(
    context: DecisionContext,
    observed: Sequence[_ObservedAction],
) -> None:
    aggressive = [
        index for index, action in enumerate(observed)
        if action.kind in {"raise", "jam"}
    ]
    if any(action.kind == "jam" for action in observed):
        expected = "all_in"
    elif not aggressive:
        limps = sum(action.kind == "call" for action in observed)
        expected = "unopened" if limps == 0 else (
            "limp" if limps == 1 else "multi_limp"
        )
    elif len(aggressive) == 1:
        expected = (
            "iso_raise"
            if any(
                action.kind == "call"
                for action in observed[:aggressive[0]]
            )
            else "raise"
        )
    elif len(aggressive) == 2:
        expected = (
            "squeeze"
            if any(
                action.kind == "call"
                for action in observed[aggressive[0] + 1:aggressive[1]]
            )
            else "three_bet"
        )
    else:
        expected = "four_bet"
    if context.action_line != expected:
        raise GTOpenError("gtopen_action_line_mismatch")


def _match_history_action(
    node: Mapping[str, Any],
    observed: _ObservedAction,
) -> int:
    if node.get("kind") != "action":
        raise GTOpenError("gtopen_history_reaches_terminal_early")
    actor_pos = node.get("actor_pos")
    if actor_pos != observed.position.value:
        raise GTOpenError("gtopen_history_actor_mismatch")
    actions = node.get("actions")
    if not isinstance(actions, list):
        raise GTOpenError("gtopen_missing_actions")
    matches = []
    for index, item in enumerate(actions):
        if not isinstance(item, Mapping) or item.get("kind") != observed.kind:
            continue
        if observed.total_bb is not None:
            value = _finite_decimal(item.get("to"), "action.to")
            if value != observed.total_bb:
                continue
        matches.append(index)
    if len(matches) != 1:
        raise GTOpenError("gtopen_history_action_not_exact")
    return matches[0]


def gtopen_hand_class_index(cards: Sequence[Any]) -> int:
    """Return GTOpen's exact 169-class index for two concrete cards."""
    if len(cards) != 2:
        raise GTOpenError("gtopen_requires_two_hero_cards")
    first, second = cards
    try:
        a = first.rank_value - 2
        b = second.rank_value - 2
        suited = first.suit == second.suit
    except AttributeError as exc:
        raise GTOpenError("gtopen_invalid_hero_cards") from exc
    hi, lo = max(a, b), min(a, b)
    if hi == lo:
        return hi * 13 + hi
    if suited:
        return hi * 13 + lo
    return lo * 13 + hi


def _action_type(value: object) -> ActionType:
    try:
        return {
            "fold": ActionType.FOLD,
            "check": ActionType.CHECK,
            "call": ActionType.CALL,
            "raise": ActionType.RAISE,
            "jam": ActionType.ALL_IN,
        }[value]
    except (KeyError, TypeError) as exc:
        raise GTOpenError("gtopen_unknown_action_kind") from exc


def _normalize(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    total = sum(values, Decimal("0"))
    if total <= 0:
        raise GTOpenError("gtopen_zero_strategy_mass")
    result = [value / total for value in values[:-1]]
    result.append(Decimal("1") - sum(result, Decimal("0")))
    if any(value < 0 or value > 1 for value in result):
        raise GTOpenError("gtopen_invalid_normalized_strategy")
    return tuple(result)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GTOpenError(f"gtopen_invalid_{name}")
    return value


def _finite_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise GTOpenError(f"gtopen_invalid_{name}")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise GTOpenError(f"gtopen_invalid_{name}") from exc
    if not result.is_finite():
        raise GTOpenError(f"gtopen_invalid_{name}")
    return result


__all__ = [
    "ADAPTER_VERSION",
    "GTOpenConfig",
    "GTOpenError",
    "GTOpenPreflopProvider",
    "GTOpenTransport",
    "PROVIDER_ID",
    "UrlLibGTOpenTransport",
    "gtopen_hand_class_index",
]
