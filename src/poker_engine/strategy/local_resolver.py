"""Fail-closed JSON subprocess adapter for a caller-configured local resolver."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from poker_engine.core._freeze import utc_now
from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from .contracts import DecisionContext
from .provider import (
    LookupState,
    MatchDimension,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)
from .serialization import strategy_serialize


RESOLVER_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class LocalResolverConfig:
    provider_id: str
    source_version: str
    command: tuple[str, ...]
    capability: ProviderCapability
    result_match_kind: MatchKind = MatchKind.INTERPOLATED
    timeout_ms: int = 2_000
    maximum_output_bytes: int = 1_048_576
    maximum_exploitability_bb100: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("provider_id", "source_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        command = tuple(self.command)
        if not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ValueError("command must contain non-empty strings")
        object.__setattr__(self, "command", command)
        if not isinstance(self.capability, ProviderCapability):
            raise TypeError("capability must be ProviderCapability")
        if not isinstance(self.result_match_kind, MatchKind):
            raise TypeError("result_match_kind must be MatchKind")
        if self.result_match_kind is MatchKind.EQUITY_ONLY:
            raise ValueError("resolver cannot return equity_only StrategyCandidate")
        if (
            self.result_match_kind is MatchKind.EXACT
            and self.capability.base_match_kind is not MatchKind.EXACT
        ):
            raise ValueError("resolver exact output requires exact capability")
        for name in ("timeout_ms", "maximum_output_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        exploitability = self.maximum_exploitability_bb100
        if exploitability is not None:
            if not isinstance(exploitability, Decimal):
                raise TypeError("maximum_exploitability_bb100 must be Decimal or None")
            if not exploitability.is_finite() or exploitability < 0:
                raise ValueError(
                    "maximum_exploitability_bb100 must be finite and >= 0"
                )


class LocalResolverProvider:
    """Run a local executable without a shell and validate its full response."""

    def __init__(self, config: LocalResolverConfig) -> None:
        if not isinstance(config, LocalResolverConfig):
            raise TypeError("config must be LocalResolverConfig")
        self._config = config

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    @property
    def source_version(self) -> str:
        return self._config.source_version

    @property
    def capability(self) -> ProviderCapability:
        return self._config.capability

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
        now = utc_now()
        if context.request.expires_at is not None and now >= context.request.expires_at:
            return self._rejected("resolver_request_expired")
        timeout_ms = self._config.timeout_ms
        if context.request.expires_at is not None:
            remaining = int((context.request.expires_at - now).total_seconds() * 1000)
            if remaining <= 0:
                return self._rejected("resolver_request_expired")
            timeout_ms = min(timeout_ms, remaining)
        request = {
            "schema_version": RESOLVER_PROTOCOL_VERSION,
            "type": "ResolverRequest",
            "provider_id": self.provider_id,
            "source_version": self.source_version,
            "context": strategy_serialize(context),
        }
        try:
            completed = subprocess.run(
                self._config.command,
                input=json.dumps(request, separators=(",", ":")).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_ms / 1000,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return self._rejected("resolver_timeout")
        except OSError as exc:
            return self._rejected(f"resolver_process_error:{type(exc).__name__}")
        if completed.returncode != 0:
            return self._rejected(f"resolver_exit:{completed.returncode}")
        if len(completed.stdout) > self._config.maximum_output_bytes:
            return self._rejected("resolver_output_too_large")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._rejected("resolver_invalid_json")
        try:
            return self._parse_response(context, payload, now)
        except (KeyError, TypeError, ValueError) as exc:
            return self._rejected(f"resolver_invalid_response:{type(exc).__name__}")

    def _parse_response(
        self,
        context: DecisionContext,
        payload: Any,
        produced_at,
    ) -> ProviderResult:
        if not isinstance(payload, Mapping):
            raise TypeError("response must be an object")
        if payload.get("schema_version") != RESOLVER_PROTOCOL_VERSION:
            raise ValueError("unsupported protocol version")
        if payload.get("type") != "ResolverResponse":
            raise ValueError("wrong response type")
        if payload.get("provider_id") != self.provider_id:
            raise ValueError("provider identity mismatch")
        if payload.get("source_version") != self.source_version:
            raise ValueError("source version mismatch")
        identity = payload.get("identity")
        if not isinstance(identity, Mapping) or (
            identity.get("hand_id") != context.hand_id
            or identity.get("state_version") != context.state_version
            or identity.get("request_id") != context.request_id
        ):
            raise ValueError("context identity mismatch")
        status = payload.get("status")
        if status == "NO_STRATEGY":
            return ProviderResult(
                LookupState.NOT_FOUND,
                self.provider_id,
                reasons=("resolver_no_strategy",),
            )
        if status == "NOT_CONVERGED":
            return self._rejected("resolver_not_converged")
        if status != "CONVERGED":
            raise ValueError("unknown resolver status")
        iterations = payload.get("iterations")
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive int")
        exploitability_raw = payload.get("exploitability_bb100")
        exploitability = (
            Decimal(exploitability_raw) if exploitability_raw is not None else None
        )
        if exploitability is not None and (
            not exploitability.is_finite() or exploitability < 0
        ):
            raise ValueError("invalid exploitability")
        maximum = self._config.maximum_exploitability_bb100
        if maximum is not None and (
            exploitability is None or exploitability > maximum
        ):
            return self._rejected("resolver_convergence_threshold_not_met")
        probabilities = _action_decimal_map(payload["action_probabilities"])
        sizes = {
            ActionType(action): tuple(ChipAmount(value) for value in values)
            for action, values in payload.get("recommended_sizes", {}).items()
        }
        action_ev = {
            ActionType(action): ChipDelta(value)
            for action, value in payload.get("action_ev", {}).items()
        }
        confidence = payload.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise TypeError("confidence must be numeric")
        candidate = StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            match_kind=self._config.result_match_kind,
            state_match_score=float(payload.get("state_match_score", 1.0)),
            match_dimensions=_match_dimensions(payload.get("match_dimensions", [])),
            action_probabilities=probabilities,
            recommended_sizes=sizes,
            action_ev=action_ev,
            confidence=float(confidence),
            evidence=(
                f"local-resolver://{self.provider_id}/{self.source_version}",
                f"resolver_iterations:{iterations}",
                f"resolver_exploitability_bb100:{exploitability}",
            ),
            assumptions=("local_resolver_subprocess",),
            produced_at=produced_at,
            expires_at=context.request.expires_at,
        )
        state = (
            LookupState.HIT_EXACT
            if candidate.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE
        )
        return ProviderResult(state, self.provider_id, candidate)

    def _rejected(self, reason: str) -> ProviderResult:
        return ProviderResult(
            LookupState.REJECTED,
            self.provider_id,
            reasons=(reason,),
        )


def _action_decimal_map(value: Any) -> dict[ActionType, Decimal]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("action_probabilities must be a non-empty object")
    return {
        ActionType(action): Decimal(probability)
        for action, probability in value.items()
    }


def _match_dimensions(value: Any) -> tuple[MatchDimension, ...]:
    if not isinstance(value, list):
        raise TypeError("match_dimensions must be an array")
    dimensions = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("match dimension must be an object")
        dimensions.append(MatchDimension(
            name=item["name"],
            requested=item["requested"],
            matched=item["matched"],
            distance=Decimal(item["distance"]),
            maximum_distance=Decimal(item["maximum_distance"]),
        ))
    return tuple(dimensions)


__all__ = [
    "LocalResolverConfig",
    "LocalResolverProvider",
    "RESOLVER_PROTOCOL_VERSION",
]
