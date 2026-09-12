"""Rule-fingerprint gate around existing providers for AA shadow evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poker_engine.core.enums import ActionType, Street

from .aa_asset_binding_v2 import AAStrategyAssetBindingV2
from .aa_rules_v2 import AARuleProfileV2, ForcedBetPlan, OpeningReconciliation
from .contracts import DecisionContext
from .provider import LookupState, StrategyProvider
from .router import RouteResult, StrategyRouter, StrategyRouteProvider


class AAShadowRouteStatus(str, Enum):
    BLOCKED = "BLOCKED"
    NO_STRATEGY = "NO_STRATEGY"
    SHADOW_CANDIDATE = "SHADOW_CANDIDATE"


@dataclass(frozen=True)
class AAShadowRouteEvaluation:
    status: AAShadowRouteStatus
    rule_fingerprint: str
    asset_binding_id: str
    reasons: tuple[str, ...]
    route: RouteResult | None = None
    advice_emitted: bool = False
    strategy_eligible: bool = False

    def __post_init__(self):
        if not isinstance(self.status, AAShadowRouteStatus):
            raise TypeError("status must be AAShadowRouteStatus")
        if not isinstance(self.rule_fingerprint, str) or not self.rule_fingerprint:
            raise ValueError("rule_fingerprint must be non-empty")
        if not isinstance(self.asset_binding_id, str) or not self.asset_binding_id:
            raise ValueError("asset_binding_id must be non-empty")
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise TypeError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)
        if self.status is AAShadowRouteStatus.SHADOW_CANDIDATE and self.route is None:
            raise ValueError("SHADOW_CANDIDATE requires RouteResult")
        if self.advice_emitted or self.strategy_eligible:
            raise ValueError("shadow route cannot become live advice")


class AARuleBoundShadowRouter:
    """Execute a provider only when rules, forced bets and sizing all match."""

    def __init__(
        self, router: StrategyRouteProvider, *, rule_fingerprint: str,
        asset_binding_id: str,
    ):
        if not isinstance(router, StrategyRouteProvider):
            raise TypeError("router must implement StrategyRouteProvider")
        if not isinstance(rule_fingerprint, str) or len(rule_fingerprint) != 64:
            raise ValueError("rule_fingerprint must be SHA-256")
        try:
            int(rule_fingerprint, 16)
        except ValueError as exc:
            raise ValueError("rule_fingerprint must be SHA-256") from exc
        if not isinstance(asset_binding_id, str) or not asset_binding_id:
            raise ValueError("asset_binding_id must be non-empty")
        self.router = router
        self.rule_fingerprint = rule_fingerprint.lower()
        self.asset_binding_id = asset_binding_id

    @classmethod
    def from_binding(cls, provider, binding):
        if not isinstance(provider, StrategyProvider):
            raise TypeError("provider must implement StrategyProvider")
        if not isinstance(binding, AAStrategyAssetBindingV2):
            raise TypeError("binding must be AAStrategyAssetBindingV2")
        if not binding.shadow_permitted:
            raise ValueError("strategy asset binding is not shadow reviewed")
        failures = binding.validate_provider(provider)
        if failures:
            raise ValueError("invalid AA asset binding: " + ",".join(failures))
        return cls(
            StrategyRouter((provider,)),
            rule_fingerprint=binding.rule_fingerprint,
            asset_binding_id=binding.binding_fingerprint,
        )

    def evaluate(
        self,
        context: DecisionContext,
        rules: AARuleProfileV2,
        plan: ForcedBetPlan,
        opening: OpeningReconciliation,
        *,
        now=None,
    ):
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be DecisionContext")
        if not isinstance(rules, AARuleProfileV2):
            raise TypeError("rules must be AARuleProfileV2")
        if not isinstance(plan, ForcedBetPlan):
            raise TypeError("plan must be ForcedBetPlan")
        if not isinstance(opening, OpeningReconciliation):
            raise TypeError("opening must be OpeningReconciliation")
        reasons = list(rules.live_strategy_blockers)
        if rules.fingerprint != self.rule_fingerprint:
            reasons.append("strategy_asset_rule_fingerprint_mismatch")
        if plan.rules_fingerprint != rules.fingerprint:
            reasons.append("forced_bet_plan_rule_fingerprint_mismatch")
        if context.game_config != rules.game_config():
            reasons.append("decision_context_rule_profile_mismatch")
        opening_exact = (
            opening.status == "EXACT_FORCED_BETS"
            and opening.strategy_eligible
            and dict(opening.expected) == dict(plan.contributions)
            and all(value == 0 for value in opening.differences.values())
            and opening.total_difference == 0
        )
        if not opening_exact:
            reasons.append("opening_forced_bets_not_exact_or_verified")
        if context.street is not Street.PREFLOP:
            reasons.append("postflop_rule_bound_provider_not_released")
        elif context.action_line != "unopened":
            reasons.append("post_opening_raise_increment_unverified")
        else:
            raises = [
                action for action in context.legal_actions
                if action.action is ActionType.RAISE
            ]
            if (len(raises) != 1
                    or raises[0].min_amount.value != plan.minimum_raise_to):
                reasons.append("straddle_legal_sizing_not_bound")
        if reasons:
            return AAShadowRouteEvaluation(
                AAShadowRouteStatus.BLOCKED,
                rules.fingerprint,
                self.asset_binding_id,
                tuple(dict.fromkeys(reasons)),
            )
        route = self.router.route(context, now=now)
        status = (
            AAShadowRouteStatus.SHADOW_CANDIDATE
            if route.state in (LookupState.HIT_EXACT, LookupState.HIT_APPROXIMATE)
            else AAShadowRouteStatus.NO_STRATEGY
        )
        return AAShadowRouteEvaluation(
            status,
            rules.fingerprint,
            self.asset_binding_id,
            route.reasons,
            route,
        )


__all__ = [
    "AARuleBoundShadowRouter", "AAShadowRouteEvaluation",
    "AAShadowRouteStatus",
]
