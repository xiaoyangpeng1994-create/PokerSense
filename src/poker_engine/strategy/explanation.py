"""Deterministic Advice explanation that cannot alter strategy values."""

from __future__ import annotations

from dataclasses import dataclass

from .advice import Advice, AdviceStatus


@dataclass(frozen=True)
class AdviceExplanation:
    language: str
    summary: str
    key_factors: tuple[str, ...]


def explain_advice(advice: Advice, *, language: str = "zh") -> AdviceExplanation:
    if not isinstance(advice, Advice):
        raise TypeError("advice must be an Advice")
    if language not in ("zh", "en"):
        raise ValueError("language must be 'zh' or 'en'")
    if advice.status is AdviceStatus.READY:
        return _ready(advice, language)
    reasons = ",".join(advice.rejection_reasons) or "none"
    if language == "zh":
        return AdviceExplanation(
            language,
            f"当前建议状态：{advice.status.value}",
            (f"原因={reasons}", f"置信度={advice.confidence}"),
        )
    return AdviceExplanation(
        language,
        f"Advice status: {advice.status.value}",
        (f"reasons={reasons}", f"confidence={advice.confidence}"),
    )


def _ready(advice: Advice, language: str) -> AdviceExplanation:
    action = advice.preferred_action or max(
        advice.action_probabilities,
        key=lambda item: (advice.action_probabilities[item], item.value),
    )
    probability = advice.action_probabilities[action]
    sizes = ",".join(
        str(value.value) for value in advice.recommended_sizes.get(action, ())
    ) or "none"
    if language == "zh":
        return AdviceExplanation(
            language,
            f"首选动作：{action.value}",
            (
                f"频率={probability}",
                f"尺度={sizes}",
                f"策略来源={advice.strategy_source}@{advice.strategy_version}",
                f"匹配={advice.match_kind.value}:{advice.state_match_score}",
                f"置信度={advice.confidence}",
            ),
        )
    return AdviceExplanation(
        language,
        f"Preferred action: {action.value}",
        (
            f"frequency={probability}",
            f"sizes={sizes}",
            f"source={advice.strategy_source}@{advice.strategy_version}",
            f"match={advice.match_kind.value}:{advice.state_match_score}",
            f"confidence={advice.confidence}",
        ),
    )


__all__ = ["AdviceExplanation", "explain_advice"]
