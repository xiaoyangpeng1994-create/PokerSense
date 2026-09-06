"""Single registration point for every released Strategy Provider.

Every production entrypoint builds its router here instead of hand-rolling a
``StrategyRouter``.  Capability expansion (7/8-max coverage, postflop, a new
asset) appends to this module only -- callers keep working unchanged.

Registration policy
-------------------
* **Bundled asset providers are mandatory.**  An integrity failure raises and
  aborts startup.  Shipping a corrupt or unreviewed strategy asset silently
  would turn real advice into unauditable guesswork, which the project's
  fail-closed rule forbids.
* **Optional external providers are opt-in.**  GTOpen is a separately checked
  out local service; an unconfigured or unreachable one simply is not
  registered, so startup is never blocked by an optional dependency.
"""

from __future__ import annotations

from poker_engine.strategy.heuristic_provider import (
    PreflopRfiHeuristicProvider,
)
from poker_engine.strategy.provider import StrategyProvider
from poker_engine.strategy.router import StrategyRouter

__all__ = [
    "build_strategy_router",
    "registered_provider_ids",
]


def build_strategy_router(
    *,
    gtopen_base_url: str | None = None,
) -> StrategyRouter:
    """Build the production router over every releasable Provider.

    ``gtopen_base_url`` opts into the optional local GTOpen service.  It stays
    ``None`` by default because PokerSense neither bundles nor starts that
    upstream service.
    """
    providers: list[StrategyProvider] = []

    # Bundled, hash-pinned, MIT-licensed RFI chart: 6-max and 9-max, preflop,
    # 100bb, ante 0, rake 0, action line "unopened" only.
    providers.append(PreflopRfiHeuristicProvider.from_builtin())

    if gtopen_base_url:
        providers.append(_build_gtopen_provider(gtopen_base_url))

    return StrategyRouter(tuple(providers))


def registered_provider_ids(
    *,
    gtopen_base_url: str | None = None,
) -> tuple[str, ...]:
    """Provider ids registered for a given configuration (for diagnostics)."""
    return tuple(
        provider.provider_id
        for provider in build_strategy_router(
            gtopen_base_url=gtopen_base_url,
        ).providers
    )


def _build_gtopen_provider(base_url: str) -> StrategyProvider:
    from poker_engine.strategy.gtopen_provider import (
        GTOpenConfig,
        GTOpenPreflopProvider,
    )

    return GTOpenPreflopProvider(GTOpenConfig(base_url=base_url))
