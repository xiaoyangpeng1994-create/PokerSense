"""Target-hardware calibration contract for adaptive equity defaults."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from poker_engine.core.enums import Street
from poker_engine.strategy.adaptive_equity import (
    AdaptiveEquityPolicy,
    EquityComputationStatus,
    calculate_adaptive_equity,
)
from poker_engine.strategy.contracts import RangeDistribution
from poker_engine.strategy.equity_cache import EquityMethod

from .helpers import NOW, context


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "configs" / "strategy" / "adaptive-equity-m1-pro-v1.json"


def _context():
    value = context(2, street=Street.FLOP)
    return replace(
        value,
        villain_ranges=(RangeDistribution(
            0,
            {"QsQd": Decimal("0.5"), "8c6c": Decimal("0.5")},
            "calibration-test",
            "v1",
            confidence=0.8,
        ),),
        request=replace(value.request, deadline_ms=300),
    )


def test_default_policy_matches_versioned_target_calibration():
    payload = json.loads(CALIBRATION.read_text())
    recommended = payload["recommended_conservative_policy"]
    policy = AdaptiveEquityPolicy()

    assert payload["environment"]["chip"] == "Apple M1 Pro"
    assert payload["environment"]["memory_gb"] == 32
    assert len(payload["cases"]) == 3
    assert policy.exact_outcomes_per_ms == recommended["exact_outcomes_per_ms"]
    assert policy.mc_trials_per_ms == recommended["mc_trials_per_ms"]
    assert policy.engine_version == "adaptive-equity-v2-m1-pro"
    exact_rate = payload["cases"][0]["units_per_ms_at_p95"]
    mc_rate = min(
        item["units_per_ms_at_p95"] for item in payload["cases"][1:]
    )
    assert exact_rate >= policy.exact_outcomes_per_ms * 2
    assert mc_rate >= policy.mc_trials_per_ms * 2


def test_calibration_records_the_exact_benchmark_tool_hash():
    payload = json.loads(CALIBRATION.read_text())
    tool = ROOT / payload["benchmark_tool"]

    assert hashlib.sha256(tool.read_bytes()).hexdigest() == (
        payload["benchmark_tool_sha256"]
    )


def test_default_300ms_flop_budget_is_conservative_and_partial():
    report = calculate_adaptive_equity(
        _context(), now=NOW, monotonic_clock=lambda: 0.0
    )

    assert report.method is EquityMethod.MONTE_CARLO
    assert report.trials == 600
    assert report.status is EquityComputationStatus.PARTIAL
    assert "planned=600" in report.evidence[0]
