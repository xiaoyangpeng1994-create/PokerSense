"""Realtime analysis layer (Task 8).

Ties FrameSource -> Vision -> StateEngine -> analysis output together via an
event-driven loop. Produces a realtime state + equity + confidence snapshot;
does NOT auto-operate the table or recommend actions.

The lightweight analysis/equity contracts are imported eagerly. Capture and
pipeline classes are loaded on first access so JSON serialization and strategy
tests do not require the optional OpenCV desktop stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .analysis import (
    ConfidenceSnapshot,
    EquitySnapshot,
    RealtimeAnalysis,
    StateSnapshot,
)
from .change_detector import ChangeReport, detect_change
from .equity import (
    EquityStrategy,
    ExactRandomRangeEquity,
    MonteCarloRandomRangeEquity,
)
from .temporal_consensus import (
    TemporalConsensus,
    TemporalConsensusResult,
)
from .hand_boundary import (
    HandBoundaryDetection,
    HandBoundaryPolicy,
    HandBoundaryStatus,
    detect_hand_boundary,
)

if TYPE_CHECKING:
    from .frame_source import FrameSource, SyntheticFrameSource
    from .pipeline import PipelineStep, RealtimePipeline


_LAZY_IMPORTS = {
    "FrameSource": (".frame_source", "FrameSource"),
    "SyntheticFrameSource": (".frame_source", "SyntheticFrameSource"),
    "PipelineStep": (".pipeline", "PipelineStep"),
    "RealtimePipeline": (".pipeline", "RealtimePipeline"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "FrameSource",
    "SyntheticFrameSource",
    "ChangeReport",
    "detect_change",
    "StateSnapshot",
    "EquitySnapshot",
    "ConfidenceSnapshot",
    "RealtimeAnalysis",
    "EquityStrategy",
    "MonteCarloRandomRangeEquity",
    "ExactRandomRangeEquity",
    "TemporalConsensus",
    "TemporalConsensusResult",
    "HandBoundaryDetection",
    "HandBoundaryPolicy",
    "HandBoundaryStatus",
    "detect_hand_boundary",
    "RealtimePipeline",
    "PipelineStep",
]
