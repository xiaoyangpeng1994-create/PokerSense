"""Recognition trace — reproducible metadata for a single recognition step.

Memory-only at runtime; the benchmark harness may write JSON/JSONL artifacts.
A trace references the asset manifest version/SHA rather than embedding it.
No hidden chain-of-thought: only structured, auditable fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from poker_engine.core.observation import ValidationStatus


@dataclass(frozen=True)
class RecognitionTrace:
    frame_seq: int
    roi_key: str                      # e.g. "board_cards", "stacks:0"
    slot_id: int | None               # None for global ROIs
    recognizer_name: str
    recognizer_version: str
    raw_score: float
    confidence: float
    validation_status: ValidationStatus
    manifest_sha: str
    template_config_version: str | None = None
    # Optional per-component raw scores (e.g. board occupancy vs identity),
    # recorded so the aggregate raw_score remains auditable.
    components: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_score", _finite_01(self.raw_score, "raw_score"))
        object.__setattr__(self, "confidence",
                           _finite_01(self.confidence, "confidence"))
        if self.components is not None:
            frozen = {}
            for k, v in self.components.items():
                frozen[k] = _finite_01(v, f"components.{k}")
            object.__setattr__(self, "components", MappingProxyType(frozen))

    def to_dict(self) -> dict:
        out = {
            "frame_seq": self.frame_seq,
            "roi_key": self.roi_key,
            "slot_id": self.slot_id,
            "recognizer_name": self.recognizer_name,
            "recognizer_version": self.recognizer_version,
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "validation_status": self.validation_status.value,
            "manifest_sha": self.manifest_sha,
            "template_config_version": self.template_config_version,
        }
        if self.components is not None:
            out["components"] = dict(self.components)
        return out


def _finite_01(v: float, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be a float")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite")
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]")
    return f


__all__ = ["RecognitionTrace"]
