"""Confidence Gate package."""

from .errors import ConfidenceGateError
from .gate import ConfidenceGate, ConfidenceGateResult

__all__ = ["ConfidenceGate", "ConfidenceGateResult", "ConfidenceGateError"]
