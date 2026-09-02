"""Vision asset manifest — versioned binding of detector configuration assets.

Runtime manifest is versioned configuration. It binds the versions used by a
Vision run so RecognitionTrace can reference them reproducibly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from poker_engine.core._freeze import freeze_mapping

_REQUIRED_FIELDS = (
    "platform_id",
    "layout_id",
    "card_layout_version",
    "template_set_version",
    "calibration_version",
    "recognizer_versions",
)


def _validate_schema(data: Mapping) -> None:
    """Deterministically reject malformed manifest data."""
    if not isinstance(data, Mapping):
        raise TypeError("manifest data must be a JSON object")
    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"manifest missing required field: {field!r}")

    platform_id = data["platform_id"]
    layout_id = data["layout_id"]
    tv = data["template_set_version"]
    if not isinstance(platform_id, str):
        raise TypeError("platform_id must be a str")
    if not isinstance(layout_id, str):
        raise TypeError("layout_id must be a str")
    if not isinstance(tv, str):
        raise TypeError("template_set_version must be a str")

    for field in ("card_layout_version", "calibration_version"):
        v = data[field]
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{field} must be an int")

    rv = data["recognizer_versions"]
    if not isinstance(rv, Mapping):
        raise TypeError("recognizer_versions must be a mapping")
    for k, v in rv.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                "recognizer_versions keys and values must be str"
            )


@dataclass(frozen=True)
class VisionAssetManifest:
    platform_id: str
    layout_id: str
    card_layout_version: int
    template_set_version: str          # SHA or version string
    calibration_version: int
    recognizer_versions: Mapping[str, str]   # detector name -> version

    def __post_init__(self) -> None:
        if not isinstance(self.platform_id, str) or not self.platform_id:
            raise ValueError("platform_id must be a non-empty str")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a non-empty str")
        if isinstance(self.card_layout_version, bool) or not isinstance(
            self.card_layout_version, int
        ):
            raise TypeError("card_layout_version must be an int")
        if not isinstance(self.template_set_version, str) or (
            not self.template_set_version
        ):
            raise ValueError("template_set_version must be a non-empty str")
        if isinstance(self.calibration_version, bool) or not isinstance(
            self.calibration_version, int
        ):
            raise TypeError("calibration_version must be an int")
        # Deep-immutable exposure (MappingProxyType via freeze_mapping).
        object.__setattr__(
            self, "recognizer_versions", freeze_mapping(self.recognizer_versions)
        )

    @property
    def sha(self) -> str:
        """Deterministic SHA-256 of the canonical JSON representation."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "platform_id": self.platform_id,
            "layout_id": self.layout_id,
            "card_layout_version": self.card_layout_version,
            "template_set_version": self.template_set_version,
            "calibration_version": self.calibration_version,
            "recognizer_versions": dict(self.recognizer_versions),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping) -> "VisionAssetManifest":
        _validate_schema(data)
        return cls(
            platform_id=data["platform_id"],
            layout_id=data["layout_id"],
            card_layout_version=data["card_layout_version"],
            template_set_version=data["template_set_version"],
            calibration_version=data["calibration_version"],
            recognizer_versions=dict(data["recognizer_versions"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "VisionAssetManifest":
        return cls.from_dict(json.loads(text))


__all__ = ["VisionAssetManifest"]
