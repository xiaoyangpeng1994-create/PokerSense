"""Deterministic capture-card frame normalization (calibration guide stage C).

A USB capture card streams whatever the UVC device reports — typically a
landscape 1920x1080 frame in which a portrait phone screen is letterboxed or
rotated. Recognition must never run on that raw frame directly; it must run on
a fixed, reproducible "game canvas".

The processing order is fixed and matches the calibration guide:

    decode -> rotate -> mirror -> crop -> content-size validation

The transform is a pure function of ``(image, NormalizationConfig)``: the same
input always yields the same output, and the config is versioned so a change in
rotation / crop / output size is a *new* config version, never an in-place
drift of ROIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .base import CaptureError

_ALLOWED_ROTATIONS = (0, 90, 180, 270)
_ALLOWED_COLOR_TRANSFORMS = ("none",)


@dataclass(frozen=True)
class NormalizationConfig:
    """Versioned capture-card frame normalization (guide stage C schema).

    Fields mirror ``normalization/normalization.json`` from the calibration
    guide:

    - ``source_size``: optional expected raw UVC size ``(width, height)``; when
      set, an input frame whose size differs is rejected (fail closed).
    - ``rotate_degrees``: one of 0/90/180/270 (counter-clockwise, as ``cv2``).
    - ``mirror_horizontal``: flip left-right after rotation.
    - ``crop_after_rotation``: optional ``[x0, y0, x1, y1)`` half-open crop
      applied *after* rotation, in rotated-frame pixels. ``None`` means no crop.
    - ``output_size``: optional expected ``(width, height)`` after crop; when
      set, a mismatch after the full transform is rejected.
    - ``color_transform``: reserved; only ``"none"`` is accepted today.
    - ``version``: a human/robot-readable version string, e.g.
      ``capture-card-normalization-v1``.
    """

    rotate_degrees: int
    mirror_horizontal: bool = False
    crop_after_rotation: tuple[int, int, int, int] | None = None
    output_size: tuple[int, int] | None = None
    source_size: tuple[int, int] | None = None
    color_transform: str = "none"
    version: str = "capture-card-normalization-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.rotate_degrees, bool) or not isinstance(
            self.rotate_degrees, int
        ):
            raise TypeError("rotate_degrees must be an int")
        if self.rotate_degrees not in _ALLOWED_ROTATIONS:
            raise ValueError(
                f"rotate_degrees must be one of {_ALLOWED_ROTATIONS}"
            )
        if not isinstance(self.mirror_horizontal, bool):
            raise TypeError("mirror_horizontal must be a bool")
        if self.crop_after_rotation is not None:
            x0, y0, x1, y1 = self.crop_after_rotation
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    "crop_after_rotation must be a non-empty [x0,y0,x1,y1) "
                    "with x1>x0 and y1>y0"
                )
            object.__setattr__(
                self, "crop_after_rotation", (int(x0), int(y0), int(x1), int(y1))
            )
        if self.output_size is not None:
            object.__setattr__(self, "output_size", _validate_size(self.output_size))
        if self.source_size is not None:
            object.__setattr__(self, "source_size", _validate_size(self.source_size))
        if self.color_transform not in _ALLOWED_COLOR_TRANSFORMS:
            raise ValueError(
                f"color_transform must be one of {_ALLOWED_COLOR_TRANSFORMS}"
            )
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be an int")

    # --- serialization ---

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "rotate_degrees": self.rotate_degrees,
            "mirror_horizontal": self.mirror_horizontal,
            "color_transform": self.color_transform,
            "version": self.version,
        }
        if self.source_size is not None:
            data["source_size"] = list(self.source_size)
        if self.crop_after_rotation is not None:
            data["crop_after_rotation"] = list(self.crop_after_rotation)
        if self.output_size is not None:
            data["output_size"] = list(self.output_size)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationConfig":
        if not isinstance(data, Mapping):
            raise TypeError("data must be a Mapping")
        version = data.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("schema_version must be an int")
        if version != 1:
            raise ValueError(f"unsupported schema_version {version!r}")

        def _opt_size(key: str) -> tuple[int, int] | None:
            raw = data.get(key)
            return _validate_size(raw) if raw is not None else None

        def _opt_crop(key: str) -> tuple[int, int, int, int] | None:
            raw = data.get(key)
            if raw is None:
                return None
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 4
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    for v in raw
                )
            ):
                raise TypeError(f"{key} must be [x0, y0, x1, y1]")
            return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))

        return cls(
            rotate_degrees=data["rotate_degrees"],
            mirror_horizontal=bool(data.get("mirror_horizontal", False)),
            crop_after_rotation=_opt_crop("crop_after_rotation"),
            output_size=_opt_size("output_size"),
            source_size=_opt_size("source_size"),
            color_transform=data.get("color_transform", "none"),
            version=data.get("version", "capture-card-normalization-v1"),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, text: str) -> "NormalizationConfig":
        return cls.from_dict(json.loads(text))


def _validate_size(size: Any) -> tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise TypeError("size must be (width, height)")
    w, h = size
    if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
        raise ValueError("size width must be a positive int")
    if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
        raise ValueError("size height must be a positive int")
    return (int(w), int(h))


def normalize(image: np.ndarray, config: NormalizationConfig) -> np.ndarray:
    """Apply the fixed transform order: rotate -> mirror -> crop -> validate.

    Raises :class:`CaptureError` on a content-size mismatch (either the raw
    source size or the post-transform output size) so the pipeline fails closed
    rather than recognizing against shifted ROIs.
    """
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise CaptureError("normalization input must be at least 2-D")
    if config.source_size is not None:
        actual = (int(arr.shape[1]), int(arr.shape[0]))
        if actual != config.source_size:
            raise CaptureError(
                f"capture-card source size {actual} != expected "
                f"{config.source_size}"
            )

    out = arr
    if config.rotate_degrees == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif config.rotate_degrees == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif config.rotate_degrees == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

    if config.mirror_horizontal:
        out = np.fliplr(out)

    if config.crop_after_rotation is not None:
        x0, y0, x1, y1 = config.crop_after_rotation
        h, w = out.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            raise CaptureError(
                f"crop_after_rotation {config.crop_after_rotation} exceeds "
                f"rotated frame {w}x{h}"
            )
        out = out[y0:y1, x0:x1]

    if config.output_size is not None:
        actual = (int(out.shape[1]), int(out.shape[0]))
        if actual != config.output_size:
            raise CaptureError(
                f"normalized size {actual} != expected {config.output_size}"
            )

    return out


__all__ = ["NormalizationConfig", "normalize"]
