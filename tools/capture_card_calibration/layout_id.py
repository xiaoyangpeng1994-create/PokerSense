"""``layout_id`` construction and validation (guide section 6).

Section 6 requires:

    phone_<model>__card_<card>__uvc_<w>x<h>_<fps>__canvas_<w>x<h>__v1

slugified to lowercase ASCII with underscores, and:

    Any setting change that affects pixel coordinates must produce a new
    layout_id.

Baking the geometry into the identifier is what makes that rule enforceable:
two captures with different UVC sizes or canvas sizes simply cannot share a
layout, so ROIs can never be silently reused across incompatible inputs.
"""

from __future__ import annotations

import re

_SLUG_RUNS = re.compile(r"_+")
_VALID_LAYOUT_ID = re.compile(
    r"\Aphone_[a-z0-9_]+__card_[a-z0-9_]+"
    r"__uvc_[0-9]+x[0-9]+_[0-9_]+__canvas_[0-9]+x[0-9]+__v[0-9]+\Z"
)


def slugify(value: str) -> str:
    """Lowercase, ASCII-fold and underscore a free-text hardware name.

    Non-ASCII characters (Chinese model names, for example) collapse to
    underscores, and repeated separators are squeezed so the result is stable
    regardless of spacing or punctuation in the source string.
    """
    characters: list[str] = []
    for char in str(value).strip().lower():
        if char.isascii() and char.isalnum():
            characters.append(char)
        else:
            characters.append("_")
    return _SLUG_RUNS.sub("_", "".join(characters)).strip("_")


def slugify_number(value: float | int) -> str:
    """Render a number as a slug; ``29.97`` becomes ``29_97``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be a number")
    return slugify(f"{value:g}")


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def build_layout_id(
    *,
    phone_model: str,
    capture_card_model: str,
    uvc_width: int,
    uvc_height: int,
    fps: float | int,
    canvas_width: int,
    canvas_height: int,
    version: int = 1,
) -> str:
    """Assemble the section 6 layout identifier."""
    phone = slugify(phone_model)
    card = slugify(capture_card_model)
    if not phone:
        raise ValueError("phone_model must contain at least one ASCII alnum")
    if not card:
        raise ValueError("capture_card_model must contain at least one alnum")
    _require_positive_int("uvc_width", uvc_width)
    _require_positive_int("uvc_height", uvc_height)
    _require_positive_int("canvas_width", canvas_width)
    _require_positive_int("canvas_height", canvas_height)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"fps must be a positive number, got {fps!r}")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive int")
    return (
        f"phone_{phone}__card_{card}"
        f"__uvc_{uvc_width}x{uvc_height}_{slugify_number(fps)}"
        f"__canvas_{canvas_width}x{canvas_height}__v{version}"
    )


def is_valid_layout_id(value: str) -> bool:
    """Return True when ``value`` matches the section 6 pattern."""
    return bool(isinstance(value, str) and _VALID_LAYOUT_ID.match(value))


__all__ = [
    "build_layout_id",
    "is_valid_layout_id",
    "slugify",
    "slugify_number",
]
