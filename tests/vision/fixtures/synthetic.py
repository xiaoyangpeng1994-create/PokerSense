"""Deterministic synthetic frame fixtures for Vision recognizer tests.

Generates simple, fixed-font card/digit images with known ground truth so
recognizers can be unit-tested offline (no real platform screenshots).

All generation is deterministic (fixed seed, fixed layout). These fixtures are
for DEVELOPMENT / unit testing only; they do NOT count toward Frozen
acceptance (which requires a Real Platform Golden Test Set).
"""

from __future__ import annotations

import numpy as np

# Rank -> single-character label for rendering (10 -> "T").
_RANK_LABELS = {
    "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
    "9": "9", "8": "8", "7": "7", "6": "6", "5": "5",
    "4": "4", "3": "3", "2": "2",
}
# Suit -> unicode-ish symbol (sprites are drawn simply; tests match on structure).
_SUIT_SYMBOLS = {
    "S": "S", "H": "H", "D": "D", "C": "C",
}


def render_card(rank: str, suit: str, width: int = 60, height: int = 84) -> np.ndarray:
    """Render a single card as a BGR uint8 image (white card, black rank/suit)."""
    import cv2

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    label = _RANK_LABELS.get(rank, "?")
    suit_sym = _SUIT_SYMBOLS.get(suit, "?")
    cv2.putText(img, label, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, suit_sym, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def render_digit(text: str, width: int = 40, height: int = 24) -> np.ndarray:
    """Render a short numeric string (e.g. amount) as a BGR uint8 image."""
    import cv2

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def render_number(text: str, char_w: int = 26, height: int = 30) -> np.ndarray:
    """Render a multi-char number with per-character spacing (for segmentation)."""
    import cv2

    width = char_w * len(text)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, ch in enumerate(text):
        x = i * char_w + 2
        cv2.putText(img, ch, (x, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def render_char(ch: str, height: int = 30) -> np.ndarray:
    """Render a single character tightly cropped to its ink (for templates)."""
    import cv2

    canvas = np.full((height, 40, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, ch, (2, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return canvas
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return canvas[y0:y1, x0:x1]


def render_empty_slot(width: int = 60, height: int = 84) -> np.ndarray:
    """Render an empty (no card) board slot as a uniform dark background."""
    return np.full((height, width, 3), 40, dtype=np.uint8)


def build_board_slots(cards: tuple[str | None, ...]) -> tuple[np.ndarray, ...]:
    """Build exactly 5 board slot images (None = EMPTY)."""
    out = []
    for c in cards:
        if c is None:
            out.append(render_empty_slot())
        else:
            rank, suit = c[0], c[1]
            out.append(render_card(rank, suit))
    return tuple(out)


def build_hero_slots(cards: tuple[str | None, ...]) -> tuple[np.ndarray, ...]:
    """Build exactly 2 hero slot images (None = EMPTY)."""
    return tuple(render_card(c[0], c[1]) if c else render_empty_slot() for c in cards)


__all__ = [
    "render_card",
    "render_digit",
    "render_number",
    "render_empty_slot",
    "build_board_slots",
    "build_hero_slots",
]
