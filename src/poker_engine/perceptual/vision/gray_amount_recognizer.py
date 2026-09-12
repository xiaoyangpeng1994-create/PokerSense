"""Source-trained gray digit bank, offline until independent acceptance.

Each digit competes against every other label; no arithmetic correction and no
reuse of previously seen values. Original gray strokes, not averaged templates.
"""

from dataclasses import dataclass
from decimal import Decimal
import math

import cv2
import numpy as np

from poker_engine.perceptual.vision.amount_recognizer import _column_spans
from poker_engine.perceptual.vision.protocols import AmountRecognition
from poker_engine.core.value_objects import ChipAmount


def gray_glyphs(patch):
    if patch is None or patch.size == 0:
        return [], "empty"
    if (patch.dtype != np.uint8 or patch.ndim not in (2, 3)
            or patch.ndim == 3 and patch.shape[2] != 3):
        return [], "unsupported_image"
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (binary[0], binary[-1], binary[:, 0], binary[:, -1])
    if any(np.any(edge) for edge in edges):
        return [], "clipped_or_border"
    spans = _column_spans(binary)
    if not 1 <= len(spans) <= 8:
        return [], "glyph_count"
    background = float(np.percentile(gray, 20))
    result = []
    for left, right in spans:
        ys, xs = np.where(binary[:, left:right] > 0)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        glyph = gray[top:bottom, left:right]
        height, width = glyph.shape
        if height < 7 or width > 16:
            return [], "unsupported_punctuation_or_merge"
        contrast = float(np.percentile(glyph, 95)) - background
        if contrast < 20:
            return [], "low_contrast"
        ink = np.clip((glyph.astype(np.float32) - background) / contrast, 0, 1)
        scale = min(24 / height, 24 / width)
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        feature = np.zeros((28, 28), np.float32)
        y, x = (28 - h) // 2, (28 - w) // 2
        resized = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)
        feature[y:y + h, x:x + w] = resized
        feature = cv2.GaussianBlur(feature, (3, 3), .5)
        result.append((feature, (left, top, right - left, bottom - top)))
    return result, None


def unit_vectors(features):
    values = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
    values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("degenerate digit feature")
    return values / norms


@dataclass(frozen=True)
class GrayRead:
    value: str | None
    raw_text: str | None
    reason: str
    scores: tuple[tuple[str, float, float], ...]


class GrayAmountRecognizer:
    def __init__(self, features, labels, floor=.90, margin=.05, *, augment=False):
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        if (features.ndim != 3 or features.shape[1:] != (28, 28)
                or not np.isfinite(features).all()
                or np.any(features < 0) or np.any(features > 1)):
            raise ValueError("invalid gray feature bank")
        if set(labels.tolist()) != set("0123456789") or len(features) != len(labels):
            raise ValueError("complete source-reviewed ten-digit bank required")
        if any(isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 1
               for v in (floor, margin)):
            raise ValueError("invalid recognition gates")
        if augment:
            expanded, expanded_labels = [], []
            for feature, label in zip(features, labels):
                for sx in (.92, 1., 1.08):
                    for dx, dy in ((0., 0.), (-.5, 0.), (.5, 0.), (0., -.5), (0., .5)):
                        matrix = np.array([[sx, 0, (1 - sx) * 13.5 + dx],
                                           [0, 1, dy]], np.float32)
                        expanded.append(cv2.warpAffine(feature, matrix, (28, 28)))
                        expanded_labels.append(label)
            features, labels = np.array(expanded), np.array(expanded_labels)
        self.vectors = unit_vectors(features)
        self.vectors.setflags(write=False)
        self.labels = labels.copy()
        self.labels.setflags(write=False)
        self.floor, self.margin = floor, margin

    def diagnose(self, patch):
        glyphs, reason = gray_glyphs(patch)
        if reason:
            return GrayRead(None, None, reason, ())
        similarities = unit_vectors([f for f, _ in glyphs]) @ self.vectors.T
        records = []
        for row in similarities:
            per_label = [(label, float(np.clip(row[self.labels == label].max(), 0, 1)))
                         for label in "0123456789"]
            ranked = sorted(per_label, key=lambda x: (-x[1], x[0]))
            records.append((ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1]))
        text = "".join(r[0] for r in records)
        accepted = all(score >= self.floor and gap >= self.margin
                       for _, score, gap in records)
        if len(text) > 1 and text.startswith("0"):
            accepted = False
        return GrayRead(text if accepted else None, text,
                        "candidate" if accepted else "ambiguous", tuple(records))

    def recognize(self, patch):
        read = self.diagnose(patch)
        score = min((s for _, s, _ in read.scores), default=0.)
        amount = ChipAmount(Decimal(read.value)) if read.value is not None else None
        return AmountRecognition(amount, score)
