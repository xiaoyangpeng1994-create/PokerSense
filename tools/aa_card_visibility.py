"""Conservative AA face-background and raised-card crop checks, offline only."""

import numpy as np


def neutral_background(patch):
    pixels = patch.astype(np.int16)
    return ((pixels.max(axis=2) - pixels.min(axis=2) <= 45)
            & (pixels.min(axis=2) >= 70))


def face_card_support(image, rect):
    if (not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3)
            or image.dtype != np.uint8):
        return False, "unsupported_canvas"
    x, y, width, height = rect
    if (any(type(v) is not int for v in rect) or x < 0 or y < 6
            or width < 12 or height < 12 or x + width > 498 or y + height > 1080):
        return False, "invalid_card_rect"
    patch = image[y + 4:y + height - 4, x + 4:x + width - 4]
    if float(neutral_background(patch).mean()) < .20:
        return False, "no_positive_face_background"
    # A face continuing above the crop means the rank may already be clipped.
    # In AA settlement animation, selected board cards rise independently.
    above = image[y - 6:y - 2, x + width // 2:x + width - 5]
    if float(neutral_background(above).mean()) > .35:
        return False, "face_extends_above_fixed_crop"
    return True, "face_background_and_top_boundary_candidate"


def locate_face_card(image, rect, *, max_vertical_shift=20):
    """Find a unique neutral top edge, never choose by a predicted rank/suit."""
    if (not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3)
            or image.dtype != np.uint8):
        return None, "unsupported_canvas"
    if type(max_vertical_shift) is not int or not 0 <= max_vertical_shift <= 30:
        raise ValueError("bounded vertical search required")
    if len(rect) != 4 or any(type(v) is not int for v in rect):
        return None, "invalid_card_rect"
    x, nominal, width, height = rect
    if x < 0 or width < 20 or x + width > 498 or height < 20:
        return None, "invalid_card_rect"
    supported, reason = face_card_support(image, rect)
    if supported:
        return tuple(rect), "nominal_face_boundary_supported"
    if reason != "face_extends_above_fixed_crop":
        return None, reason
    candidates = []
    for y in range(max(6, nominal - max_vertical_shift),
                   min(1080 - height, nominal + max_vertical_shift) + 1):
        band = image[y - 3:y + 3, x + width // 2:x + width - 5]
        fraction = neutral_background(band).mean(axis=1)
        if fraction[:3].max() > .25 or fraction[3:].min() < .75:
            continue
        candidate = (x, y, width, height)
        if face_card_support(image, candidate)[0]:
            candidates.append(candidate)
    if len(candidates) != 1:
        return None, "no_unique_face_top_edge"
    return candidates[0], "measured_face_top_edge_candidate"
