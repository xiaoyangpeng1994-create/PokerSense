"""Current native-glyph gate for AA8 temporal/preprocessed card candidates.

The frozen reader and its heads remain unchanged. A current native crop must
support the same identity as the temporal candidate before it is exposed. This
is a rejection check, not an independent classifier or proof of visual accuracy.
It never fills a missing temporal value from a single-frame prediction.
"""

from pathlib import Path

from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, FusedSlotBuffer, load_card_heads,
)
from tools.aa8_cards import AA8CardReader


class AA8CardReaderV2(AA8CardReader):
    def __init__(self, heads, *, preprocessing=None):
        if isinstance(heads, (str, Path)):
            heads = load_card_heads(Path(heads))
        super().__init__(heads, preprocessing=preprocessing)
        self._native_model = FusedCardRecognizer(
            heads, rank_floor=.5, suit_floor=.3)

    def _native_current(self, image, rect):
        if rect is None:
            return None
        x, y, width, height = rect
        crop = image[y:y + height, x:x + width]
        buffer = FusedSlotBuffer((0, 0, width, height))
        if not buffer.ingest(crop):
            return None
        glyphs = buffer.latest_glyphs()
        if glyphs is None:
            return None
        read = self._native_model.recognize_fused(*glyphs)
        return str(read.value[0]) if read.value else None

    def read(self, image, frame, pts, source):
        result = super().read(image, frame, pts, source)
        prior = list(result['board_slots'])
        checks = []
        for slot, candidate in enumerate(prior):
            rect = result['evidence'].get(f'board_{slot}', {}).get('rect')
            current = self._native_current(image, rect)
            supported = candidate is not None and current == candidate
            reason = ('native_current_and_temporal_agree' if supported else
                      'temporal_candidate_unknown' if candidate is None else
                      'native_current_unknown' if current is None else
                      'native_current_conflicts_with_temporal')
            if not supported:
                result['board_slots'][slot] = None
            checks.append({'native_current': current,
                           'temporal_candidate': candidate,
                           'supported': supported, 'reason': reason})
        result['board_qualification_v2'] = {
            'schema_version': 1, 'unqualified_temporal_board': prior,
            'checks': checks, 'candidate_only': True,
            'independent_accuracy_verified': False,
        }
        return result
