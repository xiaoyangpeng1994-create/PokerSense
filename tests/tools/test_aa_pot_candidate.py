import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayRead
from tools.aa_pot_candidate import AAPotCandidate, colon_x, prefix_score


class Bank:
    def diagnose(self, patch):
        assert patch.size > 0
        return GrayRead("30", "30", "candidate", ())


class Scene:
    def recognize(self, image):
        return {"scene": "AA_TABLE_CANDIDATE"}


def reference():
    image = np.zeros((1080, 498, 3), dtype=np.uint8)
    patch = image[279:300, 198:302]
    patch[2:17, 19:30] = 255
    patch[2:17, 36:47] = 255
    patch[7:9, 52:54] = 255
    patch[13:15, 52:54] = 255
    patch[3:15, 59:66] = 255
    return image


def test_colon_requires_two_aligned_dots_and_unique_column():
    mask = np.zeros((21, 104), np.uint8)
    mask[7:9, 52:54] = 255
    assert colon_x(mask) is None
    mask[13:15, 52:54] = 255
    assert colon_x(mask) == 52
    mask[7:9, 70:72] = 255
    mask[13:15, 70:72] = 255
    assert colon_x(mask) is None


def test_prefix_required_before_number_is_passed_to_bank():
    image = reference()
    reader = AAPotCandidate(image, Bank(), Scene())
    assert reader.recognize(image)["value"] == "30"
    image[279:300, 198:248] = 0
    assert reader.recognize(image)["value"] is None
    assert reader.recognize(None)["value"] is None


def test_prefix_translation_is_bounded():
    pattern = np.zeros((21, 36), np.uint8)
    pattern[3:16, 5:10] = 255
    pattern[4:12, 22:30] = 255
    shifted = np.roll(pattern, 1, axis=1)
    assert prefix_score(shifted, pattern) > .95


def test_one_blank_column_between_colon_and_digit_is_not_clipping():
    image = reference()
    reader = AAPotCandidate(image, Bank(), Scene())
    image[279:300, 257:264] = 0
    image[282:294, 253:260] = 255
    assert reader.recognize(image)["value"] == "30"
    image[282:294, 252] = 255
    assert reader.recognize(image)["value"] is None
