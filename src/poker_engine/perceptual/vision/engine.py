"""Vision Engine — orchestration only.

Consumes Frame + TableMap (+ Vision asset config), produces a Frozen Core
``RawObservation``. The engine delegates to detectors/recognizers/calibrators;
it does NOT implement recognition algorithms and never leaks OpenCV/Paddle
objects into Core contracts.

Confidence pipeline (plan §8): raw score -> per-detector calibrator ->
confidence [0,1]; a detector's ``abstain_floor`` (NOT a Task 5 threshold)
turns unreadable/out-of-domain inputs into UNKNOWN.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.enums import Street
from poker_engine.core.value_objects import Card

from ..capture.base import Frame
from .asset_manifest import VisionAssetManifest
from .calibration import ConfidenceCalibrator
from .card_layout import BoardSlotLayout, HeroSlotLayout
from .errors import TableMapError
from .protocols import (
    BoardSlotDetector,
    BoardSlotOccupancy,
    BoardSlotsRecognition,
    CardRecognizer,
    AmountRecognizer,
    ActionRecognizer,
    StreetDetector,
)
from .roi import extract_roi
from .street_detector import board_card_count
from .table_map import ROIKind, ROI, TableMap
from .trace import RecognitionTrace

# Required calibrator keys (each detector has a versioned calibrator).
_REQUIRED_CALIBRATORS = (
    "card", "amount", "stack", "dealer", "action", "street", "board",
)
# Required recognizer-version keys in the manifest.
_REQUIRED_VERSIONS = (
    "card", "amount", "stack", "dealer", "action", "street", "board",
)
# Allowed scalar bet_size semantic declarations.
_ALLOWED_BET_SEMANTICS = ("single", "global", "hero")


def _find_roi(
    table_map: TableMap, kind: ROIKind, slot_id: int | None = None
) -> ROI | None:
    for roi in table_map.rois:
        if roi.kind is kind and roi.slot_id == slot_id:
            return roi
    return None


def _global_roi(table_map: TableMap, kind: ROIKind) -> ROI | None:
    return _find_roi(table_map, kind, None)


def _card_key(card: Card) -> tuple:
    return (card.rank, card.suit)


def _demote(field: ObservationField) -> ObservationField:
    """Return a deterministic copy of ``field`` demoted to CONFLICT.

    Both ``validation_status`` AND ``evidence["validation_status"]`` are
    updated so the field and its RecognitionTrace summary always agree.
    """
    import dataclasses

    new_ev = dict(field.evidence)
    new_ev["validation_status"] = ValidationStatus.CONFLICT.value
    return dataclasses.replace(
        field,
        validation_status=ValidationStatus.CONFLICT,
        evidence=new_ev,
    )


def find_duplicate_card(cards: tuple[Card, ...]) -> Card | None:
    """Return the first duplicate card (same rank+suit) in ``cards``, else None."""
    seen: set[tuple] = set()
    for c in cards:
        k = _card_key(c)
        if k in seen:
            return c
        seen.add(k)
    return None


def _conflict_between(left: tuple[Card, ...], right: tuple[Card, ...]) -> bool:
    """True if any card appears in BOTH left and right (same rank+suit)."""
    right_keys = {_card_key(c) for c in right}
    return any(_card_key(c) in right_keys for c in left)


def _locate_bottom_hero_cards(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Find the two bright card faces at the bottom-centre of a table frame.

    WePoker may be captured with browser chrome included or in its fullscreen
    table mode, which shifts the fixed calibrated ROI vertically. This narrow
    fallback only considers a matching pair of white card faces in the lower
    centre; it does not attempt generic screen OCR.
    """
    if image is None or image.size == 0:
        return ()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, int, int, int]] = []
    for x, y, card_w, card_h, area in stats[1:]:
        ratio = card_h / card_w if card_w else 0.0
        if (
            y >= int(height * 0.65)
            and int(width * 0.25) <= x <= int(width * 0.75)
            and card_w >= int(width * 0.01)
            and card_h >= int(height * 0.04)
            and 1.15 <= ratio <= 2.1
            and area >= card_w * card_h * 0.45
        ):
            candidates.append((int(x), int(y), int(card_w), int(card_h)))

    pairs: list[tuple[float, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            left, right = sorted((first, second), key=lambda item: item[0])
            gap = right[0] - (left[0] + left[2])
            aligned = abs(left[1] - right[1]) <= int(height * 0.025)
            similar = abs(left[2] - right[2]) <= int(width * 0.02)
            if aligned and similar and 0 <= gap <= int(width * 0.08):
                centre = (left[0] + right[0] + right[2]) / 2
                score = abs(centre - width / 2) + abs(left[1] - right[1])
                pairs.append((score, left, right))
    if not pairs:
        return ()
    _, left, right = min(pairs, key=lambda item: item[0])
    return tuple(image[y:y + h, x:x + w] for x, y, w, h in (left, right))


class VisionEngine:
    """Orchestrates Frame+TableMap -> RawObservation."""

    def __init__(
        self,
        board_layout: BoardSlotLayout,
        hero_layout: HeroSlotLayout,
        card_recognizer: CardRecognizer,
        board_slot_detector: BoardSlotDetector,
        street_detector: StreetDetector,
        amount_recognizer: AmountRecognizer,
        stack_recognizer: AmountRecognizer,
        action_recognizer: ActionRecognizer,
        calibrators: Mapping[str, ConfidenceCalibrator],
        manifest: VisionAssetManifest,
        require_action: bool = False,
        bet_size_semantics: str | None = None,
        dealer_recognizer=None,
        empty_slot_recognizer=None,
        actor_recognizer=None,
        disable_hero_dynamic_fallback: bool = False,
    ) -> None:
        self._board_layout = board_layout
        self._hero_layout = hero_layout
        self._card = card_recognizer
        self._board_detector = board_slot_detector
        self._street_detector = street_detector
        self._amount = amount_recognizer
        self._stack_amount = stack_recognizer
        self._action = action_recognizer
        self._dealer = dealer_recognizer
        self._empty_slot = empty_slot_recognizer
        self._actor = actor_recognizer
        self._calibrators = dict(calibrators)
        self._manifest = manifest
        self._require_action = require_action
        self._no_hero_dynamic = disable_hero_dynamic_fallback

        # Validate the explicit scalar bet_size semantic declaration (plan §10).
        if bet_size_semantics is not None:
            if bet_size_semantics not in _ALLOWED_BET_SEMANTICS:
                raise TableMapError(
                    f"unsupported bet_size_semantics {bet_size_semantics!r}; "
                    f"allowed: {_ALLOWED_BET_SEMANTICS}"
                )
        self._bet_size_semantics = bet_size_semantics

        missing = [k for k in _REQUIRED_CALIBRATORS if k not in self._calibrators]
        if missing:
            raise TableMapError(f"missing calibrator(s): {sorted(missing)}")
        if self._empty_slot is not None and "occupancy" not in self._calibrators:
            raise TableMapError("empty-slot recognizer requires occupancy calibrator")

    # ------------------------------------------------------------------ utils

    def _cal(self, name: str) -> ConfidenceCalibrator:
        return self._calibrators[name]

    def _src(self, name: str) -> str:
        ver = self._manifest.recognizer_versions.get(name, "?")
        return f"{name}:v{ver}"

    def _validate_manifest(self, table_map: TableMap) -> None:
        m = self._manifest
        if m.platform_id != table_map.platform_id:
            raise TableMapError(
                f"manifest platform_id {m.platform_id!r} != table_map "
                f"{table_map.platform_id!r}"
            )
        if m.layout_id != table_map.layout_id:
            raise TableMapError(
                f"manifest layout_id {m.layout_id!r} != table_map "
                f"{table_map.layout_id!r}"
            )
        if m.card_layout_version != self._board_layout.version:
            raise TableMapError(
                f"manifest card_layout_version {m.card_layout_version} != "
                f"board layout version {self._board_layout.version}"
            )
        for name in _REQUIRED_VERSIONS:
            if name not in m.recognizer_versions:
                raise TableMapError(
                    f"manifest missing recognizer version for {name!r}"
                )

    def _evidence(
        self,
        frame_seq: int,
        roi_key: str,
        slot_id: int | None,
        name: str,
        raw_score: float,
        confidence: float,
        status: ValidationStatus,
        components: Mapping[str, float] | None = None,
    ) -> RecognitionTrace:
        """Build a RecognitionTrace (the single source of field evidence)."""
        return RecognitionTrace(
            frame_seq=frame_seq,
            roi_key=roi_key,
            slot_id=slot_id,
            recognizer_name=name,
            recognizer_version=self._manifest.recognizer_versions.get(name, "?"),
            raw_score=raw_score,
            confidence=confidence,
            validation_status=status,
            manifest_sha=self._manifest.sha,
            template_config_version=self._manifest.template_set_version,
            components=components,
        )

    def _field(
        self,
        value,
        confidence: float,
        status: ValidationStatus,
        source: str,
        evidence: RecognitionTrace | dict,
        timestamp,
    ) -> ObservationField:
        if isinstance(evidence, RecognitionTrace):
            evidence = evidence.to_dict()
        else:
            evidence = dict(evidence)
        return ObservationField(
            value=value,
            confidence=confidence,
            source=source,
            evidence=evidence,
            timestamp=timestamp,
            validation_status=status,
        )

    def _unknown_field(
        self,
        frame_seq: int,
        roi_key: str,
        slot_id: int | None,
        name: str,
        timestamp,
        value=None,
    ) -> ObservationField:
        """Deterministic UNKNOWN field with a real RecognitionTrace summary.

        Used when NO recognizer actually ran (missing ROI / disabled signal).
        The trace records the UNKNOWN reason/source — it is NOT a fabricated
        successful recognition.
        """
        trace = RecognitionTrace(
            frame_seq=frame_seq,
            roi_key=roi_key,
            slot_id=slot_id,
            recognizer_name=name,
            recognizer_version=self._manifest.recognizer_versions.get(name, "?"),
            raw_score=0.0,
            confidence=0.0,
            validation_status=ValidationStatus.UNKNOWN,
            manifest_sha=self._manifest.sha,
            template_config_version=self._manifest.template_set_version,
        )
        source = name if name == "none" else self._src(name)
        return self._field(
            value,
            0.0,
            ValidationStatus.UNKNOWN,
            source,
            trace,
            timestamp,
        )

    # ----------------------------------------------------------------- process

    def process(self, frame: Frame, table_map: TableMap) -> RawObservation:
        ts = frame.timestamp
        self._validate_manifest(table_map)

        # Bet Semantic Gap (plan §10): scalar bet_size is only written when the
        # Vision config EXPLICITLY declares an allowed scalar semantic
        # (single/global/hero) AND exactly one compatible global ROI exists.
        # 0 ROIs -> bet_size UNKNOWN (not written); >1 or per-seat -> STOP.
        bet_rois = [r for r in table_map.rois if r.kind is ROIKind.BET_SIZE]
        if self._bet_size_semantics is None:
            # No explicit scalar semantic declared -> bet_size is NOT written.
            bet_rois = []
        elif len(bet_rois) > 1 or (
            bet_rois and bet_rois[0].slot_id is not None
        ):
            raise TableMapError(
                "Bet Semantic Gap: scalar bet_size requires exactly one "
                "global BET_SIZE ROI per declared semantics "
                f"{self._bet_size_semantics!r}"
            )

        # Platform Visual Gap (plan §9): if action recognition is required but
        # the table exposes no per-seat ACTION signal, fail explicitly instead
        # of silently returning () for slot_actions.
        if self._require_action:
            if not any(r.kind is ROIKind.ACTION for r in table_map.rois):
                raise TableMapError(
                    "Platform Visual Gap: action recognition required but no "
                    "per-seat ACTION visual signal is present"
                )

        hero_cards = self._recognize_hero(frame, table_map, ts)
        board_cards, street_field = self._recognize_board(frame, table_map, ts)

        # Cross-set visible-card conflict: a card appearing in BOTH hero and
        # board is impossible -> mark hero/board CONFLICT and propagate to
        # street as CONFLICT (no guessing correction).
        if (
            hero_cards.value is not None
            and board_cards.value is not None
            and _conflict_between(hero_cards.value, board_cards.value)
        ):
            hero_cards = _demote(hero_cards)
            board_cards = _demote(board_cards)
            street_field = _demote(street_field)

        pot = self._recognize_amount_roi(frame, table_map, ROIKind.POT, ts)
        if self._bet_size_semantics is None:
            bet_size = self._unknown_field(
                frame.frame_seq, "bet_size", None, "amount", ts
            )
        else:
            bet_size = self._recognize_amount_roi(
                frame, table_map, ROIKind.BET_SIZE, ts
            )
        slot_stacks = self._recognize_stacks(frame, table_map, ts)
        slot_occupancies = self._recognize_occupancies(
            frame, slot_stacks, ts
        )
        slot_actions = self._recognize_actions(frame, table_map, ts)
        dealer_pos = self._recognize_dealer(frame, ts)

        actor = self._recognize_actor(frame, ts)
        return RawObservation(
            frame_seq=frame.frame_seq,
            timestamp=ts,
            hero_cards=hero_cards,
            board_cards=board_cards,
            pot=pot,
            stacks=self._unknown_field(
                frame.frame_seq, "stacks", None, "stack", ts, value=()
            ),
            bet_size=bet_size,
            action=self._unknown_field(
                frame.frame_seq, "action", None, "action", ts
            ),
            street=street_field,
            dealer_pos=dealer_pos,
            actor=actor,
            slot_stacks=slot_stacks,
            slot_actions=slot_actions,
            slot_occupancies=slot_occupancies,
        )

    # ------------------------------------------------------------------ cards

    def _recognize_hero(self, frame, table_map, ts):
        roi = _global_roi(table_map, ROIKind.HERO_CARDS)
        if roi is None:
            return self._unknown_field(
                frame.frame_seq, "hero_cards", None, "card", ts, value=()
            )
        crop = extract_roi(frame, roi)

        static_slots = tuple(_crop_slot(crop, sub) for sub in self._hero_layout.slots)
        cards_tuple, raw_feature = self._recognize_hero_slots(static_slots)
        cal = self._cal("card")
        status = self._hero_status(
            cards_tuple, raw_feature, cal, len(self._hero_layout.slots)
        )

        # Fixed coordinates are preferred when valid. Fall back only when the
        # table has moved because Chrome is fullscreen / its toolbar changed.
        # A stateful recognizer (temporal fusion) must NOT run the dynamic
        # fallback: its per-slot buffer would receive two different card crops
        # for the same slot in one frame, each resetting the other. Platforms
        # with a fixed normalized canvas disable the fallback.
        if status is not ValidationStatus.VALID and not self._no_hero_dynamic:
            dynamic_slots = _locate_bottom_hero_cards(frame.image)
            if len(dynamic_slots) == len(self._hero_layout.slots):
                dynamic_cards, dynamic_raw = self._recognize_hero_slots(dynamic_slots)
                dynamic_status = self._hero_status(
                    dynamic_cards, dynamic_raw, cal, len(self._hero_layout.slots)
                )
                if dynamic_status is ValidationStatus.VALID:
                    cards_tuple, raw_feature, status = (
                        dynamic_cards, dynamic_raw, dynamic_status
                    )

        conf = cal.calibrate(raw_feature).confidence
        evidence = self._evidence(
            frame.frame_seq, "hero_cards", None, "card",
            raw_feature, conf, status,
        )
        return self._field(cards_tuple, conf, status, self._src("card"), evidence, ts)

    def _recognize_hero_slots(self, slots):
        cards: list[Card] = []
        slot_scores: list[float] = []
        for i, sub_img in enumerate(slots):
            # ``card_model`` doubles as the slot identity for the (optional)
            # temporal-fusion card recognizer; single-frame recognizers ignore
            # it. hero slots use ("hero", i) so they never share a buffer with
            # the board's slots.
            rec = self._card.recognize(sub_img, ("hero", i))
            # include rank AND suit: per-card raw feature is min(rank, suit).
            per_card = min(
                (
                    min(s.rank_score, s.suit_score) for s in rec.slots
                ),
                default=0.0,
            )
            slot_scores.append(per_card if rec.value else per_card)
            if rec.value:
                cards.extend(rec.value)
        return tuple(cards), min(slot_scores) if slot_scores else 0.0

    def _hero_status(self, cards, raw_feature, cal, expected_len):
        # Duplicate within hero is a DETERMINISTIC conflict: it must never be
        # downgraded to UNKNOWN by abstain. Check it FIRST.
        if find_duplicate_card(cards) is not None:
            return ValidationStatus.CONFLICT
        # abstain / incomplete -> UNKNOWN (abstain only downgrades VALID)
        if cal.should_abstain(raw_feature):
            return ValidationStatus.UNKNOWN
        if len(cards) < expected_len:
            return ValidationStatus.UNKNOWN
        return ValidationStatus.VALID

    def _recognize_board(self, frame, table_map, ts):
        roi = _global_roi(table_map, ROIKind.BOARD_CARDS)
        if roi is None:
            empty_board = self._unknown_field(
                frame.frame_seq, "board_cards", None, "board", ts, value=()
            )
            empty_street = self._unknown_field(
                frame.frame_seq, "board_cards", None, "street", ts
            )
            return empty_board, empty_street

        crop = extract_roi(frame, roi)

        # OCCUPANCY (independent presence/empty signal) — authoritative for
        # street derivation, does NOT depend on rank/suit recognition.
        slots_rec = self._board_detector.detect(crop)
        street_rec = self._street_detector.derive(slots_rec)

        # CARD IDENTITY (independent rank/suit recognition) — the genuine
        # second signal; board_cards value comes from THIS path, not occupancy.
        independent = self._recognize_board_cards_independently(crop)
        cal_card = self._cal("card")
        board_cards = tuple(
            card for (_, card, score) in independent
            if card is not None and not cal_card.should_abstain(score)
        )

        # Determine the deterministic conflict set (NOT affected by abstain).
        board_conflict, board_unknown = self._board_disagreement(
            slots_rec, independent, street_rec, cal_card,
        )

        # Build board status with priority CONFLICT > UNKNOWN > VALID.
        board_status = self._resolve_board_status(
            board_conflict, board_unknown,
        )

        # board confidence via SEPARATE per-signal calibration (abstain only
        # downgrades VALID->UNKNOWN; it NEVER clears a CONFLICT).
        #
        # Board is TWO signals reconciled at the CALIBRATED level (never a raw
        # min across two different scales):
        #   - occupancy evidence -> board calibrator
        #   - identity recognition -> card calibrator
        # a CARD slot's confidence is the min of the two calibrated values, an
        # EMPTY slot's is its board-calibrated no-card confidence.
        cal_board = self._cal("board")
        board_conf = _board_calibrated_confidence(
            slots_rec, independent, cal_board, cal_card,
        )
        board_components = _board_components(
            slots_rec, independent, cal_board, cal_card
        )
        if board_status is ValidationStatus.CONFLICT:
            pass  # conflict is authoritative; abstain must not override
        elif _board_should_abstain(slots_rec, independent, cal_board, cal_card):
            board_status = ValidationStatus.UNKNOWN
        # trace.raw_score keeps its "raw" semantics: the board's occupancy raw
        # (the presence/empty signal representative). The calibrated confidence
        # lives in `confidence`, and both signals' raw + calibrated values are
        # in `components`. We deliberately do NOT label the calibrated board
        # confidence as raw_score.
        board_ev = self._evidence(
            frame.frame_seq, "board_cards", None, "board",
            board_components["occupancy_app_raw"], board_conf, board_status,
            components=board_components,
        )
        board_field = self._field(
            board_cards, board_conf, board_status, self._src("board"), board_ev, ts
        )

        # street status: deterministic conflict first, then abstain.
        cal_street = self._cal("street")
        street_status = street_rec.status
        if street_status is ValidationStatus.CONFLICT:
            pass
        elif cal_street.should_abstain(street_rec.raw_score):
            street_status = ValidationStatus.UNKNOWN
        street_conf = cal_street.calibrate(street_rec.raw_score).confidence
        street_ev = self._evidence(
            frame.frame_seq, "board_cards", None, "street",
            street_rec.raw_score, street_conf, street_status,
        )
        street_field = self._field(
            street_rec.street, street_conf, street_status,
            self._src("street"), street_ev, ts,
        )

        # Unified propagation (priority CONFLICT > UNKNOWN > VALID): if EITHER
        # board or street is CONFLICT, BOTH must be CONFLICT.
        if (
            board_status is ValidationStatus.CONFLICT
            or street_status is ValidationStatus.CONFLICT
        ):
            if board_status is not ValidationStatus.CONFLICT:
                board_field = _demote(board_field)
            if street_status is not ValidationStatus.CONFLICT:
                street_field = _demote(street_field)

        return board_field, street_field

    def _board_disagreement(self, slots_rec, independent, street_rec, cal_card):
        """Return (conflict: bool, unknown: bool) from deterministic checks."""
        conflict = False
        unknown = any(
            s.occupancy is BoardSlotOccupancy.UNKNOWN for s in slots_rec.slots
        )

        # 1) per-slot occupancy vs identity disagreement — identity is trusted
        #    only when it clears the card calibrator's abstain floor (NOT a
        #    hard-coded threshold).
        for slot_index, card, score in independent:
            slot = slots_rec.slots[slot_index]
            occupancy = slot.occupancy
            has_identity = card is not None and not cal_card.should_abstain(score)
            if occupancy is BoardSlotOccupancy.CARD and not has_identity:
                conflict = True  # presence says CARD, identity absent/failed
            elif occupancy is BoardSlotOccupancy.EMPTY and has_identity:
                conflict = True  # presence says EMPTY, identity found a card

        # 2) deterministic duplicate among recognized board cards
        recognized = tuple(
            card for (_, card, score) in independent
            if card is not None and not cal_card.should_abstain(score)
        )
        if find_duplicate_card(recognized) is not None:
            conflict = True

        # 3) illegal street occupancy pattern already reported by street
        if street_rec.status is ValidationStatus.CONFLICT:
            conflict = True

        # 4) occupancy card count vs street expected count
        expected = _expected_board_count(street_rec.street)
        occ_count = board_card_count(slots_rec)
        if expected is not None and occ_count != expected:
            conflict = True

        return conflict, unknown

    def _resolve_board_status(self, conflict: bool, unknown: bool):
        if conflict:
            return ValidationStatus.CONFLICT
        if unknown:
            return ValidationStatus.UNKNOWN
        return ValidationStatus.VALID

    def _recognize_board_cards_independently(self, crop):
        """Independently recognize each board slot's card IDENTITY (rank/suit).

        Returns per-slot (slot_index, card, raw_score) WITHOUT any occupancy
        logic; this is the genuine second signal for the consistency check.
        """
        out: list[tuple[int, Card | None, float]] = []
        for i, sub in enumerate(self._board_layout.slots):
            sub_img = _crop_slot(crop, sub)
            # slot identity ("board", i) feeds the temporal-fusion recognizer;
            # single-frame recognizers ignore it.
            rec = self._card.recognize(sub_img, ("board", i))
            card = rec.value[0] if rec.value is not None else None
            out.append((i, card, rec.raw_score))
        return out

    # ----------------------------------------------------------------- amount

    def _recognize_amount_roi(self, frame, table_map, kind, ts):
        roi = _global_roi(table_map, kind)
        if roi is None:
            return self._unknown_field(
                frame.frame_seq, kind.value, None, "amount", ts
            )
        crop = extract_roi(frame, roi)
        rec = self._amount.recognize(crop)
        cal = self._cal("amount")
        if cal.should_abstain(rec.raw_score) or rec.value is None:
            conf = cal.calibrate(rec.raw_score).confidence
            ev = self._evidence(
                frame.frame_seq, kind.value, None, "amount",
                rec.raw_score, conf, ValidationStatus.UNKNOWN,
            )
            return self._field(
                None, conf, ValidationStatus.UNKNOWN, self._src("amount"), ev, ts
            )
        conf = cal.calibrate(rec.raw_score).confidence
        ev = self._evidence(
            frame.frame_seq, kind.value, None, "amount",
            rec.raw_score, conf, ValidationStatus.VALID,
        )
        return self._field(
            rec.value, conf, ValidationStatus.VALID, self._src("amount"), ev, ts
        )

    def _recognize_stacks(self, frame, table_map, ts):
        out = []
        stack_rois = sorted(
            (r for r in table_map.rois if r.kind is ROIKind.STACK),
            key=lambda r: r.slot_id,
        )
        for roi in stack_rois:
            crop = extract_roi(frame, roi)
            rec = self._stack_amount.recognize(crop)
            cal = self._cal("stack")
            if cal.should_abstain(rec.raw_score) or rec.value is None:
                conf = cal.calibrate(rec.raw_score).confidence
                status = ValidationStatus.UNKNOWN
                field_val = None
            else:
                conf = cal.calibrate(rec.raw_score).confidence
                status = ValidationStatus.VALID
                field_val = rec.value
            ev = self._evidence(
                frame.frame_seq, "stack", roi.slot_id, "stack",
                rec.raw_score, conf, status,
            )
            field = self._field(field_val, conf, status, self._src("stack"), ev, ts)
            out.append(SlotObservation(slot_id=roi.slot_id, field=field))
        return tuple(out)

    def _recognize_actions(self, frame, table_map, ts):
        action_rois = sorted(
            (r for r in table_map.rois if r.kind is ROIKind.ACTION),
            key=lambda r: r.slot_id,
        )
        out = []
        for roi in action_rois:
            crop = extract_roi(frame, roi)
            rec = self._action.recognize(crop, roi.slot_id)
            cal = self._cal("action")
            if cal.should_abstain(rec.raw_score) or rec.value is None:
                conf = cal.calibrate(rec.raw_score).confidence
                status = ValidationStatus.UNKNOWN
                field_val = None
            else:
                conf = cal.calibrate(rec.raw_score).confidence
                status = ValidationStatus.VALID
                field_val = rec.value
                if (
                    cal.abstain_floor is not None
                    and not cal.should_abstain(rec.runner_up_score)
                ):
                    status = ValidationStatus.CONFLICT
                    field_val = None
            ev = self._evidence(
                frame.frame_seq, "action", roi.slot_id, "action",
                rec.raw_score, conf, status,
                components={"runner_up_raw": rec.runner_up_score},
            )
            field = self._field(field_val, conf, status, self._src("action"), ev, ts)
            out.append(SlotObservation(slot_id=roi.slot_id, field=field))
        return tuple(out)

    def _recognize_occupancies(self, frame, slot_stacks, ts):
        if self._empty_slot is None:
            return ()
        empty_scores = self._empty_slot.recognize(frame.image)
        stack_by_slot = {slot.slot_id: slot.field for slot in slot_stacks}
        slot_ids = sorted(set(stack_by_slot) | set(empty_scores))
        cal = self._cal("occupancy")
        out = []
        for slot_id in slot_ids:
            stack = stack_by_slot.get(slot_id)
            stack_raw = (
                float(stack.evidence.get("raw_score", 0.0))
                if stack is not None else 0.0
            )
            stack_present = bool(
                stack is not None
                and stack.validation_status is ValidationStatus.VALID
                and stack.value is not None
            )
            empty_raw = float(empty_scores.get(slot_id, 0.0))
            empty_present = not cal.should_abstain(empty_raw)
            raw_score = max(stack_raw, empty_raw)
            confidence = cal.calibrate(raw_score).confidence
            if stack_present and empty_present:
                value = None
                status = ValidationStatus.CONFLICT
            elif stack_present:
                value = True
                status = ValidationStatus.VALID
            elif empty_present:
                value = False
                status = ValidationStatus.VALID
            else:
                value = None
                status = ValidationStatus.UNKNOWN
            evidence = self._evidence(
                frame.frame_seq, "occupancy", slot_id, "occupancy",
                raw_score, confidence, status,
                components={
                    "stack_raw": stack_raw,
                    "empty_marker_raw": empty_raw,
                },
            )
            field = self._field(
                value, confidence, status, self._src("occupancy"),
                evidence, ts,
            )
            out.append(SlotObservation(slot_id=slot_id, field=field))
        return tuple(out)

    def _recognize_dealer(self, frame, ts):
        if self._dealer is None:
            return self._unknown_field(
                frame.frame_seq, "dealer_pos", None, "dealer", ts
            )
        rec = self._dealer.recognize(frame.image)
        cal = self._cal("dealer")
        confidence = cal.calibrate(rec.raw_score).confidence
        if rec.slot_id is None or cal.should_abstain(rec.raw_score):
            status = ValidationStatus.UNKNOWN
            value = None
        elif not cal.should_abstain(rec.runner_up_score):
            status = ValidationStatus.CONFLICT
            value = None
        else:
            status = ValidationStatus.VALID
            value = rec.slot_id
        evidence = self._evidence(
            frame.frame_seq, "dealer_pos", value, "dealer",
            rec.raw_score, confidence, status,
            components={"runner_up_raw": rec.runner_up_score},
        )
        return self._field(
            value, confidence, status, self._src("dealer"), evidence, ts
        )

    def _recognize_actor(self, frame, ts):
        if self._actor is None:
            return self._unknown_field(
                frame.frame_seq, "actor", None, "actor", ts
            )
        rec = self._actor.recognize(frame.image)
        cal = self._cal("actor")
        confidence = cal.calibrate(rec.raw_score).confidence
        if rec.actor_slot is None or cal.should_abstain(rec.raw_score):
            status = ValidationStatus.UNKNOWN
            value = None
        else:
            status = ValidationStatus.VALID
            value = rec.actor_slot
        evidence = self._evidence(
            frame.frame_seq, "actor", value, "actor",
            rec.raw_score, confidence, status,
        )
        return self._field(
            value, confidence, status, self._src("actor"), evidence, ts
        )


def _expected_board_count(street: Street | None) -> int | None:
    return {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }.get(street)


def _board_components(
    slots_rec: BoardSlotsRecognition,
    independent,
    cal_board,
    cal_card,
) -> dict:
    """Return per-component raw + calibrated evidence for board (auditability).

    Board is TWO signals, each calibrated against its OWN calibrator:
      - ``occupancy``: weakest occupancy raw evidence across 5 slots, plus its
        board-calibrator confidence.
      - ``identity``: weakest identity raw score across CARD slots (0.0 if no
        CARD slots), plus its card-calibrator confidence.

    Returns a dict of four keys:
      occupancy_app_raw / occupancy_app_conf / identity_app_raw / identity_app_conf
    so a reviewer can see both the raw score and its calibrated confidence for
    each signal — never a cross-signal min at the raw level.
    """
    occ_raws = [float(s.raw_score) for s in slots_rec.slots]
    occupancy_raw = min(occ_raws) if occ_raws else 0.0
    id_raws = [
        float(independent[i][2])
        for i, s in enumerate(slots_rec.slots)
        if s.occupancy is BoardSlotOccupancy.CARD and i < len(independent)
    ]
    identity_raw = min(id_raws) if id_raws else 0.0
    return {
        "occupancy_app_raw": occupancy_raw,
        "occupancy_app_conf": cal_board.calibrate(occupancy_raw).confidence,
        "identity_app_raw": identity_raw,
        "identity_app_conf": cal_card.calibrate(identity_raw).confidence,
    }


def _board_calibrated_confidence(
    slots_rec: BoardSlotsRecognition,
    independent,
    cal_board,
    cal_card,
) -> float:
    """Combined board confidence: calibrate occupancy and identity separately.

    Per slot:
      - CARD  slot -> min(board_cal(presence), card_cal(identity_score)); the
        identity-derived value is trustworthy only if BOTH its presence and its
        recognition calibrate confidently.
      - EMPTY slot -> board_cal(empty evidence).
      - UNKNOWN    -> 0.0 (weakest).

    The overall board confidence is the weakest (min) calibrated slot value.
    This never mixes raw scores across the two different scales — each signal
    is calibrated by its own calibrator first, then combined.
    """
    slots_conf = []
    for slot in slots_rec.slots:
        i = slot.slot_index
        if slot.occupancy is BoardSlotOccupancy.CARD:
            presence = float(slot.raw_score)
            id_score = independent[i][2] if i < len(independent) else 0.0
            occ_conf = cal_board.calibrate(presence).confidence
            id_conf = cal_card.calibrate(float(id_score)).confidence
            slots_conf.append(min(occ_conf, id_conf))
        elif slot.occupancy is BoardSlotOccupancy.EMPTY:
            slots_conf.append(cal_board.calibrate(float(slot.raw_score)).confidence)
        else:
            slots_conf.append(0.0)
    return float(min(slots_conf)) if slots_conf else 0.0


def _board_should_abstain(
    slots_rec: BoardSlotsRecognition, independent, cal_board, cal_card
) -> bool:
    """Abstain if ANY slot's relevant signal abstaains (per-signal calibrator).

    - CARD  slot: abstain if board presence OR card identity abstains.
    - EMPTY slot: abstain if board empty evidence abstains.
    """
    for slot in slots_rec.slots:
        i = slot.slot_index
        if slot.occupancy is BoardSlotOccupancy.CARD:
            presence = float(slot.raw_score)
            id_score = independent[i][2] if i < len(independent) else 0.0
            if cal_board.should_abstain(presence) or cal_card.should_abstain(
                float(id_score)
            ):
                return True
        elif slot.occupancy is BoardSlotOccupancy.EMPTY:
            if cal_board.should_abstain(float(slot.raw_score)):
                return True
    return False


def _crop_slot(img: np.ndarray, sub) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(sub.x * w)
    y0 = int(sub.y * h)
    x1 = int((sub.x + sub.width) * w)
    y1 = int((sub.y + sub.height) * h)
    return img[y0:y1, x0:x1]


__all__ = ["VisionEngine", "find_duplicate_card", "_conflict_between"]
