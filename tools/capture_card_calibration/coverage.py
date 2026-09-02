"""Stage G minimum coverage evaluation (guide section 10).

Section 10's table is a **floor**, not a screenshot count: it counts
human-confirmed, dispersed samples. This module turns it into an executable
check so a dataset can never be declared ready on vibes.

Two honest limitations, stated up front:

- **Negative counts are proxies.** The label schema records what is *not*
  readable, not why (card back vs. occlusion vs. animation), because the
  guide does not define a negative-reason field. A negative count is
  therefore an upper bound on "genuine hard negatives" and must be
  spot-checked against contact sheets before it counts as evidence.
- **Where the guide gives no number**, this module picks an explicit project
  minimum and labels it as such in the requirement description, rather than
  silently treating an unquantified rule as satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import SLOT_COUNT
from .schema import (
    MIN_SESSIONS,
    CompletedAction,
    FrameLabel,
    LabelStatus,
    Scene,
    Street,
)

MIN_ANOMALY_FRAMES = 50
STREET_REQUIREMENTS: dict[str, int] = {
    Street.FLOP.value: 20,
    Street.TURN.value: 15,
    Street.RIVER.value: 15,
}
MIN_PREFLOP_EMPTY_BOARDS = 20
ACTION_REQUIREMENTS: dict[str, int] = {
    CompletedAction.FOLD.value: 10,
    CompletedAction.CHECK.value: 10,
    CompletedAction.CALL.value: 10,
    CompletedAction.BET.value: 10,
    CompletedAction.RAISE.value: 10,
    CompletedAction.ALL_IN.value: 6,
}
# Section 10 says actions must be spread across slots but gives no number.
MIN_ACTION_SLOTS = 4
TEMPORAL_REQUIREMENTS: dict[str, int] = {
    "deal": 20,
    "action": 30,
    "street_change": 10,
    "hand_end": 10,
}
MIN_RECONNECT_GROUPS = 5

ANOMALY_SCENES = frozenset(
    {
        Scene.MENU,
        Scene.OVERLAY,
        Scene.SIGNAL_LOSS,
        Scene.RECONNECT,
    }
)
DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class Requirement:
    """One row of the section 10 table, with what was actually measured."""

    field: str
    description: str
    required_positive: int
    required_negative: int
    measured_positive: int
    measured_negative: int
    shortfalls: tuple[str, ...] = ()

    @property
    def met(self) -> bool:
        return not self.shortfalls

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "description": self.description,
            "required_positive": self.required_positive,
            "required_negative": self.required_negative,
            "measured_positive": self.measured_positive,
            "measured_negative": self.measured_negative,
            "shortfalls": list(self.shortfalls),
            "met": self.met,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Result of evaluating a labelled dataset against section 10."""

    sessions: tuple[str, ...]
    frame_count: int
    requirements: tuple[Requirement, ...]

    @property
    def unmet(self) -> tuple[Requirement, ...]:
        return tuple(item for item in self.requirements if not item.met)

    @property
    def is_complete(self) -> bool:
        return not self.unmet

    @property
    def gap_lines(self) -> list[str]:
        """Precise top-up list for the review report (section 17)."""
        lines: list[str] = []
        for item in self.unmet:
            for shortfall in item.shortfalls:
                lines.append(f"{item.field}: {shortfall}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": list(self.sessions),
            "frame_count": self.frame_count,
            "complete": self.is_complete,
            "requirements": [item.to_dict() for item in self.requirements],
        }


def _digits_of(values: Iterable[int]) -> set[str]:
    found: set[str] = set()
    for value in values:
        found.update(str(int(value)))
    return found


def _headcount_bucket(occupied: int) -> str | None:
    if occupied == 2:
        return "2"
    if 3 <= occupied <= 5:
        return "3-5"
    if 6 <= occupied <= 8:
        return "6-8"
    return None


@dataclass
class _Tally:
    """Single pass accumulator over the labelled frames."""

    sessions: set[str] = field(default_factory=set)
    frame_count: int = 0
    # hero cards
    hero_hands: set[tuple[str, ...]] = field(default_factory=set)
    hero_stable_frames: int = 0
    hero_negative: int = 0
    hero_ranks: set[str] = field(default_factory=set)
    hero_suits: set[str] = field(default_factory=set)
    # board / street
    board_faces: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)
    preflop_empty_stable: int = 0
    board_negative: int = 0
    # pot
    pot_values: set[int] = field(default_factory=set)
    pot_negative: int = 0
    # stacks
    stack_values: dict[int, set[int]] = field(default_factory=dict)
    stack_negative: dict[int, int] = field(default_factory=dict)
    stack_digits: set[str] = field(default_factory=set)
    # dealer
    dealer_positive: dict[int, int] = field(default_factory=dict)
    dealer_negative: int = 0
    # occupancy
    occupancy_states: int = 0
    occupancy_slot_labels: int = 0
    headcount_buckets: set[str] = field(default_factory=set)
    # completed actions
    action_counts: dict[str, int] = field(default_factory=dict)
    action_slots: set[int] = field(default_factory=set)
    action_negative: int = 0
    # hero actor
    actor_positive: int = 0
    actor_negative: int = 0
    # scenes and groups
    anomaly_frames: int = 0
    group_scenes: dict[str, set[str]] = field(default_factory=dict)
    group_streets: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.board_faces = {
            street: set() for street in STREET_REQUIREMENTS
        }
        self.stack_values = {slot: set() for slot in range(SLOT_COUNT)}
        self.stack_negative = {slot: 0 for slot in range(SLOT_COUNT)}
        self.dealer_positive = {slot: 0 for slot in range(SLOT_COUNT)}
        self.action_counts = {name: 0 for name in ACTION_REQUIREMENTS}


def _accumulate(labels: Sequence[FrameLabel]) -> _Tally:
    tally = _Tally()
    for label in labels:
        tally.frame_count += 1
        tally.sessions.add(label.session_id)
        group = label.group_id
        tally.group_scenes.setdefault(group, set()).add(label.scene.value)

        street_value = (
            label.street.value
            if label.street.status is LabelStatus.VALID
            else None
        )
        if street_value is not None:
            tally.group_streets.setdefault(group, set()).add(street_value)

        if label.scene in ANOMALY_SCENES:
            tally.anomaly_frames += 1

        # --- hero cards ---
        if label.hero_cards.status is LabelStatus.VALID:
            cards = tuple(sorted(label.hero_cards.value))
            tally.hero_hands.add(cards)
            for card in cards:
                tally.hero_ranks.add(card[0])
                tally.hero_suits.add(card[1])
            if label.stable:
                tally.hero_stable_frames += 1
        else:
            tally.hero_negative += 1

        # --- board / street ---
        if label.board_cards.status is LabelStatus.VALID and street_value:
            board = tuple(sorted(label.board_cards.value))
            if street_value in tally.board_faces:
                tally.board_faces[street_value].add(board)
        elif street_value == Street.PRE_FLOP.value and label.stable:
            tally.preflop_empty_stable += 1
        else:
            tally.board_negative += 1

        # --- pot ---
        if label.pot.status is LabelStatus.VALID:
            tally.pot_values.add(int(label.pot.value))
        else:
            tally.pot_negative += 1

        # --- per-slot fields ---
        frame_has_dealer = False
        frame_has_actor = False
        frame_has_action = False
        occupancy_signature: list[str] = []
        all_occupancy_valid = True
        for slot in label.slots:
            if slot.occupancy.status is LabelStatus.VALID:
                tally.occupancy_slot_labels += 1
                occupancy_signature.append(str(slot.occupancy.value))
            else:
                all_occupancy_valid = False
                occupancy_signature.append("?")

            if slot.stack.status is LabelStatus.VALID:
                tally.stack_values[slot.slot_id].add(int(slot.stack.value))
            else:
                tally.stack_negative[slot.slot_id] += 1

            if slot.dealer.status is LabelStatus.VALID:
                tally.dealer_positive[slot.slot_id] += 1
                frame_has_dealer = True

            if slot.completed_action.status is LabelStatus.VALID:
                name = str(slot.completed_action.value)
                if name in tally.action_counts:
                    tally.action_counts[name] += 1
                tally.action_slots.add(slot.slot_id)
                frame_has_action = True

            if slot.current_actor.status is LabelStatus.VALID:
                tally.actor_positive += 1
                frame_has_actor = True

        if all_occupancy_valid and label.stable and label.scene is Scene.TABLE:
            tally.occupancy_states += 1
            occupied = sum(
                1 for value in occupancy_signature if value == "OCCUPIED"
            )
            bucket = _headcount_bucket(occupied)
            if bucket:
                tally.headcount_buckets.add(bucket)

        if not frame_has_dealer:
            tally.dealer_negative += 1
        if not frame_has_action:
            tally.action_negative += 1
        if not frame_has_actor:
            tally.actor_negative += 1

    tally.stack_digits = _digits_of(
        value for values in tally.stack_values.values() for value in values
    )
    return tally


def _requirement(
    name: str,
    description: str,
    required_positive: int,
    required_negative: int,
    measured_positive: int,
    measured_negative: int,
    *shortfalls: str,
) -> Requirement:
    return Requirement(
        field=name,
        description=description,
        required_positive=required_positive,
        required_negative=required_negative,
        measured_positive=measured_positive,
        measured_negative=measured_negative,
        shortfalls=tuple(item for item in shortfalls if item),
    )


def evaluate_coverage(labels: Sequence[FrameLabel]) -> CoverageReport:
    """Evaluate a labelled dataset against the section 10 minimums."""
    tally = _accumulate(labels)
    requirements: list[Requirement] = []

    # --- sessions (section 5 / 10) ---
    session_shortfall = ""
    if len(tally.sessions) < MIN_SESSIONS:
        session_shortfall = (
            f"need >= {MIN_SESSIONS} independent capture sessions, "
            f"found {len(tally.sessions)}"
        )
    requirements.append(
        _requirement(
            "sessions",
            "At least 3 independent capture sessions",
            MIN_SESSIONS,
            0,
            len(tally.sessions),
            0,
            session_shortfall,
        )
    )

    # --- hero cards ---
    hero_shortfalls: list[str] = []
    if len(tally.hero_hands) < 40:
        hero_shortfalls.append(
            f"need >= 40 distinct hero hands, found {len(tally.hero_hands)}"
        )
    if tally.hero_stable_frames < 80:
        hero_shortfalls.append(
            "need >= 80 stable frames with readable hero cards, found "
            f"{tally.hero_stable_frames}"
        )
    if tally.hero_negative < 40:
        hero_shortfalls.append(
            "need >= 40 card-back / folded / occluded / deal-animation "
            f"frames, found {tally.hero_negative}"
        )
    missing_ranks = sorted(set("23456789TJQKA") - tally.hero_ranks)
    if missing_ranks:
        hero_shortfalls.append(
            "ranks never observed: " + ", ".join(missing_ranks)
        )
    missing_suits = sorted(set("CDHS") - tally.hero_suits)
    if missing_suits:
        hero_shortfalls.append(
            "suits never observed: " + ", ".join(missing_suits)
        )
    requirements.append(
        _requirement(
            "hero_cards",
            "40 distinct hands / >= 80 stable frames; all ranks and suits",
            40,
            40,
            len(tally.hero_hands),
            tally.hero_negative,
            *hero_shortfalls,
        )
    )

    # --- board / street ---
    board_shortfalls: list[str] = []
    board_positive = 0
    for street, minimum in STREET_REQUIREMENTS.items():
        found = len(tally.board_faces[street])
        board_positive += found
        if found < minimum:
            board_shortfalls.append(
                f"{street}: need >= {minimum} distinct stable boards, "
                f"found {found}"
            )
    if tally.preflop_empty_stable < MIN_PREFLOP_EMPTY_BOARDS:
        board_shortfalls.append(
            "need >= 20 stable preflop frames with an empty board, found "
            f"{tally.preflop_empty_stable}"
        )
    if tally.board_negative < 40:
        board_shortfalls.append(
            f"need >= 40 deal / occluded negatives, found "
            f"{tally.board_negative}"
        )
    requirements.append(
        _requirement(
            "board_cards",
            "Flop >= 20, Turn >= 15, River >= 15 distinct boards + preflop",
            50,
            40,
            board_positive,
            tally.board_negative,
            *board_shortfalls,
        )
    )

    # --- pot ---
    pot_shortfalls: list[str] = []
    if len(tally.pot_values) < 40:
        pot_shortfalls.append(
            f"need >= 40 distinct pot values, found {len(tally.pot_values)}"
        )
    if tally.pot_negative < 60:
        pot_shortfalls.append(
            f"need >= 60 non-pot / animated / occluded negatives, found "
            f"{tally.pot_negative}"
        )
    missing_digits = sorted(DIGITS - _digits_of(tally.pot_values))
    if missing_digits:
        pot_shortfalls.append(
            "digits never observed in pot values: " + "".join(missing_digits)
        )
    requirements.append(
        _requirement(
            "pot",
            ">= 40 distinct pot values; digits 0-9 all covered",
            40,
            60,
            len(tally.pot_values),
            tally.pot_negative,
            *pot_shortfalls,
        )
    )

    # --- stacks ---
    stack_shortfalls: list[str] = []
    stack_positive = sum(len(values) for values in tally.stack_values.values())
    stack_negative = sum(tally.stack_negative.values())
    thin_slots = [
        slot
        for slot, values in tally.stack_values.items()
        if len(values) < 12
    ]
    if thin_slots:
        stack_shortfalls.append(
            "slots with < 12 readable stacks: "
            + ", ".join(str(slot) for slot in thin_slots)
        )
    if stack_positive < 96:
        stack_shortfalls.append(
            f"need >= 96 readable stack values in total, found {stack_positive}"
        )
    if stack_negative < 80:
        stack_shortfalls.append(
            f"need >= 80 empty / occluded / animated negatives, found "
            f"{stack_negative}"
        )
    missing_stack_digits = sorted(DIGITS - tally.stack_digits)
    if missing_stack_digits:
        stack_shortfalls.append(
            "digits never observed in stacks: " + "".join(missing_stack_digits)
        )
    requirements.append(
        _requirement(
            "stack",
            "per slot >= 12, total >= 96; digits 0-9 all covered",
            96,
            80,
            stack_positive,
            stack_negative,
            *stack_shortfalls,
        )
    )

    # --- dealer ---
    dealer_shortfalls: list[str] = []
    dealer_positive = sum(tally.dealer_positive.values())
    for slot, count in tally.dealer_positive.items():
        if count < 6:
            dealer_shortfalls.append(
                f"slot {slot}: need >= 6 dealer observations, found {count}"
            )
    if dealer_positive < 48:
        dealer_shortfalls.append(
            f"need >= 48 dealer observations in total, found {dealer_positive}"
        )
    if tally.dealer_negative < 50:
        dealer_shortfalls.append(
            "need >= 50 absent / hidden / animated / multi-candidate "
            f"negatives, found {tally.dealer_negative}"
        )
    requirements.append(
        _requirement(
            "dealer",
            "per slot >= 6, total >= 48",
            48,
            50,
            dealer_positive,
            tally.dealer_negative,
            *dealer_shortfalls,
        )
    )

    # --- occupancy ---
    occupancy_shortfalls: list[str] = []
    if tally.occupancy_states < 40:
        occupancy_shortfalls.append(
            "need >= 40 stable table states with all 8 slots labelled, "
            f"found {tally.occupancy_states}"
        )
    if tally.occupancy_slot_labels < 320:
        occupancy_shortfalls.append(
            f"need >= 320 slot occupancy labels, found "
            f"{tally.occupancy_slot_labels}"
        )
    missing_buckets = sorted({"2", "3-5", "6-8"} - tally.headcount_buckets)
    if missing_buckets:
        occupancy_shortfalls.append(
            "head-count buckets never observed: " + ", ".join(missing_buckets)
        )
    requirements.append(
        _requirement(
            "occupancy",
            ">= 40 stable table states, >= 320 slot labels, 2/3-5/6-8 players",
            320,
            0,
            tally.occupancy_slot_labels,
            0,
            *occupancy_shortfalls,
        )
    )

    # --- completed actions ---
    action_shortfalls: list[str] = []
    for name, minimum in ACTION_REQUIREMENTS.items():
        found = tally.action_counts[name]
        if found < minimum:
            action_shortfalls.append(
                f"{name}: need >= {minimum}, found {found}"
            )
    if tally.action_negative < 80:
        action_shortfalls.append(
            "need >= 80 hard negatives (hero controls, nicknames, avatars, "
            f"cards, menus, occlusion), found {tally.action_negative}"
        )
    if len(tally.action_slots) < MIN_ACTION_SLOTS:
        action_shortfalls.append(
            f"actions must spread across slots; project minimum "
            f"{MIN_ACTION_SLOTS} distinct slots, found "
            f"{len(tally.action_slots)}"
        )
    requirements.append(
        _requirement(
            "completed_action",
            "FOLD/CHECK/CALL/BET/RAISE >= 10, ALL_IN >= 6, spread across slots",
            56,
            80,
            sum(tally.action_counts.values()),
            tally.action_negative,
            *action_shortfalls,
        )
    )

    # --- hero actor ---
    actor_shortfalls: list[str] = []
    if tally.actor_positive < 40:
        actor_shortfalls.append(
            f"need >= 40 hero-turn positives, found {tally.actor_positive}"
        )
    if tally.actor_negative < 160:
        actor_shortfalls.append(
            "need >= 160 non-hero-turn / result / menu / animation "
            f"negatives, found {tally.actor_negative}"
        )
    requirements.append(
        _requirement(
            "hero_actor",
            ">= 40 hero-turn positives, >= 160 negatives",
            40,
            160,
            tally.actor_positive,
            tally.actor_negative,
            *actor_shortfalls,
        )
    )

    # --- temporal groups ---
    temporal_shortfalls: list[str] = []
    deal_groups = sum(
        1
        for scenes in tally.group_scenes.values()
        if Scene.DEAL_TRANSITION.value in scenes
    )
    action_groups = sum(
        1
        for scenes in tally.group_scenes.values()
        if Scene.ACTION_TRANSITION.value in scenes
    )
    street_groups = sum(
        1 for streets in tally.group_streets.values() if len(streets) >= 2
    )
    hand_end_groups = sum(
        1
        for scenes in tally.group_scenes.values()
        if Scene.RESULT.value in scenes
    )
    reconnect_groups = sum(
        1
        for scenes in tally.group_scenes.values()
        if Scene.SIGNAL_LOSS.value in scenes or Scene.RECONNECT.value in scenes
    )
    observed_groups = {
        "deal": deal_groups,
        "action": action_groups,
        "street_change": street_groups,
        "hand_end": hand_end_groups,
    }
    for name, minimum in TEMPORAL_REQUIREMENTS.items():
        found = observed_groups[name]
        if found < minimum:
            temporal_shortfalls.append(
                f"{name}: need >= {minimum} groups, found {found}"
            )
    if reconnect_groups < MIN_RECONNECT_GROUPS:
        temporal_shortfalls.append(
            f"need >= {MIN_RECONNECT_GROUPS} signal-loss / reconnect groups, "
            f"found {reconnect_groups}"
        )
    requirements.append(
        _requirement(
            "temporal",
            "deal >= 20, action >= 30, street change >= 10, hand end >= 10",
            70,
            MIN_RECONNECT_GROUPS,
            sum(observed_groups.values()),
            reconnect_groups,
            *temporal_shortfalls,
        )
    )

    # --- global anomaly scenes ---
    anomaly_shortfalls: list[str] = []
    if tally.anomaly_frames < MIN_ANOMALY_FRAMES:
        anomaly_shortfalls.append(
            "need >= 50 menu / occluded / black / corrupted / signal-loss "
            f"frames, found {tally.anomaly_frames}"
        )
    requirements.append(
        _requirement(
            "anomaly_scenes",
            "menu, occlusion, black screen, corruption, signal loss >= 50",
            0,
            MIN_ANOMALY_FRAMES,
            0,
            tally.anomaly_frames,
            *anomaly_shortfalls,
        )
    )

    return CoverageReport(
        sessions=tuple(sorted(tally.sessions)),
        frame_count=tally.frame_count,
        requirements=tuple(requirements),
    )


__all__ = [
    "ACTION_REQUIREMENTS",
    "ANOMALY_SCENES",
    "CoverageReport",
    "MIN_ACTION_SLOTS",
    "MIN_ANOMALY_FRAMES",
    "MIN_PREFLOP_EMPTY_BOARDS",
    "MIN_RECONNECT_GROUPS",
    "Requirement",
    "STREET_REQUIREMENTS",
    "TEMPORAL_REQUIREMENTS",
    "evaluate_coverage",
]
