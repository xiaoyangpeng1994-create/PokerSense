"""Live capture from WePoker Android running in LDPlayer.

Assembles the whole chain the engine was built for —

    ADB CaptureService -> VisionEngine -> ChangeDetector -> Orchestrator
        -> StateEngine -> Equity -> RealtimeAnalysis

— against the emulator's raw portrait framebuffer, using calibration measured
from real 1440x2560 LDPlayer ADB captures.

Scope, stated plainly: **hero cards, board cards, street, pot, seat occupancy,
visual-slot stacks, the Dealer marker, and completed action labels** are
calibrated for Android. The versioned physical-slot mapping promotes those
fields to canonical seats and derives completed-action chip amounts from stack
deltas. The next player to act remains unavailable, so strategy advice still
fails closed while visible-card equity remains available.

Recoverable problems (ADB unavailable, emulator stopped, ambiguous devices,
or a resolution outside the calibrated aspect tolerance) are surfaced as
:class:`LiveCaptureError` so the UI can show what is wrong instead of the
process dying.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import cv2

from poker_engine.confidence.gate import ConfidenceGate
from poker_engine.core.enums import ActionType, PlayerStatus, Position, Street
from poker_engine.core.state import PlayerState, PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.perceptual.capture.base import (
    CaptureError,
    CaptureService,
    CaptureTarget,
    Frame,
)
from poker_engine.perceptual.vision.action_recognizer import (
    ActionTemplateSet,
    TemplateActionGlyphRecognizer,
    TemplateActionRecognizer,
)
from poker_engine.perceptual.vision.amount_recognizer import (
    DigitTemplateSet,
    TemplateAmountRecognizer,
)
from poker_engine.perceptual.vision.asset_manifest import VisionAssetManifest
from poker_engine.perceptual.vision.board_slot_detector import (
    TemplateBoardSlotDetector,
)
from poker_engine.perceptual.vision.calibration import (
    CalibrationBins,
    ConfidenceCalibrator,
    MeasuredCalibration,
)
from poker_engine.perceptual.vision.card_layout import (
    BoardSlotLayout,
    CardSubROI,
    board_layout_from_dict,
    hero_layout_from_dict,
)
from poker_engine.perceptual.vision.corner_glyph_recognizer import (
    CornerGlyphCardRecognizer,
    CornerGlyphTemplateSet,
)
from poker_engine.perceptual.vision.engine import VisionEngine
from poker_engine.perceptual.vision.errors import TableMapError
from poker_engine.perceptual.vision.hero_turn_recognizer import (
    AndroidHeroTurnRecognizer,
)
from poker_engine.perceptual.vision.street_detector import TemplateStreetDetector
from poker_engine.perceptual.vision.slot_marker_recognizer import (
    TemplatePerSlotMarkerRecognizer,
    TemplateSlotMarkerRecognizer,
    slot_marker_layout_from_dict,
)
from poker_engine.perceptual.vision.table_map import TableMap
from poker_engine.realtime.pipeline import RealtimePipeline
from poker_engine.state_engine.platform_mapping import (
    PlatformMappedStateEngine,
    PlatformSeatMapping,
)
from poker_engine.strategy.contracts import GameConfig, GameType
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.router import StrategyRouter

from .errors import LiveCaptureError
from .serialize import DesktopFrame
from .strategy_live import LiveStrategySession


def _resource_root() -> Path:
    """Return the checkout root or PyInstaller's bundled resource directory."""
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base is not None:
        return Path(frozen_base)
    return Path(__file__).resolve().parents[3]


_REPO_ROOT = _resource_root()
DEFAULT_PLATFORM = "wepoker_android"
DEFAULT_LAYOUT = "ldplayer_portrait_1440x2560"
DEFAULT_DEVICE_SERIAL = os.environ.get("POKERSENSE_ADB_SERIAL", "auto")


def build_capture_backend() -> CaptureService:
    """Return the production ADB backend."""
    from poker_engine.perceptual.capture.adb_backend import AdbBackend

    try:
        return AdbBackend()
    except RuntimeError as exc:
        raise LiveCaptureError(str(exc)) from exc


def load_platform_seat_mapping(
    platform: str = DEFAULT_PLATFORM,
    layout: str = DEFAULT_LAYOUT,
) -> PlatformSeatMapping:
    """Load the versioned Android visual-slot to canonical-seat contract."""
    path = (
        _REPO_ROOT / "configs" / "platform"
        / f"{platform}__{layout}_seat_mapping.json"
    )
    if not path.is_file():
        raise LiveCaptureError(f"seat mapping missing at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise LiveCaptureError("unsupported seat mapping schema")

    def slot_map(name: str) -> dict[int, int]:
        return {int(slot): seat for slot, seat in data[name].items()}

    if data.get("platform_id") != platform or data.get("layout_id") != layout:
        raise LiveCaptureError("seat mapping platform/layout mismatch")
    return PlatformSeatMapping(
        platform_id=platform,
        layout_id=layout,
        version=data["version"],
        stack_slot_to_seat=slot_map("stack_slot_to_seat"),
        action_slot_to_seat=slot_map("action_slot_to_seat"),
        actor_slot_to_seat=slot_map("actor_slot_to_seat"),
        dealer_slot_to_seat=slot_map("dealer_slot_to_seat"),
        occupancy_slot_to_seat=slot_map("occupancy_slot_to_seat"),
        actor_observation_is_current=data["actor_observation_is_current"],
    )


def _uncalibrated(name: str) -> ConfidenceCalibrator:
    """A calibrator for a field nothing has been measured about.

    It reports zero confidence at every raw score, so the field is always
    gated to UNKNOWN. This is deliberate: an uncalibrated field must not
    inherit a calibrated field's confidence just because they share a frame.
    """
    return ConfidenceCalibrator(
        name=name,
        version=1,
        bins=CalibrationBins(edges=(0.0, 1.0), confidence=(0.0,)),
        abstain_floor=1.0,
    )


def load_measured_calibration(vision_dir: Path) -> MeasuredCalibration:
    """Load the committed accuracy measurement for a platform's cards."""
    path = vision_dir / "calibration.json"
    if not path.is_file():
        raise LiveCaptureError(f"no calibration measurement at {path}")
    data = json.loads(path.read_text())["card"]
    return MeasuredCalibration(
        samples=data["samples"],
        correct=data["correct"],
        readable_score_floor=data["readable_score_floor"],
        unreadable_score_ceiling=data["unreadable_score_ceiling"],
        source=data["source"],
    )


def load_measured_calibrations(
    vision_dir: Path,
) -> dict[str, MeasuredCalibration]:
    """Load every independently measured field calibration in a profile."""
    path = vision_dir / "calibration.json"
    if not path.is_file():
        raise LiveCaptureError(f"no calibration measurement at {path}")
    data = json.loads(path.read_text())
    measured = {}
    for name in (
        "card", "board", "street", "amount", "stack", "dealer", "action",
        "occupancy", "actor",
    ):
        field = data.get(name)
        if field is None:
            continue
        measured[name] = MeasuredCalibration(
            samples=field["samples"],
            correct=field["correct"],
            readable_score_floor=field["readable_score_floor"],
            unreadable_score_ceiling=field["unreadable_score_ceiling"],
            source=field["source"],
        )
    if "card" not in measured:
        raise LiveCaptureError(f"card calibration missing at {path}")
    return measured


def build_confidence_gate(
    measured: MeasuredCalibration | dict[str, MeasuredCalibration],
) -> ConfidenceGate:
    """Gate thresholds set to what the measurement actually supports.

    The threshold is not a wish about how good recognition ought to be; it
    is the accuracy the evidence demonstrates. Collecting more samples
    raises what can be claimed, and the threshold rises with it — see
    ``configs/vision/<platform>/calibration.json``.

    Fields with no measurement keep a threshold of 1.0, which nothing can
    clear, so they stay UNKNOWN until they are calibrated too.
    """
    if isinstance(measured, MeasuredCalibration):
        fields = {"card": measured}
    else:
        fields = dict(measured)
    card_confidence = fields["card"].justified_confidence
    thresholds = {
        "hero_cards": card_confidence,
        "board_cards": (
            fields["board"].justified_confidence if "board" in fields else 1.0
        ),
        "street": (
            fields["street"].justified_confidence if "street" in fields else 1.0
        ),
        "pot": (
            fields["amount"].justified_confidence if "amount" in fields else 1.0
        ),
        "stacks": (
            fields["stack"].justified_confidence if "stack" in fields else 1.0
        ),
        "bet_size": 1.0,
        "action": (
            fields["action"].justified_confidence
            if "action" in fields else 1.0
        ),
    }
    occupancy = (
        fields["occupancy"].justified_confidence
        if "occupancy" in fields else 1.0
    )
    return ConfidenceGate(
        thresholds=thresholds,
        slot_thresholds={"occupancy": occupancy},
    )


def _load_templates(directory: Path) -> dict:
    if not directory.is_dir():
        raise LiveCaptureError(f"template directory missing: {directory}")
    templates = {
        path.stem: cv2.imread(str(path)) for path in sorted(directory.glob("*.png"))
    }
    if not templates:
        raise LiveCaptureError(f"no templates found in {directory}")
    return templates


def load_calibration(
    platform: str = DEFAULT_PLATFORM, layout: str = DEFAULT_LAYOUT
) -> tuple[TableMap, VisionEngine]:
    """Load a platform's committed calibration into a ready VisionEngine."""
    table_map_path = _REPO_ROOT / "configs" / "platform" / f"{platform}__{layout}.json"
    vision_dir = _REPO_ROOT / "configs" / "vision" / platform
    if not table_map_path.is_file():
        raise LiveCaptureError(
            f"no calibration for {platform}/{layout}: {table_map_path}"
        )

    table_map = TableMap.from_json(table_map_path.read_text())
    calibration_data = json.loads((vision_dir / "calibration.json").read_text())
    template_source = calibration_data.get("template_source", platform)
    template_dir = _REPO_ROOT / "configs" / "vision" / template_source
    hero_layout = hero_layout_from_dict(
        json.loads((vision_dir / "hero_slot_layout.json").read_text())
    )

    card_recognizer = CornerGlyphCardRecognizer(
        CornerGlyphTemplateSet(
            rank_templates=_load_templates(template_dir / "rank"),
            suit_templates=_load_templates(template_dir / "suit"),
            version=f"{platform}-{layout}-v1",
        )
    )

    board_layout_path = vision_dir / "board_slot_layout.json"
    if board_layout_path.is_file():
        board_layout = board_layout_from_dict(
            json.loads(board_layout_path.read_text())
        )
    else:
        board_layout = BoardSlotLayout(
            layout_id=f"{layout}_board_uncalibrated",
            version=1,
            slots=tuple(
                CardSubROI(x=i * 0.2, y=0.0, width=0.19, height=1.0)
                for i in range(5)
            ),
        )
    _placeholder_rank = cv2.imread(str(next((template_dir / "rank").glob("*.png"))))
    digit_dir = vision_dir / "digit"
    digit_templates = (
        _load_templates(digit_dir)
        if digit_dir.is_dir()
        else {"0": _placeholder_rank}
    )
    amount_recognizer = TemplateAmountRecognizer(
        DigitTemplateSet(
            templates=digit_templates,
            version=(f"{platform}-amount-v2" if digit_dir.is_dir() else "uncalibrated"),
        ),
        min_score=0.8,
    )
    stack_digit_dir = vision_dir / "stack_digit"
    stack_recognizer = TemplateAmountRecognizer(
        DigitTemplateSet(
            templates=(
                _load_templates(stack_digit_dir)
                if stack_digit_dir.is_dir()
                else {"0": _placeholder_rank}
            ),
            version=(
                f"{platform}-stack-v1"
                if stack_digit_dir.is_dir()
                else "uncalibrated"
            ),
        ),
        min_score=0.8,
    )
    action_glyph_dir = vision_dir / "action_glyph"
    if action_glyph_dir.is_dir():
        action_files = _load_templates(action_glyph_dir)
        action_recognizer = TemplateActionGlyphRecognizer(
            ActionTemplateSet(
                templates={ActionType(name): image
                           for name, image in action_files.items()},
                version=f"{platform}-action-v1",
            ),
            min_score=0.8,
        )
    else:
        action_recognizer = TemplateActionRecognizer(
            ActionTemplateSet(
                templates={ActionType.FOLD: cv2.imread(
                    str(next((template_dir / "suit").glob("*.png")))
                )},
                version="uncalibrated",
            )
        )
    dealer_layout_path = vision_dir / "dealer_slot_layout.json"
    dealer_template_path = vision_dir / "dealer" / "marker.png"
    dealer_recognizer = None
    if dealer_layout_path.is_file() and dealer_template_path.is_file():
        dealer_recognizer = TemplateSlotMarkerRecognizer(
            cv2.imread(str(dealer_template_path)),
            slot_marker_layout_from_dict(
                json.loads(dealer_layout_path.read_text())
            ),
            version=f"{platform}-dealer-v1",
        )
    empty_layout_path = vision_dir / "empty_slot_layout.json"
    empty_template_path = vision_dir / "empty_slot" / "plus.png"
    empty_slot_recognizer = None
    if empty_layout_path.is_file() and empty_template_path.is_file():
        empty_slot_recognizer = TemplatePerSlotMarkerRecognizer(
            cv2.imread(str(empty_template_path)),
            slot_marker_layout_from_dict(
                json.loads(empty_layout_path.read_text())
            ),
            version=f"{platform}-occupancy-v1",
        )

    measured = load_measured_calibrations(vision_dir)
    card_calibrator = measured["card"].to_calibrator("card")
    # Fields with no calibration measurement of their own get a calibrator
    # that can never clear a gate: nothing has been measured about them, so
    # they must read UNKNOWN rather than borrow the card measurement's
    # credibility.
    calibrators = {"card": card_calibrator}
    for name in (
        "amount", "stack", "dealer", "action", "occupancy", "actor", "street",
        "board",
    ):
        field_measurement = measured.get(name)
        calibrators[name] = (
            field_measurement.to_calibrator(name)
            if field_measurement is not None
            else _uncalibrated(name)
        )
    manifest = VisionAssetManifest(
        platform_id=platform,
        layout_id=layout,
        card_layout_version=board_layout.version,
        template_set_version=f"{platform}-v1",
        calibration_version=3,
        recognizer_versions={
            "card": "1", "amount": "2", "stack": "1", "dealer": "1",
            "action": "2", "occupancy": "1", "actor": "1",
            "street": "2", "board": "2",
        },
    )

    engine = VisionEngine(
        board_layout=board_layout,
        hero_layout=hero_layout,
        card_recognizer=card_recognizer,
        board_slot_detector=TemplateBoardSlotDetector(
            board_layout,
            empty_min_evidence=0.55,
            card_min_presence=0.50,
        ),
        street_detector=TemplateStreetDetector(),
        amount_recognizer=amount_recognizer,
        stack_recognizer=stack_recognizer,
        action_recognizer=action_recognizer,
        calibrators=calibrators,
        manifest=manifest,
        bet_size_semantics=None,
        dealer_recognizer=dealer_recognizer,
        empty_slot_recognizer=empty_slot_recognizer,
        actor_recognizer=AndroidHeroTurnRecognizer(),
    )
    return table_map, engine


class DeviceFrameSource:
    """FrameSource that captures one explicit ADB device on every pull."""

    def __init__(
        self,
        backend: CaptureService,
        device_serial: str,
    ) -> None:
        self._backend = backend
        self._target = CaptureTarget(window_id=device_serial)

    def next_frame(self) -> Frame | None:
        try:
            return self._backend.capture(self._target)
        except CaptureError as exc:
            raise LiveCaptureError(str(exc)) from exc


def _seed_player(seat: int, hero: bool) -> PlayerState:
    return PlayerState(
        player_id=f"p{seat}",
        seat=seat,
        position=Position.UNKNOWN,
        stack=ChipAmount("0"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.SITTING_OUT,
        has_cards=False,
        is_hero=hero,
        is_dealer=False,
    )


def _seed_state(hand_id: str = "live-1") -> PokerState:
    """An empty preflop state; recognition drives every field from here."""
    return PokerState(
        state_version=0,
        hand_id=hand_id,
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=tuple(_seed_player(seat, hero=seat == 0) for seat in range(8)),
        pot=ChipAmount("0"),
        current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"),
        actor=None,
    )


def build_pipeline(
    device_serial: str = DEFAULT_DEVICE_SERIAL,
    platform: str = DEFAULT_PLATFORM,
    layout: str = DEFAULT_LAYOUT,
) -> RealtimePipeline:
    """Assemble a pipeline reading raw portrait frames from LDPlayer."""
    table_map, vision = load_calibration(platform, layout)
    measured = load_measured_calibrations(
        _REPO_ROOT / "configs" / "vision" / platform
    )
    seat_mapping = load_platform_seat_mapping(platform, layout)
    orchestrator = ApplicationOrchestrator(
        state_engine=PlatformMappedStateEngine(seat_mapping),
        hand_memory=InMemoryHandMemory(),
        confidence_gate=build_confidence_gate(measured),
    )
    next_hand_number = 1

    def next_hand_state() -> PokerState:
        nonlocal next_hand_number
        next_hand_number += 1
        return _seed_state(hand_id=f"live-{next_hand_number}")

    orchestrator.start_hand(_seed_state(hand_id="live-1"))
    frame_source = DeviceFrameSource(build_capture_backend(), device_serial)
    return RealtimePipeline(
        frame_source,
        vision,
        table_map,
        orchestrator,
        # A valid hero hand is immutable within a deal. Requiring two
        # identical frames prevents one half-rendered / miscaptured frame
        # from becoming the canonical hand shown to the player.
        hero_confirmation_frames=2,
        confirmation_frames={
            "board_cards": 2,
            "pot": 2,
            "stacks": 2,
            "bet_size": 2,
            "action": 2,
            "street": 2,
            "dealer_pos": 2,
            "actor": 2,
            "slot_stacks": 2,
            "slot_actions": 2,
            "slot_occupancies": 2,
        },
        new_hand_state_factory=next_hand_state,
    )


async def live_analysis_stream(
    device_serial: str = DEFAULT_DEVICE_SERIAL,
    interval_seconds: float = 1.0,
) -> AsyncIterator[DesktopFrame]:
    """Yield fresh analysis while the selected ADB device is available.

    Capture and recognition are CPU-bound and run off the event loop, so the
    server stays responsive between frames.
    """
    try:
        pipeline = await asyncio.to_thread(
            build_pipeline, device_serial
        )
    except LiveCaptureError:
        raise
    except Exception as exc:
        raise LiveCaptureError(
            f"capture engine initialization failed: {exc}"
        ) from exc
    # Recognition now promotes measured seat, stack, dealer, Hero-turn, and
    # completed-action evidence through the Android mapping. No released
    # multiplayer Provider is bundled, so advice still emits an auditable
    # ABSTAIN rather than inventing a strategy.
    strategy_session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter()),
        GameConfig(
            variant="NLHE",
            game_type=GameType.CASH,
            max_seats=8,
            dealt_player_count=8,
            small_blind=ChipAmount("1"),
            big_blind=ChipAmount("2"),
            minimum_chip=ChipAmount("1"),
        ),
    )
    while True:
        try:
            step = await asyncio.to_thread(pipeline.step)
        except (LiveCaptureError, TableMapError) as exc:
            raise LiveCaptureError(str(exc)) from exc
        except Exception as exc:
            raise LiveCaptureError(f"capture engine failed: {exc}") from exc
        if step is not None:
            yield strategy_session.frame(
                step.analysis,
                pipeline.current_state(),
                action_history=pipeline.action_history(),
            )
        await asyncio.sleep(interval_seconds)


__all__ = [
    "LiveCaptureError",
    "build_confidence_gate",
    "load_measured_calibration",
    "load_measured_calibrations",
    "load_platform_seat_mapping",
    "DeviceFrameSource",
    "build_capture_backend",
    "build_pipeline",
    "live_analysis_stream",
    "load_calibration",
    "DEFAULT_DEVICE_SERIAL",
]
