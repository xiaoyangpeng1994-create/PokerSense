"""Stage D: frame sampling, perceptual dedup, and scene tagging.

Reads a finalized capture-card MKV, normalizes each sampled frame (stage C),
tags its scene, perceptual-dedups stable frames, and writes
``normalized/manifest.json`` plus the PNG frames themselves.

Design notes
------------
- The pipeline is *offline*: it reads a recorded ``source/raw/<session>.mkv``
  via OpenCV (FFV1 intra-frame lossless decodes fine) and emits normalized
  frames. It never requires the live UVC device.
- Scene classification is a *soft* heuristic used only to *tag* frames and to
  optionnally exclude non-game frames (phone home screen / register page) from
  the calibration set. It never invents labels: the frame manifest records the
  observed scene verbatim, and ``--exclude-nongame`` drops the known-bad
  categories instead of relabeling them as a poker table.
- Perceptual dedup uses a dHash (difference hash) hamming distance plus a
  normalized mean-absolute-difference check. Two sampled frames are "near
  duplicate stable" only when BOTH agree; this conservatively keeps timing
  frames around transitions.

Privacy note (guide rule 4): normalized frames and the manifest live in the
private calibration dataset and must never enter Git or a public upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .dataset import FrameEntry, frame_filename, write_frame_manifest
from .hashing import sha256_bytes
from .schema import Scene, SchemaError

# Standard frame sampling cadence for stable states (milliseconds).
DEFAULT_STABLE_INTERVAL_MS = 700
# A frame is "stable" when its dHash hamming distance to the previous kept
# frame is this score or lower.
STABLE_DHASH_THRESHOLD = 10
# And when the normalized mean absolute difference is this low.
STABLE_PIXEL_THRESHOLD = 3.0
# dHash resolution (width x height); 8x8 yields a 64-bit hash.
DHASH_SIZE = 8
# Minimum green-felt ratio that marks a frame as an actual poker table.
GREEN_TABLE_MIN = 0.10
# A frame is "black" (signal loss / lock screen etc) when its mean is this low.
BLACK_MEAN_MAX = 4.0


@dataclass
class SampleOptions:
    """Tunable behavior for :func:`sample_session`."""

    stable_interval_ms: int = DEFAULT_STABLE_INTERVAL_MS
    dhash_threshold: int = STABLE_DHASH_THRESHOLD
    pixel_threshold: float = STABLE_PIXEL_THRESHOLD
    green_table_min: float = GREEN_TABLE_MIN
    black_mean_max: float = BLACK_MEAN_MAX
    # Keep at least one frame on each side of a signal event (start/reconnect).
    event_context_frames: int = 2
    exclude_nongame: bool = False


@dataclass
class SampledFrame:
    """One emitted, normalized frame plus its manifest metadata."""

    file: str
    sha256: str
    timestamp_ms: int
    source_frame: int
    normalization_version: str
    stable: bool
    scene: Scene
    group_id: str
    reason: str
    image: np.ndarray | None = field(default=None, repr=False)


class _FrameReader:
    """Reads frames from a video file and reports per-frame timing.

    The reader stays a thin adapter (a small callable is injected in tests) so
    the sampling logic can be exercised without a real capture file.
    """

    def __init__(
        self, read_cb: Callable[[int], "tuple[bool, np.ndarray | None, float]"]
    ) -> None:
        self._read = read_cb

    def __call__(
        self, frame_index: int
    ) -> "tuple[bool, np.ndarray | None, float]":
        return self._read(frame_index)


def default_reader(
    path: Path,
    *,
    max_seconds: float | None = None,
    framerate: float = 30.0,
) -> _FrameReader:
    """Build a sequential reader over ``path`` using OpenCV.

    Frames are read in order (never seeked), which is much faster for FFV1
    intra-frame streams that have no keyframe acceleration. ``ts_ms`` is
    derived from the frame index and the known framerate, so timestamps are
    deterministic and reproducible (the guide's timestamps are relative to
    the video, not wall-clock).

    Because sampling may skip frames, the reader buffers the *next* frame and
    advances sequentially until it reaches the requested index. This keeps a
    single forward pass with no seeks.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SchemaError(f"cannot open video {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or framerate
    max_frames = None
    if max_seconds is not None:
        max_frames = int(max_seconds * fps)
    if max_frames is None or max_frames <= 0:
        max_frames = total

    state = {
        "next_index": 0,
        "next_frame": None,
        "done": False,
    }

    def _peek():
        if state["done"]:
            return None
        if state["next_frame"] is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                state["done"] = True
                state["next_frame"] = None
                return None
            state["next_frame"] = frame
        return state["next_frame"]

    def _read(frame_index: int):
        if state["done"]:
            return False, None, float(min(frame_index, total) / fps * 1000.0)
        # Advance sequentially to frame_index (forward only).
        while state["next_index"] < frame_index:
            state["next_index"] += 1
            state["next_frame"] = None
            # Re-read into the buffer; EOF ends the pass.
            frame = _peek()
            if frame is None:
                state["done"] = True
                return False, None, float(frame_index / fps * 1000.0)
        frame = _peek()
        if frame is None:
            state["done"] = True
            return False, None, float(frame_index / fps * 1000.0)
        # Consume the frame at this index.
        state["next_index"] = frame_index + 1
        state["next_frame"] = None
        return True, frame, float(frame_index / fps * 1000.0)

    reader = _FrameReader(_read)
    reader.max_frames = max_frames  # type: ignore[attr-defined]
    reader.total_frames = total  # type: ignore[attr-defined]
    reader.fps = fps  # type: ignore[attr-defined]
    return reader


def _dhash(image: np.ndarray) -> np.ndarray:
    """64-bit difference hash of the (downscaled) grayscale frame."""
    small = cv2.resize(
        image, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return np.packbits(diff.reshape(-1))


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.bitwise_xor(a, b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    diff = cv2.absdiff(a, b)
    return float(diff.mean())


def classify_scene(gray: np.ndarray, color: np.ndarray, *, green_min: float) -> Scene:
    """Classify a normalized frame into a Scene by visual cues.

    This is a soft, deterministic heuristic used for *tagging* and
    exclusions only. It returns one of the schema ``Scene`` values, never
    inventing a new category.
    """
    # Signal loss = black.
    if float(gray.mean()) < BLACK_MEAN_MAX:
        return Scene.SIGNAL_LOSS
    # Green felt ratio decides table vs non-game.
    b, g, r = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    green = (g > 90) & (r < g - 30) & (b < g)
    ratio = float(green.mean())
    if ratio >= green_min:
        # A table with a hand in flight (5 board cards visible) or an action
        # marker is classified as table; the coarse distinction between
        # deal/action/result is refined by downstream labeling, not here.
        return Scene.TABLE
    # Otherwise it's a menu / overlay / non-game screen.
    return Scene.MENU


def _is_table(gray: np.ndarray, color: np.ndarray, green_min: float) -> bool:
    b, g, r = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    green = (g > 90) & (r < g - 30) & (b < g)
    return float(green.mean()) >= green_min


def _is_black(gray: np.ndarray, black_max: float) -> bool:
    return float(gray.mean()) < black_max


def sample_session(
    root: Path | str,
    session_id: str,
    *,
    normalization_config: Any,
    options: SampleOptions | None = None,
    reader: _FrameReader | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Sample a session into normalized frames + a manifest.

    Args:
        root: the private calibration dataset root.
        session_id: the session id (e.g. ``session_001``), also the source
            video id and the file stub.
        normalization_config: a :class:`NormalizationConfig` (stage C).
        options: sampling/classification tuning.
        reader: an injected frame reader (tests stub the real MKV).
        on_progress: optional callback ``(done, total)``.

    Returns:
        ``(frames_written, manifest_entries)`` as ints.

    Raises:
        SchemaError: if the video cannot be opened, or the manifest cannot be
            written.
    """
    root = Path(root)
    options = options or SampleOptions()
    raw = root / "source" / "raw" / f"{session_id}.mkv"
    if reader is None:
        reader = default_reader(raw)

    frames_dir = root / "normalized" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "normalized" / "manifest.json"

    # Start clean so the manifest and the frames on disk stay one-to-one:
    # a re-run replaces the whole normalized frame set for this session.
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    max_frames = int(getattr(reader, "max_frames", 0) or 0)

    # --- pass 1: walk the video at the stable cadence, classify + dedup -----
    entries: list[FrameEntry] = []
    written: list[SampledFrame] = []
    last_kept: tuple[np.ndarray, np.ndarray] | None = None  # (dhash, normalized bgr)
    last_timestamp_ms: int = -1
    step = max(1, int(options.stable_interval_ms * 0.001 * 30.0))

    frame_index = 0
    processed = 0
    while True:
        if max_frames and frame_index >= max_frames:
            break
        ok, frame, ts_ms = reader(frame_index)
        processed += 1
        if not ok or frame is None:
            break
        try:
            normalized = _normalize_frame(frame, normalization_config)
        except Exception:
            # A frame outside the expected canvas is a page/screen we should
            # not decode into the table set; skip it.
            frame_index += step
            continue
        norm_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        is_table = _is_table(norm_gray, normalized, options.green_table_min)
        is_black = _is_black(norm_gray, options.black_mean_max)
        scene = classify_scene(norm_gray, normalized, green_min=options.green_table_min)

        # Perceptual dedup against the last *kept* stable frame.
        dhash = _dhash(normalized)
        near_dup = False
        if last_kept is not None:
            dist = _hamming(last_kept[0], dhash)
            mae = _mae(last_kept[1], normalized)
            near_dup = (
                dist <= options.dhash_threshold
                and mae <= options.pixel_threshold
            )

        # Decide whether to keep. Stable cadence keeps a frame unless it is a
        # near-duplicate; black frames and non-game frames are only kept when
        # the operator opted into preserving them via the manifest reason.
        keep = False
        reason = ""
        if is_black:
            reason = "signal_loss"
            keep = not options.exclude_nongame
            scene = Scene.SIGNAL_LOSS
        elif not is_table:
            reason = "menu_screen"
            keep = not options.exclude_nongame
            scene = Scene.MENU
        elif near_dup:
            reason = "stable_near_dup_skipped"
            keep = False
        else:
            # A real table frame, either a fresh stable state or a transition.
            delta_ms = ts_ms - last_timestamp_ms
            if last_kept is None or delta_ms > options.stable_interval_ms:
                reason = "stable_table"
            else:
                reason = "table_transition"
            keep = True

        if keep:
            digest = sha256_bytes(
                cv2.imencode(".png", normalized)[1].tobytes()
            )
            fname = frame_filename(session_id, int(ts_ms), frame_index, digest)
            group_id = _group_id(session_id, frame_index, scene)
            stable = not near_dup and not is_black
            entry = FrameEntry(
                file=fname,
                sha256=digest,
                source_video_id=session_id,
                timestamp_ms=int(ts_ms),
                source_frame=frame_index,
                normalization_version=normalization_config.version,
                stable=stable,
                scene=scene,
                group_id=group_id,
                reason=reason,
            )
            entries.append(entry)
            written.append(
                SampledFrame(
                    file=fname,
                    sha256=digest,
                    timestamp_ms=int(ts_ms),
                    source_frame=frame_index,
                    normalization_version=normalization_config.version,
                    stable=stable,
                    scene=scene,
                    group_id=group_id,
                    reason=reason,
                    image=normalized,
                )
            )
            last_kept = (dhash, normalized)
            last_timestamp_ms = ts_ms

        frame_index += step
        if on_progress and max_frames:
            on_progress(frame_index, max_frames)

    # --- pass 2: write PNGs (unicode-safe) then the manifest ----------------
    for item in written:
        ok, enc = cv2.imencode(".png", item.image)
        if not ok:
            raise SchemaError(f"failed to encode frame {item.file}")
        (frames_dir / item.file).write_bytes(enc.tobytes())

    write_frame_manifest(manifest_path, entries)

    # Frame manifest must round-trip through the reader, so we validate here.
    from .dataset import read_frame_manifest

    roundtrip = read_frame_manifest(manifest_path)
    if len(roundtrip) != len(entries):
        raise SchemaError("frame manifest did not round-trip")

    return len(written), len(entries)


def _normalize_frame(frame: np.ndarray, config: Any) -> np.ndarray:
    from poker_engine.perceptual.capture.normalization import normalize

    return normalize(frame, config)


def _group_id(session_id: str, frame_index: int, scene: Scene) -> str:
    return f"{session_id}__scene_{scene.value}"


__all__ = [
    "SampledFrame",
    "SampleOptions",
    "classify_scene",
    "default_reader",
    "sample_session",
]
