"""Real macOS capture backend (Quartz/CoreGraphics window capture).

Implements the standard visible-window capture path:
- Window lookup by a stable window title / id (``kCGWindowName``), matched
  against the full window list to distinguish "not found" from "found but
  not on-screen" (minimized / hidden).
- ``CGWindowListCreateImage`` targeted at the resolved window id, which
  captures that window's own composited content (not a blind screen-rect
  grab) and automatically returns full physical-pixel resolution on Retina
  displays -- no manual DPI-scale-factor arithmetic needed (unlike the
  Windows backend, which has to negotiate DPI awareness with user32).
- Pixel format is verified against the expected bitmap layout before use;
  an unexpected layout fails fast rather than silently producing
  wrong-channel-order pixels.

Minimized / closed / not-on-screen windows raise ``CaptureError`` (no
fullscreen fallback), mirroring ``MssBackend``. Occlusion by another window
is a documented limitation shared with the Windows backend: capturing by
window id still only returns the on-screen-visible portion of the window.

This backend requires macOS + ``pyobjc-framework-Quartz`` (the ``perceptual``
extra) and, on first use, the "Screen Recording" privacy permission -- there
is no Windows equivalent of that prompt, and it must be granted once per
signed binary identity (an unsigned dev script and a packaged app are
treated as different identities by macOS).
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np

from .base import CaptureError, CaptureTarget, CaptureService, Frame, WindowRect

try:
    import Quartz  # type: ignore

    _HAS_QUARTZ = True
except ImportError:  # pragma: no cover - non-macOS / pyobjc not installed
    Quartz = None
    _HAS_QUARTZ = False

# Expected bitmap layout for window-capture images: 8bpc, alpha-first,
# premultiplied, little-endian byte order -> raw bytes come out as BGRA.
# Verified empirically against a real CGWindowListCreateImage capture on
# this codebase's development machine (not assumed from documentation).
_EXPECTED_BITMAP_INFO = None
if _HAS_QUARTZ:
    _EXPECTED_BITMAP_INFO = (
        Quartz.kCGImageAlphaPremultipliedFirst | Quartz.kCGBitmapByteOrder32Little
    )


def _list_windows(*, on_screen_only: bool) -> list[dict]:
    options = Quartz.kCGWindowListExcludeDesktopElements
    if on_screen_only:
        options |= Quartz.kCGWindowListOptionOnScreenOnly
    info = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    return list(info) if info else []


def _find_window_matches(title: str) -> tuple[list[dict], list[dict]]:
    """Return (all matches by title, on-screen-visible matches by title)."""
    all_windows = _list_windows(on_screen_only=False)
    all_matches = [w for w in all_windows if w.get("kCGWindowName") == title]
    if not all_matches:
        return [], []
    onscreen = _list_windows(on_screen_only=True)
    onscreen_ids = {int(w["kCGWindowNumber"]) for w in onscreen}
    onscreen_matches = [
        w for w in all_matches if int(w["kCGWindowNumber"]) in onscreen_ids
    ]
    return all_matches, onscreen_matches


def has_screen_capture_permission() -> bool:
    """Return whether this exact app identity may inspect/capture windows.

    macOS assigns Screen Recording permission to the signed app identity, so
    a terminal running the source tree and the packaged PokerSense.app do not
    share it.  Without permission Quartz can return an incomplete window list
    that is indistinguishable from a missing title unless checked first.
    """
    preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
    return bool(preflight()) if preflight is not None else True


def request_screen_capture_permission() -> bool:
    """Request Screen Recording access for the running app, if needed."""
    if has_screen_capture_permission():
        return True
    request = getattr(Quartz, "CGRequestScreenCaptureAccess", None)
    if request is not None:
        request()
    return has_screen_capture_permission()


@dataclass(frozen=True)
class WindowCandidate:
    """A selectable, currently visible macOS window.

    ``index`` is only stable for the current enumeration.  It is intended as
    an explicit user choice when two same-titled windows are visible, not as a
    persistent window identifier.
    """

    index: int
    window_number: int
    owner_name: str
    left: int
    top: int
    width: int
    height: int


def list_window_candidates(title: str) -> tuple[WindowCandidate, ...]:
    """List visible candidates for a title in Quartz' current z-order."""
    _, matches = _find_window_matches(title)
    candidates = []
    for index, window in enumerate(matches):
        bounds = window.get("kCGWindowBounds", {})
        candidates.append(
            WindowCandidate(
                index=index,
                window_number=int(window["kCGWindowNumber"]),
                owner_name=str(window.get("kCGWindowOwnerName", "")),
                left=int(bounds.get("X", 0)),
                top=int(bounds.get("Y", 0)),
                width=int(bounds.get("Width", 0)),
                height=int(bounds.get("Height", 0)),
            )
        )
    return tuple(candidates)


def _cgimage_to_bgr_ndarray(image_ref) -> np.ndarray:
    bitmap_info = Quartz.CGImageGetBitmapInfo(image_ref)
    if bitmap_info != _EXPECTED_BITMAP_INFO:
        raise CaptureError(
            f"unexpected CGImage bitmap layout {bitmap_info!r} "
            f"(expected {_EXPECTED_BITMAP_INFO!r}); refusing to guess channel order"
        )

    width = int(Quartz.CGImageGetWidth(image_ref))
    height = int(Quartz.CGImageGetHeight(image_ref))
    bytes_per_row = int(Quartz.CGImageGetBytesPerRow(image_ref))
    provider = Quartz.CGImageGetDataProvider(image_ref)
    data = Quartz.CGDataProviderCopyData(provider)
    buf = bytes(data)

    expected_bytes = height * bytes_per_row
    if len(buf) < expected_bytes:
        raise CaptureError(
            f"CGImage pixel buffer is truncated ({len(buf)} < {expected_bytes})"
        )
    # CGDisplayCreateImage can append provider padding after the final scan
    # line. Only the documented height × bytes-per-row pixel region belongs
    # to this image; reshaping the entire provider buffer would otherwise
    # raise ValueError and leave the desktop UI stuck at "waiting".
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr = arr[:expected_bytes]
    arr = arr.reshape((height, bytes_per_row // 4, 4))
    arr = arr[:, :width, :3]  # BGRA (verified) -> drop alpha -> BGR
    return np.ascontiguousarray(arr)


class QuartzBackend(CaptureService):
    """Quartz-based macOS visible-window capture.

    Resolves a ``CaptureTarget.window_id`` (stable window title) to a
    ``kCGWindowNumber``, validates it is on-screen (not minimized/hidden),
    and captures its composited content via ``CGWindowListCreateImage``.
    """

    def __init__(self) -> None:
        if not _HAS_QUARTZ:
            raise RuntimeError(
                "QuartzBackend requires macOS (pyobjc Quartz unavailable); "
                "install with the perceptual extra on macOS"
            )
        super().__init__()

    def _resolve_window(self, target: CaptureTarget) -> tuple[int, WindowRect]:
        window_id = target.window_id
        if not has_screen_capture_permission():
            raise CaptureError(
                "Screen Recording permission is required for PokerSense. "
                "Enable PokerSense in System Settings > Privacy & Security > "
                "Screen Recording, then reopen the app."
            )
        all_matches, onscreen_matches = _find_window_matches(window_id)

        if not all_matches:
            raise CaptureError(f"target window {window_id!r} not found or closed")
        if not onscreen_matches:
            raise CaptureError(
                f"target window {window_id!r} is not on the active macOS Space "
                "or is minimized/hidden; switch to the Space containing the table"
            )
        if len(onscreen_matches) > 1 and target.window_index is None:
            raise CaptureError(
                f"target window {window_id!r} is ambiguous ({len(onscreen_matches)} "
                "visible matches); set window_index explicitly using "
                "tools/list_windows.py"
            )
        if target.window_index is not None:
            if target.window_index >= len(onscreen_matches):
                raise CaptureError(
                    f"window_index {target.window_index} is out of range for "
                    f"{window_id!r} ({len(onscreen_matches)} visible matches)"
                )
            w = onscreen_matches[target.window_index]
        else:
            w = onscreen_matches[0]
        if not w:
            raise CaptureError(
                f"target window {window_id!r} is minimized or not visible"
            )
        window_number = int(w["kCGWindowNumber"])
        bounds = w["kCGWindowBounds"]
        left = int(bounds["X"])
        top = int(bounds["Y"])
        width = int(bounds["Width"])
        height = int(bounds["Height"])
        if width <= 0 or height <= 0:
            raise CaptureError(f"target window {window_id!r} has invalid bounds")

        # WindowRect is reported in points (Quartz's native coordinate
        # space); the captured Frame's pixel buffer may be larger on
        # Retina displays (see capture() below) -- physical-pixel scale is
        # read back directly from the CGImage, never computed by hand.
        return window_number, WindowRect(left=left, top=top, width=width, height=height)

    def _capture_main_display(self, target: CaptureTarget) -> Frame:
        """Capture the full primary display for an explicitly opted-in target.

        Quartz can occasionally expose pixel capture but omit third-party
        window titles to a newly authorized packaged application.  WePoker's
        calibrated full-screen layout has the same aspect ratio as the
        primary display, so its live adapter may explicitly use this recovery
        path.  It is never a default for arbitrary titled windows.
        """
        display_id = Quartz.CGMainDisplayID()
        image_ref = Quartz.CGDisplayCreateImage(display_id)
        if image_ref is None:
            raise CaptureError(
                "failed to capture the primary display after Screen Recording "
                "permission was granted"
            )
        bounds = Quartz.CGDisplayBounds(display_id)
        rect = WindowRect(
            left=int(bounds.origin.x),
            top=int(bounds.origin.y),
            width=int(bounds.size.width),
            height=int(bounds.size.height),
        )
        img = _cgimage_to_bgr_ndarray(image_ref)
        height, width = img.shape[:2]
        return Frame(
            frame_seq=self._next_seq(),
            timestamp=datetime.now(timezone.utc),
            window_id=target.window_id,
            window_rect=rect,
            image=img,
            width=width,
            height=height,
        )

    def capture(self, target: CaptureTarget) -> Frame:
        if not isinstance(target, CaptureTarget):
            raise TypeError("target must be a CaptureTarget")

        try:
            window_number, rect = self._resolve_window(target)
        except CaptureError as exc:
            if (
                target.allow_fullscreen_fallback
                and "not found or closed" in str(exc)
            ):
                return self._capture_main_display(target)
            raise

        cg_rect = Quartz.CGRectMake(rect.left, rect.top, rect.width, rect.height)
        image_ref = Quartz.CGWindowListCreateImage(
            cg_rect,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_number,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )
        if image_ref is None:
            raise CaptureError(
                f"failed to capture image for {target.window_id!r} "
                "(window may have closed, or Screen Recording permission "
                "was not granted)"
            )

        img = _cgimage_to_bgr_ndarray(image_ref)
        height, width = img.shape[0], img.shape[1]
        ts = datetime.now(timezone.utc)
        return Frame(
            frame_seq=self._next_seq(),
            timestamp=ts,
            window_id=target.window_id,
            window_rect=rect,
            image=img,
            width=width,
            height=height,
        )


__all__ = [
    "QuartzBackend",
    "WindowCandidate",
    "has_screen_capture_permission",
    "list_window_candidates",
    "request_screen_capture_permission",
]
