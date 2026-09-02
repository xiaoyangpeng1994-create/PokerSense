"""Real Windows capture backend (mss + ctypes user32).

Implements the standard visible-window capture path:
- DPI awareness initialization (per-monitor aware).
- HWND lookup by a stable window title / id.
- IsWindow / IsWindowVisible / IsIconic checks.
- GetClientRect + ClientToScreen -> physical-pixel client-area WindowRect.
- mss grab -> Frame.

Minimized / closed / invisible windows raise CaptureError (no fullscreen
fallback). Partial occlusion is a documented limitation (mss is a desktop-pixel
capture and cannot reliably detect occlusion).

This backend runs on a real Windows host. CI uses FakeBackend; actual
Windows smoke evidence is recorded in ``manual-smoke-task6.md``.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone

import numpy as np

from .base import CaptureError, CaptureTarget, CaptureService, Frame, WindowRect

try:
    import mss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mss = None

# --- user32 bindings -------------------------------------------------------

_HAS_WIN32 = True
try:
    # use_last_error=True so ctypes.get_last_error() / set_last_error() reliably
    # reflect the thread's last Win32 error (not a private copy).
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
except (AttributeError, OSError):  # pragma: no cover - non-Windows / sandbox
    _HAS_WIN32 = False
    _user32 = None

_ERROR_ACCESS_DENIED = 5

# Per-monitor aware v2 context handle (declared in windows.h). Value is stable.
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

# GetAwarenessFromDpiAwarenessContext returns one of these ints.
_DPI_AWARENESS_UNAWARE = 0
_DPI_AWARENESS_SYSTEM_AWARE = 1
_DPI_AWARENESS_PER_MONITOR_AWARE = 2


def _is_current_context_per_monitor() -> bool:
    """Query the thread's actual DPI awareness; True iff PER_MONITOR aware.

    Uses GetThreadDpiAwarenessContext + GetAwarenessFromDpiAwarenessContext.
    Returns False when the APIs are unavailable (cannot prove per-monitor).
    """
    if not _HAS_WIN32:  # pragma: no cover - non-Windows
        return False
    try:
        get_ctx = _user32.GetThreadDpiAwarenessContext
        get_ctx.argtypes = []
        get_ctx.restype = ctypes.c_void_p
        ctx = get_ctx()

        get_awareness = _user32.GetAwarenessFromDpiAwarenessContext
        get_awareness.argtypes = [ctypes.c_void_p]
        get_awareness.restype = ctypes.c_int
        awareness = get_awareness(ctx)
        return awareness == _DPI_AWARENESS_PER_MONITOR_AWARE
    except (AttributeError, OSError):  # pragma: no cover - older Windows
        return False


def _try_set_thread_dpi_awareness() -> bool:
    """Override the current worker thread to Per-Monitor V2 awareness.

    A packaged WebView host may establish a process-wide DPI mode before the
    capture backend starts. Windows 10's mixed-mode DPI API still permits the
    capture worker itself to use Per-Monitor V2 coordinates, which is exactly
    the scope needed by ClientToScreen and mss.
    """
    if not _HAS_WIN32:  # pragma: no cover - non-Windows
        return False
    try:
        fn = _user32.SetThreadDpiAwarenessContext
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_void_p
        previous = fn(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        return bool(previous) and _is_current_context_per_monitor()
    except (AttributeError, OSError):  # pragma: no cover - older Windows
        return False


def _try_set_dpi_awareness() -> str:
    """Declare per-monitor DPI awareness (v2 preferred), returning status.

    Return values:
      - "v2"                       -> SetProcessDpiAwarenessContext(v2) succeeded.
      - "already_set_per_monitor"  -> already configured AND verified per-monitor.
      - "fallback"                 -> set via legacy SetProcessDpiAwareness(2).
      - "thread_v2"                -> current worker overridden to Per-Monitor V2.
      - "none"                     -> genuine failure or cannot prove per-monitor.
    """
    if not _HAS_WIN32:  # pragma: no cover - non-Windows
        return "none"

    # Modern API: SetProcessDpiAwarenessContext (Win10 1703+).
    try:
        fn = _user32.SetProcessDpiAwarenessContext
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        ok = fn(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        if ok:
            return "v2"
        err = ctypes.get_last_error()
        if err == _ERROR_ACCESS_DENIED:
            # Already set by manifest / earlier call. Must verify the ACTUAL
            # awareness is per-monitor — ACCESS_DENIED alone is not enough.
            if _is_current_context_per_monitor():
                return "already_set_per_monitor"
            # A WebView host can legitimately establish a different process
            # default first. Continue to the thread-level mixed-DPI fallback.
        # Any other error is a genuine failure: do NOT silently succeed.
    except (AttributeError, OSError):  # pragma: no cover - older Windows
        pass

    # Legacy API: SetProcessDpiAwareness (Win8.1+), HRESULT return.
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.HRESULT
        hr = shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        hr_val = hr & 0xFFFFFFFF  # unsigned view of the HRESULT
        if hr_val == 0:  # S_OK
            return "fallback"
        if hr_val == 0x80070005:  # E_ACCESSDENIED -> already set, verify actual
            if _is_current_context_per_monitor():
                return "already_set_per_monitor"
    except (AttributeError, OSError):  # pragma: no cover - older Windows
        pass

    if _try_set_thread_dpi_awareness():
        return "thread_v2"
    return "none"


def _normalized_title(title: str) -> str:
    """Normalize cosmetic title differences without weakening identity."""
    return " ".join(title.split()).casefold()


def _window_title_matches(actual: str, requested: str) -> bool:
    """Match a page title with an optional browser/app suffix.

    Browsers expose titles such as ``WePoker-H5 - Google Chrome`` while the
    page's stable identity is ``WePoker-H5``. Accept common title separators,
    but never an arbitrary substring match that could select another page.
    """
    actual_normalized = _normalized_title(actual)
    requested_normalized = _normalized_title(requested)
    if actual_normalized == requested_normalized:
        return True
    return any(
        actual_normalized.startswith(f"{requested_normalized}{separator}")
        for separator in (" - ", " – ", " — ", " | ")
    )


def _find_hwnd_matches(title: str) -> list[int]:
    """Return visible HWNDs matching a stable page/application title."""
    matches: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd, lparam):
        if _user32.IsWindow(hwnd) and _user32.IsWindowVisible(hwnd):
            length = _user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                _user32.GetWindowTextW(hwnd, buf, length + 1)
                if _window_title_matches(buf.value, title):
                    matches.append(hwnd)
        return True

    _user32.EnumWindows(_enum_proc, 0)
    return matches


class MssBackend(CaptureService):
    """mss-based Windows visible-window capture.

    Resolves a ``CaptureTarget.window_id`` (stable window title/identifier) to a
    HWND, validates it is visible + non-minimized, computes its physical-pixel
    client-area rect, and captures it via mss.
    """

    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError(
                "mss is not installed; install with the perceptual extra"
            )
        if not _HAS_WIN32:
            raise RuntimeError("MssBackend requires Windows (user32 unavailable)")
        super().__init__()
        self._dpi_status = _try_set_dpi_awareness()
        if self._dpi_status == "none":
            # We cannot guarantee physical-pixel correctness without DPI
            # awareness. Fail fast rather than silently capture misaligned pixels.
            raise RuntimeError(
                "DPI awareness could not be established; physical-pixel "
                "correctness is not guaranteed"
            )
        self._sct = mss.mss()

    @property
    def dpi_status(self) -> str:
        """Exposed DPI-awareness result for verifiability / smoke reports."""
        return self._dpi_status

    def _primary_display_rect(self) -> WindowRect:
        monitors = self._sct.monitors
        if len(monitors) < 2:
            raise CaptureError("primary display is unavailable for capture")
        primary = monitors[1]
        return WindowRect(
            left=int(primary["left"]),
            top=int(primary["top"]),
            width=int(primary["width"]),
            height=int(primary["height"]),
        )

    def _resolve_window_rect(self, target: CaptureTarget) -> WindowRect:
        window_id = target.window_id
        matches = _find_hwnd_matches(window_id)

        if len(matches) == 0:
            if target.allow_fullscreen_fallback:
                return self._primary_display_rect()
            raise CaptureError(f"target window {window_id!r} not found or closed")
        if len(matches) > 1 and target.window_index is None:
            raise CaptureError(
                f"target window {window_id!r} is ambiguous ({len(matches)} matches); "
                "set window_index explicitly"
            )
        if target.window_index is not None:
            if target.window_index >= len(matches):
                raise CaptureError(
                    f"window_index {target.window_index} is out of range for "
                    f"{window_id!r} ({len(matches)} visible matches)"
                )
            hwnd = matches[target.window_index]
        else:
            hwnd = matches[0]
        if not _user32.IsWindow(hwnd):
            raise CaptureError(f"target window {window_id!r} is closed")
        if not _user32.IsWindowVisible(hwnd):
            raise CaptureError(f"target window {window_id!r} is not visible")
        if _user32.IsIconic(hwnd):
            raise CaptureError(f"target window {window_id!r} is minimized")

        # Client-area rect in client coordinates.
        rect = wintypes.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise CaptureError(
                f"failed to read client rect for {window_id!r}"
            )

        # Convert client (0,0) top-left to screen coordinates.
        pt = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            raise CaptureError(
                f"failed to convert client coords for {window_id!r}"
            )

        left = int(pt.x)
        top = int(pt.y)
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)

        if width <= 0 or height <= 0:
            raise CaptureError(
                f"target window {window_id!r} has invalid client rect"
            )
        # left/top may be negative on multi-monitor setups (virtual desktop).
        return WindowRect(left=left, top=top, width=width, height=height)

    def capture(self, target: CaptureTarget) -> Frame:
        if not isinstance(target, CaptureTarget):
            raise TypeError("target must be a CaptureTarget")

        # asyncio.to_thread may schedule successive frames on different pool
        # threads. DPI awareness is a thread property in a mixed-mode WebView
        # process, so establish it on the thread doing this exact capture.
        if (
            not _is_current_context_per_monitor()
            and not _try_set_thread_dpi_awareness()
        ):
            raise CaptureError(
                "DPI awareness could not be established for the capture thread"
            )

        rect = self._resolve_window_rect(target)
        monitor = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        raw = self._sct.grab(monitor)
        # mss returns BGRA; drop alpha -> BGR.
        img = np.asarray(raw)[:, :, :3]
        ts = datetime.now(timezone.utc)
        return Frame(
            frame_seq=self._next_seq(),
            timestamp=ts,
            window_id=target.window_id,
            window_rect=rect,
            image=img,
            width=rect.width,
            height=rect.height,
        )


__all__ = ["MssBackend"]
