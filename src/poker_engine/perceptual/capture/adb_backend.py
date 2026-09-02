"""ADB capture backend for Android emulators and attached devices.

The production PokerSense target is WePoker running in a portrait LDPlayer
instance.  ``adb exec-out screencap -p`` returns the emulator's client pixels
directly, so recognition does not depend on the host window position, DPI,
decorations, occlusion, or desktop scaling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

from .base import CaptureError, CaptureService, CaptureTarget, Frame, WindowRect


_Run = Callable[..., subprocess.CompletedProcess]


class AdbBackend(CaptureService):
    """Capture PNG frames from one explicitly selected ADB device.

    ``CaptureTarget.window_id`` carries the ADB serial for this backend.  The
    special value ``"auto"`` is accepted only when exactly one authorized
    device is connected.  Ambiguous multi-instance setups fail closed and
    list the serials the user can select.
    """

    def __init__(
        self,
        adb_path: str | None = None,
        timeout_seconds: float = 5.0,
        runner: _Run = subprocess.run,
    ) -> None:
        super().__init__()
        configured = adb_path or os.environ.get("POKERSENSE_ADB_PATH")
        resolved = configured or shutil.which("adb")
        if not resolved:
            raise RuntimeError(
                "ADB was not found; set POKERSENSE_ADB_PATH to LDPlayer's "
                "adb.exe or add adb to PATH"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._adb_path = resolved
        self._timeout_seconds = float(timeout_seconds)
        self._runner = runner

    def _run(self, args: list[str], *, binary: bool = False):
        try:
            return self._runner(
                [self._adb_path, *args],
                capture_output=True,
                text=not binary,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureError(
                f"ADB timed out after {self._timeout_seconds:g}s"
            ) from exc
        except OSError as exc:
            raise CaptureError(f"could not run ADB: {exc}") from exc

    def list_devices(self) -> tuple[str, ...]:
        """Return authorized device serials reported by ``adb devices``."""
        result = self._run(["devices"])
        if result.returncode != 0:
            detail = (result.stderr or "ADB devices failed").strip()
            raise CaptureError(detail)
        devices = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return tuple(devices)

    def _resolve_serial(self, requested: str) -> str:
        devices = self.list_devices()
        if requested != "auto":
            if requested not in devices:
                found = ", ".join(devices) if devices else "none"
                raise CaptureError(
                    f"ADB device {requested!r} is not available; "
                    f"authorized devices: {found}"
                )
            return requested
        if len(devices) == 1:
            return devices[0]
        if not devices:
            raise CaptureError(
                "no authorized ADB device found; start LDPlayer and enable ADB"
            )
        raise CaptureError(
            "multiple ADB devices found; select one with --device-serial: "
            + ", ".join(devices)
        )

    def capture(self, target: CaptureTarget) -> Frame:
        if target.window_index is not None or target.allow_fullscreen_fallback:
            raise CaptureError(
                "window_index/fullscreen fallback do not apply to ADB capture"
            )
        serial = self._resolve_serial(target.window_id)
        result = self._run(
            ["-s", serial, "exec-out", "screencap", "-p"], binary=True
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if result.stderr else ""
            raise CaptureError(
                (stderr.strip() or f"ADB screenshot failed for {serial}")
            )
        payload = result.stdout or b""
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise CaptureError(f"ADB returned an invalid PNG for {serial}")
        height, width = image.shape[:2]
        return Frame(
            frame_seq=self._next_seq(),
            timestamp=datetime.now(timezone.utc),
            window_id=serial,
            window_rect=WindowRect(0, 0, width, height),
            image=image,
            width=width,
            height=height,
        )


__all__ = ["AdbBackend"]
