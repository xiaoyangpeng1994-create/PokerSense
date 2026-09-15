"""Explicit AA image sources; constructing the application never opens hardware."""

import json
from pathlib import Path
import threading
import time

from poker_engine.perceptual.capture.base import CaptureTarget
from poker_engine.perceptual.capture.capture_card_backend import CaptureCardBackend
from poker_engine.perceptual.capture.normalization import NormalizationConfig


class AACaptureSource:
    def __init__(self, options, *, backend_factory=CaptureCardBackend):
        index = options.get("device_index", 0)
        api = options.get("api", "MSMF")
        if type(index) is not int or not 0 <= index <= 20:
            raise ValueError("设备编号必须为 0–20 的整数")
        if api not in {"MSMF", "DSHOW"}:
            raise ValueError("采集接口必须为 MSMF 或 DSHOW")
        self.backend = backend_factory(
            device_index=index, api=api, width=1920, height=1080, fps=30,
            normalization=NormalizationConfig(
                rotate_degrees=0, source_size=(1920, 1080),
                crop_after_rotation=(711, 0, 1209, 1080),
                output_size=(498, 1080), version="aa8-capture-canvas-v1"),
        )
        self.target = CaptureTarget(f"uvc-{index}")
        self.started = time.monotonic()
        self.condition = threading.Condition()
        self.cancel = threading.Event()
        self.thread = None
        self.latest = None
        self.error = None
        self.delivered = None

    def _pump(self):
        try:
            while not self.cancel.is_set():
                frame = self.backend.capture(self.target)
                with self.condition:
                    self.latest = {
                        "image": frame.image, "source_frame": frame.frame_seq,
                        "pts_seconds": time.monotonic() - self.started,
                        "source_kind": "capture-card"}
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.latest = None
                self.condition.notify_all()
        finally:
            self.backend.release()

    def read(self):
        with self.condition:
            if self.cancel.is_set():
                return None
            if self.thread is None:
                self.thread = threading.Thread(target=self._pump, daemon=True,
                                               name="aa-latest-capture-frame")
                self.thread.start()
            ready = self.condition.wait_for(
                lambda: self.error is not None or self.cancel.is_set()
                or self.latest is not None and self.latest["source_frame"] != (
                    self.delivered), timeout=2.0)
            if self.error is not None:
                raise self.error
            if self.cancel.is_set():
                return None
            if not ready:
                raise RuntimeError("采集卡两秒内未提供新帧")
            result = self.latest
            self.delivered = result["source_frame"]
            return result

    def close(self):
        self.cancel.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            # Runs in session worker, never the API thread. A blocked driver
            # leaves STOPPING visible and prevents a second device owner.
            self.thread.join()
        else:
            self.backend.release()


class AADevelopmentSource:
    """Only manifest-bound, contiguous development PNGs; no video/holdout path."""

    def __init__(self, pool, audit, *, first=None, last=None):
        from tools.aa8_action_transfer import inventory

        self.pool = Path(pool).resolve(strict=True)
        self.rows = inventory(self.pool, audit)
        keys = list(self.rows)
        first = keys[0] if first is None else first
        last = keys[-1] if last is None else last
        if (type(first) is not int or type(last) is not int or first > last
                or first not in self.rows or last not in self.rows):
            raise ValueError("开发回放区间必须属于已登记开发帧")
        self.frames = iter(range(first, last + 1))
        self.closed = False

    def read(self):
        from tools.aa8_action_transfer import load

        if self.closed:
            return None
        frame = next(self.frames, None)
        if frame is None:
            return None
        row = self.rows[frame]
        pts = row["pts_seconds"]
        if isinstance(pts, bool):
            raise ValueError("开发帧 PTS 必须为数值")
        return {"image": load(self.pool, row), "source_frame": frame,
                "pts_seconds": float(pts), "source_pts_exact": str(pts),
                "source_kind": "development-replay"}

    def close(self):
        self.closed = True


def source_factory(profile_path, *, replay_pool=None, replay_first=None,
                   replay_last=None, allow_capture=False):
    """Browser cannot supply a filesystem path, normalization or audit identity."""
    def create(options):
        mode = options.get("mode")
        if mode == "capture-card":
            if not allow_capture:
                raise ValueError("本次启动未启用采集卡；当前仅允许离线验证")
            return AACaptureSource(options)
        if mode == "development-replay" and replay_pool is not None:
            spec = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            return AADevelopmentSource(replay_pool, spec["audit"],
                                       first=replay_first, last=replay_last)
        raise ValueError("请选择本次启动已配置的来源")

    return create
