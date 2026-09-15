"""Cross-process AA device ownership, with fake sources and temporary locks."""

import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from poker_engine.desktop.aa_device_lock import AACaptureDeviceLock
from poker_engine.desktop.aa_sources import AACaptureSource


def lock(path):
    return AACaptureDeviceLock(path, legacy_lock_path=None)


def start_child(script, path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
           "PYTHONUTF8": "1"}
    return subprocess.Popen([sys.executable, "-c", script, str(path)], cwd=root,
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8")


@pytest.mark.parametrize("crash", [False, True])
def test_other_process_excluded_then_os_releases_on_exit(tmp_path, crash):
    path = tmp_path / "capture.lock"
    script = """
import os,sys
from poker_engine.desktop.aa_device_lock import AACaptureDeviceLock
owner=AACaptureDeviceLock(sys.argv[1],legacy_lock_path=None)
owner.acquire();print('HELD',flush=True);sys.stdin.readline()
""" + ("os._exit(0)" if crash else "owner.release()")
    child = start_child(script, path)
    try:
        assert child.stdout.readline().strip() == "HELD"
        with pytest.raises(RuntimeError, match="另一 AA"):
            lock(path).acquire()
        child.communicate("exit\n", timeout=10)
        assert child.returncode == 0
        next_owner = lock(path).acquire()
        next_owner.release()
        assert path.exists()  # Presence is not treated as stale ownership.
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


def test_existing_passive_recorder_marker_rejects_without_changing_it(tmp_path):
    legacy = tmp_path / "legacy.lock"
    legacy.write_text("external owner")
    owner = AACaptureDeviceLock(tmp_path / "aa.lock", legacy_lock_path=legacy)
    with pytest.raises(RuntimeError, match="旁观录制锁"):
        owner.acquire()
    assert legacy.read_text() == "external owner"
    assert not (tmp_path / "aa.lock").exists()


def test_source_does_not_lock_or_capture_until_first_read(tmp_path):
    path = tmp_path / "device.lock"
    captured = []

    class Backend:
        def __init__(self, **kwargs):
            pass

        def capture(self, target):
            captured.append(target)
            raise AssertionError("device ownership must precede capture")

        def release(self):
            pass

    source = AACaptureSource({}, backend_factory=Backend,
                             device_lock_factory=lambda: lock(path))
    competitor = lock(path).acquire()
    try:
        with pytest.raises(RuntimeError, match="另一 AA"):
            source.read()
        assert not captured
    finally:
        source.close()
        competitor.release()


def test_failed_open_keeps_lock_until_backend_cleanup_finishes(tmp_path):
    path = tmp_path / "device.lock"
    cleanup_started = threading.Event()
    cleanup_allowed = threading.Event()

    class Backend:
        def __init__(self, **kwargs):
            pass

        def capture(self, target):
            raise RuntimeError("failed opening fake device")

        def release(self):
            cleanup_started.set()
            assert cleanup_allowed.wait(5)

    source = AACaptureSource({}, backend_factory=Backend,
                             device_lock_factory=lambda: lock(path))
    try:
        with pytest.raises(RuntimeError, match="failed opening"):
            source.read()
        assert cleanup_started.wait(1)
        with pytest.raises(RuntimeError, match="另一 AA"):
            lock(path).acquire()
    finally:
        cleanup_allowed.set()
        source.close()
    next_owner = lock(path).acquire()
    next_owner.release()


def test_failed_cleanup_holds_lock_after_source_discard_until_process_exit(tmp_path):
    path = tmp_path / "device.lock"
    script = """
import gc,sys
from poker_engine.desktop.aa_device_lock import AACaptureDeviceLock
from poker_engine.desktop.aa_sources import AACaptureSource
class Backend:
 def __init__(self,**kwargs):pass
 def capture(self,target):raise RuntimeError('read failed')
 def release(self):raise RuntimeError('release failed')
source=AACaptureSource({},backend_factory=Backend,
 device_lock_factory=lambda:AACaptureDeviceLock(sys.argv[1],legacy_lock_path=None))
try:source.read()
except RuntimeError:pass
try:source.close()
except RuntimeError:pass
del source;gc.collect()
print('QUARANTINED',flush=True);sys.stdin.readline()
"""
    child = start_child(script, path)
    try:
        assert child.stdout.readline().strip() == "QUARANTINED"
        with pytest.raises(RuntimeError, match="另一 AA"):
            lock(path).acquire()
        child.communicate("exit\n", timeout=10)
        assert child.returncode == 0
        next_owner = lock(path).acquire()
        next_owner.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
