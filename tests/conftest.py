"""Shared pytest setup for the PokerSense test suite.

Installs a ``cv2.imread`` shim so fixture loading survives non-ASCII paths.
"""

import os

import cv2
import numpy as np


_ORIGINAL_IMREAD = cv2.imread


def _imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    """``cv2.imread`` that also works when the path contains non-ASCII text.

    On Windows ``cv2.imread`` silently returns ``None`` (after logging a
    decoder warning) for an absolute path containing non-ASCII characters --
    which describes every path in this checkout
    (``...\\WorkBuddy\\扑克\\PokerSense\\...``). Reading the bytes ourselves
    and decoding them with ``cv2.imdecode`` bypasses the filesystem-encoding
    path that breaks.

    ASCII paths keep the native fast path so the shim stays quiet and free.
    Installing it here (instead of editing every call site) keeps tests
    readable and makes new tests immune to the same trap for free.
    """
    try:
        path = os.fspath(filename)
    except TypeError:
        return _ORIGINAL_IMREAD(filename, flags)
    if not isinstance(path, str) or path.isascii():
        return _ORIGINAL_IMREAD(filename, flags)
    try:
        with open(path, "rb") as handle:
            data = np.frombuffer(handle.read(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


cv2.imread = _imread_unicode
