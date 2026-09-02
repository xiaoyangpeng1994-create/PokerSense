"""Small, durable user preferences for the packaged desktop app."""

from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_LANGUAGE = "auto"
_ALLOWED_LANGUAGES = frozenset(("auto", "en", "zh"))


def _settings_path() -> Path:
    """Return a writable per-user path, never a path inside the app bundle."""
    override = os.environ.get("POKERSENSE_SETTINGS_PATH")
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif os.sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "PokerSense" / "settings.json"


def load_settings() -> dict[str, str]:
    """Load validated preferences; a missing/corrupt file means defaults."""
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"language": _DEFAULT_LANGUAGE}
    language = data.get("language") if isinstance(data, dict) else None
    selected = language if language in _ALLOWED_LANGUAGES else _DEFAULT_LANGUAGE
    return {"language": selected}


def save_language(language: str) -> dict[str, str]:
    """Atomically persist a supported display-language preference."""
    if language not in _ALLOWED_LANGUAGES:
        raise ValueError(f"unsupported language preference: {language!r}")
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"language": language}), encoding="utf-8")
    temporary.replace(path)
    return {"language": language}


__all__ = ["load_settings", "save_language"]
