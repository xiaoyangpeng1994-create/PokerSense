"""List selectable macOS windows for PokerSense live capture.

Run this while the poker table is on the active macOS Space:

    ./.venv/bin/python tools/list_windows.py --title WePoker-H5

Pass a reported index to ``run_desktop_server.py --window-index N`` when
more than one same-titled table is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poker_engine.perceptual.capture.quartz_backend import (  # noqa: E402
    list_window_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="List visible capture windows")
    parser.add_argument("--title", default="WePoker-H5", help="exact window title")
    args = parser.parse_args()

    candidates = list_window_candidates(args.title)
    if not candidates:
        print(
            f"No visible window titled {args.title!r}. Switch to its macOS Space "
            "and make sure it is not minimized."
        )
        return
    for candidate in candidates:
        print(
            f"index={candidate.index} window_number={candidate.window_number} "
            f"owner={candidate.owner_name!r} "
            f"bounds={candidate.left},{candidate.top} "
            f"{candidate.width}x{candidate.height}"
        )


if __name__ == "__main__":
    main()
