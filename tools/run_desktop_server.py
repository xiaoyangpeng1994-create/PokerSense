"""Dev entry point: run the desktop server without an editable install.

    ./.venv/bin/python tools/run_desktop_server.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poker_engine.desktop.live import DEFAULT_DEVICE_SERIAL  # noqa: E402
from poker_engine.desktop.server import run  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PokerSense local server")
    parser.add_argument("--device-serial", default=DEFAULT_DEVICE_SERIAL)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(port=args.port, device_serial=args.device_serial)
