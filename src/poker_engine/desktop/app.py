"""Desktop app entry point: run the local server + open a native window.

    python -m poker_engine.desktop.app

Starts the FastAPI server on a background thread and opens a companion UI.
Table pixels default to the WPK phone capture card. ADB remains an explicit
alternative source.
"""

from __future__ import annotations

import argparse
import threading
import time

import uvicorn

from .live import DEFAULT_DEVICE_SERIAL, live_analysis_stream
from .server import CAPTURE_SOURCES, create_app

HOST = "127.0.0.1"
PORT = 8765
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 520
SERVER_STARTUP_TIMEOUT_SECONDS = 10.0


def _create_server(device_serial: str, *, source: str = "capture-card",
                   device_index: int = 0, api: str = "MSMF") -> uvicorn.Server:
    def stream():
        return live_analysis_stream(device_serial, source=source,
                                    device_index=device_index, api=api)

    config = uvicorn.Config(
        create_app(stream), host=HOST, port=PORT, log_level="warning"
    )
    return uvicorn.Server(config)


def _wait_for_server(
    server: uvicorn.Server,
    server_thread: threading.Thread,
    timeout_seconds: float = SERVER_STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Wait until uvicorn is listening before navigating the native webview."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.started:
            return
        if not server_thread.is_alive():
            raise RuntimeError("PokerSense local server stopped during startup")
        time.sleep(0.02)
    raise RuntimeError("PokerSense local server did not start within 10 seconds")


def main(device_serial: str = DEFAULT_DEVICE_SERIAL, *, source="capture-card",
         device_index=0, api="MSMF") -> None:
    import webview

    server = _create_server(device_serial, source=source,
                            device_index=device_index, api=api)
    server_thread = threading.Thread(
        target=server.run, daemon=True
    )
    server_thread.start()
    _wait_for_server(server, server_thread)

    webview.create_window(
        "PokerSense",
        url=f"http://{HOST}:{PORT}/",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
        on_top=False,
    )
    webview.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open the PokerSense companion app")
    parser.add_argument(
        "--device-serial",
        default=DEFAULT_DEVICE_SERIAL,
        help="ADB serial from `adb devices`; auto is allowed for one device",
    )
    parser.add_argument("--source", choices=CAPTURE_SOURCES, default="capture-card")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--api", choices=("MSMF", "DSHOW", "ANY"), default="MSMF")
    args = parser.parse_args()
    main(device_serial=args.device_serial, source=args.source,
         device_index=args.device_index, api=args.api)
