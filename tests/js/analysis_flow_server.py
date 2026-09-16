"""Dedicated AA server for the shipped-JS analysis-flow harness.

Run as its own process: the conditional-analysis worker uses the ``spawn`` start
method, which re-imports ``__main__`` — a real module-level ``__main__`` guard
(not the test runner's) keeps that deterministic, and the worker's 10 s deadline
never shares a process with the test.

Usage: python analysis_flow_server.py --work <dir> --port <port>
Prints one JSON line {"base": "http://127.0.0.1:<port>"} once it is listening.
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="AA analysis-flow test server")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import uvicorn

    from poker_engine.desktop import aa_server

    app = aa_server.create_app(
        args.work / "profile.json",
        records_dir=args.work / "records",
        rules_path=args.work / "table-rules.json")
    print(json.dumps({"base": f"http://127.0.0.1:{args.port}"}), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
