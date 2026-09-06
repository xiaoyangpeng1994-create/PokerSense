"""AA 扑克采集卡录制脚本（后台长跑版）。

独立于 capture_card_calibration 工具链调用 record_session，
连续录制 AA 扑克牌桌素材到全新的独立数据集目录。
与 WPK 平台共用同一套采集卡/手机硬件，但所有几何/阈值证据独立重新生成。

用法:
    python tools/aa_record_session.py --root <root> --session session_001 \
        [--max-seconds N] [--codec FFV1]

停止:
    Ctrl+C 会触发 finally 完成视频封装（安全停录）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from capture_card_calibration.record import (
    record_session,
    write_session_log,
    update_device_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--api", default="MSMF", choices=["MSMF", "DSHOW", "ANY"])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="YUY2")
    parser.add_argument(
        "--codec", default="FFV1", choices=["FFV1", "MJPG", "H264", "HEVC"]
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    raw = args.root / "source" / "raw" / f"{args.session}.mkv"
    log_path = args.root / "source" / "probe" / f"{args.session}.capture.json"

    print(f"recording {args.session} -> {raw}", flush=True)
    print(f"  device={args.device} api={args.api} "
          f"size={args.width}x{args.height} fps={args.fps} "
          f"fourcc={args.fourcc} codec={args.codec}", flush=True)
    print("  Ctrl+C 停止并封装视频", flush=True)

    log = record_session(
        raw,
        session_id=args.session,
        device_index=args.device,
        api=args.api,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        codec=args.codec,
        max_seconds=args.max_seconds,
    )

    write_session_log(log_path, log)
    print(f"\nrecorded {log.written_frames} frame(s) ({log.duration_s:.1f}s), "
          f"{len(log.events)} signal event(s)", flush=True)
    print(f"capture log -> {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; video finalized by record_session finally block",
              flush=True)
        raise SystemExit(130)
