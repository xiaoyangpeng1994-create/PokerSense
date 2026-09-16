"""USABLE-001 U2 offline trial launcher (versioned, independent, non-destructive).

Start this file with the same local Python the project already uses; it needs no
new dependencies, no global environment change and no installer. It keeps its own
profile, table rules and records under its own state directory, so an existing
installation's program files, configuration and records are never touched.

What it deliberately does NOT do: open a capture device, start an observation
session, call a vision API, kill another process, or take over a busy port.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import socket
import sys
import threading
import time
import webbrowser


VERSION = "aa-trial-u2.0"
IMPLEMENTATION = "aa-analysis-record-v1"

# Stated on every start so nobody mistakes the trial for an accepted release.
UNVERIFIED = (
    "真实牌局输入尚未验证（REAL_HAND_ACCEPTANCE_PENDING）：现有授权牌局不满足内核前提",
    "策略强度未评估（NOT_ASSESSED）：这里只有条件净 EV，不是 GTO，也不是实战胜率",
    "只支持河牌、恰好三名活跃玩家、单底池；不支持边池、全下跨越、翻牌/转牌",
    "对手范围与响应权重都是人工假设，换个假设数字就会变",
    "本入口不会打开采集设备，也不会给出实战提示",
)
REQUIRED_RESOURCES = ("ui/aa-live/index.html", "ui/aa-live/app.js",
                      "ui/aa-live/hand_input.js", "ui/aa-live/analysis_records.js",
                      "configs/strategy/examples/threeway-river-response-manual.json")


def repo_root():
    """The checkout this launcher lives in - never a hard-coded path."""
    return Path(__file__).resolve().parents[2]


def port_is_free(port):
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        return False
    with closing(socket.socket()) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_port(preferred):
    """Use the preferred port when it is free, otherwise an explicit free one.

    Never kills, rebinds or otherwise disturbs whatever already holds it.
    """
    if port_is_free(preferred):
        return preferred, None
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        fallback = probe.getsockname()[1]
    return fallback, (f"端口 {preferred} 已被占用（没有结束任何进程）；"
                      f"本次改用空闲端口 {fallback}")


def preflight(root):
    """Missing resources are reported before anything is started."""
    return [name for name in REQUIRED_RESOURCES if not (root / name).is_file()]


def state_paths(state_dir):
    state_dir = Path(state_dir)
    return {"root": state_dir, "profile": state_dir / "profile.json",
            "rules": state_dir / "table-rules.json",
            "records": state_dir / "records"}


def prepare_state(state_dir):
    """Create the trial's own state; existing files are left exactly as they are."""
    paths = state_paths(state_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["records"].mkdir(parents=True, exist_ok=True)
    if not paths["profile"].is_file():
        paths["profile"].write_text(json.dumps({
            "version": VERSION,
            "note": "离线试用专用配置；本入口不打开采集设备，也不回放媒体",
            "capture": {"enabled": False},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def banner(base, paths, port_note):
    lines = [f"=== PokerSense 离线试用 {VERSION}（记录实现 {IMPLEMENTATION}）==="]
    if port_note:
        lines.append(f"注意：{port_note}")
    lines += [
        f"地址：{base}",
        f"试用状态目录（本次不会改动你已有的程序与记录）：{paths['root']}",
        f"分析记录目录：{paths['records'] / 'analysis-records'}",
        "",
        "尚未验收、试用时必须知道：",
    ]
    lines += [f"  · {item}" for item in UNVERIFIED]
    lines += ["", "关闭窗口即可退出；再次运行会读回同一份记录目录里的已保存分析。"]
    return "\n".join(lines)


def serve(args):
    root = repo_root()
    missing = preflight(root)
    if missing:
        print("缺少必要资源，未启动：" + "、".join(missing), file=sys.stderr)
        return 2
    # The project's documented environment is PYTHONPATH="src;." and the app
    # imports both `poker_engine` (from src) and the `tools` package (from the
    # repository root). The trial must set that up itself so it does not depend on
    # the caller's PYTHONPATH, the current directory or an installed copy.
    for entry in (root, root / "src"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import uvicorn

    from poker_engine.desktop import aa_server

    paths = prepare_state(args.state)
    port, note = pick_port(args.port)
    app = aa_server.create_app(
        paths["profile"], records_dir=paths["records"],
        rules_path=paths["rules"])
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        print("服务未能在 60 秒内就绪，未打开浏览器。", file=sys.stderr)
        return 3
    base = f"http://127.0.0.1:{port}/"
    # The ready line is written BEFORE the banner: a caller that waits on this
    # file must not depend on the banner being flushable in its environment.
    if args.ready_file:
        Path(args.ready_file).write_text(
            json.dumps({"base": base, "port": port, "records": str(paths["records"]),
                        "version": VERSION}), encoding="utf-8")
    print(banner(base, paths, note), flush=True)
    if not args.no_browser:
        # Only after the service really answers, so the page never opens blank.
        webbrowser.open(base)
    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\n正在退出试用（记录已保存在上面的目录里）。")
        server.should_exit = True
        thread.join(timeout=10)
    return 0


def self_check(args):
    root = repo_root()
    missing = preflight(root)
    port, note = pick_port(args.port)
    paths = prepare_state(args.state)
    report = {"version": VERSION, "implementation": IMPLEMENTATION,
              "repo_root": str(root), "missing_resources": missing,
              "preferred_port": args.port, "port": port, "port_note": note,
              "state": {key: str(value) for key, value in paths.items()},
              "unverified": list(UNVERIFIED),
              "capture_enabled": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=f"PokerSense 离线试用 {VERSION}（不打开采集设备，不改动已有程序）")
    parser.add_argument("--port", type=int, default=8791,
                        help="首选端口；被占用时自动改用明确空闲端口（不会结束任何进程）")
    parser.add_argument("--state", type=Path,
                        default=Path(__file__).resolve().parent / "state",
                        help="试用状态目录（配置/桌规/已保存分析都在这里）")
    parser.add_argument("--no-browser", action="store_true",
                        help="只启动服务，不打开浏览器")
    parser.add_argument("--ready-file", type=Path,
                        help="就绪后把实际地址写入该文件（便于脚本化启动）")
    parser.add_argument("--self-check", action="store_true",
                        help="只做资源/端口/目录检查并打印版本，不启动服务")
    parser.add_argument("--version", action="version",
                        version=f"{VERSION} ({IMPLEMENTATION})")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check(args)
    try:
        return serve(args)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - a trial start must never fail silently
        import traceback
        traceback.print_exc()
        if args.ready_file:
            Path(str(args.ready_file) + ".error").write_text(
                traceback.format_exc(), encoding="utf-8")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
