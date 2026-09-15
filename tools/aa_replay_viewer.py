"""Read-only localhost viewer for the pinned AA 0-1800 observation replay."""

import argparse
import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


REPLAY_SHA = "7f86f4cd631dfae997b2fe7dc8ce42fda819f6b52d8b0127f56e6173e18161bc"
UNIMPLEMENTED = ("pot", "seat_presence", "current_actor", "full_actions",
                 "special_modes")
UI = Path(__file__).resolve().parents[1] / "ui" / "aa-replay"


def load_replay(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPLAY_SHA:
        raise ValueError("REPLAY_SHA256_MISMATCH")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != 1801 or [r["source_frame"] for r in rows] != list(range(1801)):
        raise ValueError("REPLAY_FRAME_RANGE_MISMATCH")
    if any(r.get("strategy_eligible") is not False for r in rows):
        raise ValueError("REPLAY_STRATEGY_MUST_BE_FALSE")
    return rows


def present(row):
    invalid = []
    if row.get("gap_reset"):
        invalid.append("源记录：融合上下文已重置；不沿用上一帧")
    if row.get("scene") != "AA_TABLE_CANDIDATE":
        invalid.append("源记录：场景不支持")
    fields = []

    def field(name, value, *, absent=False):
        fields.append({
            "name": name, "value": None if absent else value,
            "status": "未实现" if absent else "拒识／未知" if value is None
            else "候选值（未验收）",
            "confidence": "未记录（冻结日志不含数值置信度）",
            "reason": "指定冻结回放未实现此字段" if absent else
            "原回放未记录细分拒识原因" if value is None else
            "原回放已输出；不等于独立准确率验证",
        })

    field("hero", row.get("hero"))
    for slot, value in enumerate(row["board_slots"]):
        field(f"board_slots[{slot}]", value)
    for slot in range(9):
        field(f"stacks[{slot}]", row["stacks"].get(str(slot)))
        field(f"visible_action_glyph[{slot}]", row["actions"].get(str(slot)))
    for name in UNIMPLEMENTED:
        field(name, None, absent=True)
    participation = row.get("hero_participation", "UNKNOWN")
    return {
        "frame": row["source_frame"], "timestamp_ms": row["opencv_pos_msec"],
        "source_sha256": row["source"], "replay_sha256": REPLAY_SHA,
        "scene": row.get("scene"), "fields": fields,
        "participation": "未实现（源值 UNKNOWN）" if participation == "UNKNOWN"
        else participation + "（仅可见候选）",
        "invalidity": invalid or ["源记录未报告重置或场景失效；不代表可实战"],
        "strategy_eligible": False,
    }


def create_app(rows):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index():
        return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/app.js")
    def script():
        return FileResponse(UI / "app.js", media_type="text/javascript")

    @app.get("/api/frame/{frame}")
    def frame_data(frame: int):
        if not 0 <= frame < len(rows):
            raise HTTPException(404, "FRAME_OUT_OF_RANGE")
        return present(rows[frame])

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    rows = load_replay(args.observations)
    print(f"AA frozen replay: {len(rows)} frames; http://127.0.0.1:{args.port}")
    import uvicorn
    uvicorn.run(create_app(rows), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
