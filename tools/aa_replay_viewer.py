"""Localhost pinned replay with an optional private human confirmation ledger."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from tools.aa_confirmation_ledger import (
    Ledger, declarations, known, model_value, validate_value,
)


REPLAY_SHA = "7f86f4cd631dfae997b2fe7dc8ce42fda819f6b52d8b0127f56e6173e18161bc"
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
    for row in rows:
        declarations(row)
    return rows


def present(row, confirmations=()):
    missing = declarations(row)
    manual = {(e["field"], e["seat"]): e for e in confirmations
              if e["frame"] == row["source_frame"]}
    invalid = []
    if row.get("gap_reset"):
        invalid.append("源记录：融合上下文已重置；不沿用上一帧")
    if row.get("scene") != "AA_TABLE_CANDIDATE":
        invalid.append("源记录：场景不支持")
    fields = []

    def field(name, value, *, declaration=None, confirm=None, seat=None):
        absent = (declaration or name) in missing
        human = manual.get((confirm, seat)) if confirm else None
        conflict = absent and known(value)
        fields.append({
            "name": name, "value": human["human_value"] if human else value,
            "model_output": value,
            "status": "人工确认" if human else "声明冲突" if conflict else
            "未实现" if absent else "拒识／未知" if not known(value)
            else "候选值（未验收）",
            "confidence": "未记录（冻结日志不含数值置信度）",
            "reason": ("人工确认；自动值及原缺失声明保留。" if human else "") +
            ("incomplete_fields:" + (declaration or name) if absent else
             "原回放未记录此字段的缺失原因" if not known(value) else
             "原回放已输出；不等于独立准确率验证"),
            "confirmation_field": confirm, "seat": seat,
            "confirmation_sha256": human["entry_sha256"] if human else None,
        })
        if conflict:
            invalid.append("INCOMPLETE_FIELDS_VALUE_CONFLICT:" + name)

    field("hero", row.get("hero"))
    for slot, value in enumerate(row["board_slots"]):
        field(f"board_slots[{slot}]", value)
    for slot in range(9):
        field(f"stacks[{slot}]", row["stacks"].get(str(slot)))
        field(f"visible_action_glyph[{slot}]", row["actions"].get(str(slot)))
    for name in ("seat_presence", "current_bet", "action"):
        for slot in range(9):
            field(f"{name}[{slot}]", model_value(row, name, slot),
                  declaration="visible_action_glyph" if name == "action" else name,
                  confirm=name, seat=slot)
    for name, key in (("pot", "pot"), ("actor", "current_actor"),
                      ("special_mode", "special_modes")):
        field(key, model_value(row, name, None), confirm=name)
    for name in ("full_actions", "participation", "acceptance"):
        value = (row.get("hero_participation") if name == "participation"
                 else row.get(name))
        field(name, value)
    participation = row.get("hero_participation", "UNKNOWN")
    return {
        "frame": row["source_frame"], "timestamp_ms": row["opencv_pos_msec"],
        "source_sha256": row["source"], "replay_sha256": REPLAY_SHA,
        "scene": row.get("scene"), "fields": fields,
        "participation": ("未实现（源值 UNKNOWN）" if "participation" in missing
                          else "未知（源值 UNKNOWN）") if participation == "UNKNOWN"
        else participation + "（仅可见候选）",
        "invalidity": invalid or ["源记录未报告重置或场景失效；不代表可实战"],
        "incomplete_fields": missing,
        "contract_errors": [v for v in invalid if v.startswith("INCOMPLETE_")],
        "strategy_eligible": strategy_ready(row, manual, invalid),
        "strategy_scope": "offline_field_completeness_only;no_Advice_or_live",
    }


def strategy_ready(row, manual, invalid):
    """A completeness flag only. Never bypass source acceptance or legal history."""
    def effective(field, seat=None):
        item = manual.get((field, seat))
        return item["human_value"] if item else model_value(row, field, seat)

    def cards(value, counts):
        return (isinstance(value, list) and len(value) in counts
                and all(isinstance(c, str) and re.fullmatch(r"[2-9TJQKA][cdhs]", c)
                        for c in value))

    def valid(field, value, seat=None):
        if field in {"pot", "current_bet"} and type(value) is int:
            value = str(value)
        try:
            validate_value(field, seat, value)
            return True
        except ValueError:
            return False

    if (invalid or row.get("acceptance") is not True
            or not cards(row.get("hero"), {2})
            or not cards(row.get("board_slots"), {0, 3, 4, 5})
            or row.get("hero_participation") != "PARTICIPATING"
            or not isinstance(row.get("full_actions"), list)
            or not known(row["full_actions"])):
        return False
    if any(not valid(name, effective(name))
           for name in ("pot", "actor", "special_mode")):
        return False
    if effective("seat_presence", effective("actor")) != "participating":
        return False
    for seat in range(9):
        presence = effective("seat_presence", seat)
        if not valid("seat_presence", presence, seat):
            return False
        if presence in {"participating", "all_in"} and (
                not valid("current_bet", effective("current_bet", seat), seat)
                or not valid("current_bet", row["stacks"].get(str(seat)), seat)):
            return False
    return True


def create_app(rows, ledger=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_write(request: Request, call_next):
        from fastapi.responses import JSONResponse
        if request.method == "POST":
            origin = request.headers.get("origin")
            expected = str(request.base_url).rstrip("/")
            if (origin is not None and origin != expected
                    or request.headers.get("x-aa-confirmation") != "1"
                    or not request.headers.get("content-type", "").startswith(
                        "application/json")):
                return JSONResponse({"detail": "LOCAL_JSON_WRITE_REQUIRED"},
                                    status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def require_ledger():
        if ledger is None:
            raise HTTPException(409, "LEDGER_NOT_CONFIGURED")
        return ledger

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
        try:
            entries = ledger.snapshot()[1] if ledger else []
            result = present(rows[frame], entries)
            result.update(confirmation_enabled=ledger is not None,
                          frame_zone=ledger.zone(frame) if ledger else "unknown")
            return result
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/confirm")
    async def confirm(request: Request):
        book = require_ledger()
        try:
            body = await request.json()
            if set(body) != {"frame", "seat", "field", "human_value"}:
                raise ValueError("CONFIRMATION_KEYS_MISMATCH")
            return book.confirm(body["frame"], body["field"], body["seat"],
                                body["human_value"])
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/report")
    def report():
        return require_ledger().report()

    @app.get("/api/export/{kind}")
    def export(kind: str):
        try:
            return require_ledger().export(kind)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/exposure")
    async def exposure(request: Request):
        try:
            body = await request.json()
            if set(body) != {"entry_sha256"}:
                raise ValueError("EXPOSURE_KEYS_MISMATCH")
            return require_ledger().mark_trained(body["entry_sha256"])
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--zone-policy", type=Path)
    args = parser.parse_args()
    rows = load_replay(args.observations)
    if args.ledger and args.ledger.resolve() == args.observations.resolve():
        parser.error("LEDGER_MUST_NOT_BE_SOURCE")
    if args.zone_policy and not args.ledger:
        parser.error("ZONE_POLICY_REQUIRES_LEDGER")
    policy = (json.loads(args.zone_policy.read_text(encoding="utf-8"))
              if args.zone_policy else None)
    ledger = Ledger(args.ledger, rows, REPLAY_SHA, policy) if args.ledger else None
    print(f"AA frozen replay: {len(rows)} frames; http://127.0.0.1:{args.port}")
    import uvicorn
    uvicorn.run(create_app(rows, ledger), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
