"""AA eight-seat monitor: shared session, explicit controls, no game inputs."""

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit
import webbrowser

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response

from .aa_reader import AA8Reader, preflight_profile
from .aa_session import AARecognitionSession
from .aa_sources import source_factory
from .aa_table_config import AATableConfigStore
from .aa_issues import save_issue
from .aa_analysis import AAConditionalAnalysis


def ui_root():
    frozen = getattr(sys, "_MEIPASS", None)
    root = Path(frozen) if frozen else Path(__file__).resolve().parents[3]
    return root / "ui" / "aa-live"


def create_app(profile_path, *, replay_pool=None, replay_first=None,
               replay_last=None, replay_playlist=None, allow_capture=False,
               session=None, rules_path=None, records_dir=None, bundle_sha256=None,
               analysis_service=None):
    profile_path = Path(profile_path)
    service = session or AARecognitionSession(
        source_factory(profile_path, replay_pool=replay_pool,
                       replay_first=replay_first, replay_last=replay_last,
                       replay_playlist=replay_playlist,
                       allow_capture=allow_capture),
        lambda: AA8Reader(profile_path, bundle_sha256=bundle_sha256),
        interval_seconds=0.15,
    )
    rules = AATableConfigStore(rules_path)
    analysis = analysis_service or AAConditionalAnalysis()
    controls_lock = threading.RLock()
    profile_status = (preflight_profile(profile_path, bundle_sha256=bundle_sha256)
                      if bundle_sha256 else preflight_profile(profile_path))

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.stop()
        analysis.cancel()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.aa_session = service
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if request.headers.get("X-AA-Live") != "1":
                return Response("需要页面发起的操作", status_code=403)
            if origin:
                parsed = urlsplit(origin)
                if (parsed.scheme not in {"http", "https"}
                        or parsed.netloc != request.url.netloc):
                    return Response("拒绝跨来源操作", status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != (
                    "application/json"):
                return Response("需要 JSON 请求", status_code=415)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(ui_root() / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(ui_root() / "app.js",
                            media_type="text/javascript")

    @app.get("/style.css")
    def style():
        return FileResponse(ui_root() / "style.css", media_type="text/css")

    @app.get("/controls.js")
    def controls_script():
        return FileResponse(ui_root() / "controls.js", media_type="text/javascript")

    @app.get("/analysis.js")
    def analysis_script():
        return FileResponse(ui_root() / "analysis.js", media_type="text/javascript")

    @app.get("/api/status")
    def status():
        with controls_lock:
            result = service.snapshot()
            table_rules = rules.get()
            analysis_status = analysis.status()
            binding = analysis_status.get("binding", {})
            if (binding and analysis_status["status"] not in ("IDLE", "CANCELLED")
                    and (binding.get("table_rules_revision") != (
                    table_rules["revision"]) or binding.get("generation") != (
                        result["generation"]))):
                analysis_status = analysis.cancel()
        result.update(profile=profile_status,
                      replay_available=replay_pool is not None or (
                          replay_playlist is not None),
                      capture_available=allow_capture,
                      issue_recording_available=records_dir is not None,
                      table_rules=table_rules,
                      analysis=analysis_status,
                      strategy_scope="AA8_OBSERVATION_ONLY_NO_ADVICE")
        return result

    @app.get("/api/preview.jpg")
    def preview():
        content = service.preview()
        if content is None:
            raise HTTPException(404, "当前没有新鲜画面")
        return Response(content, media_type="image/jpeg")

    @app.post("/api/start")
    async def start(request: Request):
        try:
            options = await request.json()
        except ValueError:
            raise HTTPException(400, "无法读取 JSON") from None
        if not isinstance(options, dict) or set(options) - {
                "mode", "device_index", "api", "fps"}:
            raise HTTPException(400, "来源设置字段不受支持")
        if options.get("mode") not in {"capture-card", "development-replay"}:
            raise HTTPException(400, "请选择来源")
        if "fps" in options and (type(options["fps"]) is not int
                                 or options["fps"] != 30):
            raise HTTPException(400, "当前 AA 来源仅请求 30 FPS 采集")
        if options["mode"] == "capture-card" and not allow_capture:
            raise HTTPException(403, "本次启动未启用采集卡")
        if (options["mode"] == "development-replay" and replay_pool is None
                and replay_playlist is None):
            raise HTTPException(400, "未配置开发回放")
        profile = (preflight_profile(profile_path, bundle_sha256=bundle_sha256)
                   if bundle_sha256 else preflight_profile(profile_path))
        if not profile["ready"]:
            raise HTTPException(409, {"message": "AA 识别资源未准备好",
                                      "errors": profile["errors"]})
        with controls_lock:
            analysis.cancel()
            service.start(options)
            return service.snapshot()

    @app.post("/api/stop")
    def stop():
        with controls_lock:
            analysis.cancel()
            service.stop()
            return service.snapshot()

    @app.get("/api/rules")
    def get_rules():
        return rules.get()

    @app.post("/api/rules")
    async def update_rules(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"document", "revision"}:
                raise ValueError("规则请求需要 document 和 revision")
            with controls_lock:
                result = rules.save(body["document"], body["revision"])
                analysis.cancel()
                # Rule changes invalidate the old observation/analysis context.
                service.stop()
                return result
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/issues")
    async def record_issue(request: Request):
        if records_dir is None:
            raise HTTPException(403, "本次启动没有配置问题记录目录")
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"note", "category"}:
                raise ValueError("问题记录需要 note 和 category")
            with controls_lock:
                evidence, current_rules = service.evidence(), rules.get()
            return save_issue(records_dir, evidence, body["note"],
                              body["category"], current_rules)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/analysis/example/{kind}")
    def analysis_example(kind: str):
        names = {"terminal": "terminal-multiway-river-manual.json",
                 "threeway": "threeway-river-response-manual.json"}
        if kind not in names:
            raise HTTPException(404, "未知示例")
        root = ui_root().parent.parent
        path = root / "configs" / "strategy" / "examples" / names[kind]
        return {"scope": "MANUAL_EXAMPLE_NOT_CURRENT_TABLE",
                "document": json.loads(path.read_text(encoding="utf-8"))}

    @app.post("/api/analysis")
    async def start_analysis(request: Request):
        raw = await request.body()
        if len(raw) > 220000:
            raise HTTPException(413, "分析输入过大")
        try:
            from tools.analyze_terminal_multiway import unique_object
            body = json.loads(raw, object_pairs_hook=unique_object)
            if not isinstance(body, dict) or set(body) != {
                    "kind", "document", "rules_source", "rules_revision"}:
                raise ValueError("分析请求字段不完整")
            if body["rules_source"] not in ("document", "table"):
                raise ValueError("请选择规则来自手工场景或本桌设置")
            document = body["document"]
            if not isinstance(document, dict):
                raise ValueError("场景必须为 JSON 对象")
            with controls_lock:
                current = rules.get()
                if body["rules_revision"] != current["revision"]:
                    raise ValueError("本桌规则已变更，请重新检查分析输入")
                if body["rules_source"] == "table":
                    if not current["conditional_analysis_ready"]:
                        raise ValueError("本桌规则不完整或含不支持的特殊机制")
                    document["rules"] = current["simulation_rules"]
                observation = service.snapshot()
                return analysis.start(body["kind"], document, binding={
                    "table_rules_revision": current["revision"],
                    "generation": observation["generation"],
                    "rules_source": body["rules_source"],
                    "effective_rules": document.get("rules"),
                    "input_source": "MANUAL_HYPOTHESIS_NOT_LIVE_STATE",
                })
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/analysis/cancel")
    def cancel_analysis():
        with controls_lock:
            return analysis.cancel()

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="AA 扑克八座识别监控")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--replay-pool", type=Path)
    parser.add_argument("--replay-first", type=int)
    parser.add_argument("--replay-last", type=int)
    parser.add_argument("--replay-playlist", type=Path)
    parser.add_argument("--rules-path", type=Path)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--allow-capture", action="store_true")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.profile, replay_pool=args.replay_pool,
                     replay_first=args.replay_first, replay_last=args.replay_last,
                     replay_playlist=args.replay_playlist,
                     rules_path=args.rules_path, records_dir=args.records_dir,
                     bundle_sha256=args.bundle_sha256,
                     allow_capture=args.allow_capture)
    if args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(
            f"http://127.0.0.1:{args.port}")).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
