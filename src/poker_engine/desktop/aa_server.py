"""AA eight-seat monitor: shared session, explicit controls, no game inputs."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from urllib.parse import urlsplit
import webbrowser

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response

from .aa_reader import AA8Reader, preflight_profile
from .aa_session import AARecognitionSession
from .aa_sources import source_factory


def ui_root():
    frozen = getattr(sys, "_MEIPASS", None)
    root = Path(frozen) if frozen else Path(__file__).resolve().parents[3]
    return root / "ui" / "aa-live"


def create_app(profile_path, *, replay_pool=None, replay_first=None,
               replay_last=None, allow_capture=False, session=None):
    profile_path = Path(profile_path)
    service = session or AARecognitionSession(
        source_factory(profile_path, replay_pool=replay_pool,
                       replay_first=replay_first, replay_last=replay_last,
                       allow_capture=allow_capture),
        lambda: AA8Reader(profile_path), interval_seconds=0.15,
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.stop()

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

    @app.get("/api/status")
    def status():
        result = service.snapshot()
        result.update(profile=preflight_profile(profile_path),
                      replay_available=replay_pool is not None,
                      capture_available=allow_capture,
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
        if options["mode"] == "development-replay" and replay_pool is None:
            raise HTTPException(400, "未配置开发回放")
        profile = preflight_profile(profile_path)
        if not profile["ready"]:
            raise HTTPException(409, {"message": "AA 识别资源未准备好",
                                      "errors": profile["errors"]})
        service.start(options)
        return service.snapshot()

    @app.post("/api/stop")
    def stop():
        service.stop()
        return service.snapshot()

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="AA 扑克八座识别监控")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--replay-pool", type=Path)
    parser.add_argument("--replay-first", type=int)
    parser.add_argument("--replay-last", type=int)
    parser.add_argument("--allow-capture", action="store_true")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.profile, replay_pool=args.replay_pool,
                     replay_first=args.replay_first, replay_last=args.replay_last,
                     allow_capture=args.allow_capture)
    if args.open_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(
            f"http://127.0.0.1:{args.port}")).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
