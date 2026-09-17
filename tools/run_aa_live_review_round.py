"""One explicitly authorized, finite round of live DeepSeek image reviews.

Talks only to the local AA app. Does not open the device or hold an API key.
Each AI request uses a saved frame, and receipt files distinguish timed samples
from individual manual marks. No automatic retries or model/code modification.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class LocalAPI:
    def __init__(self, base_url):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in (
                "127.0.0.1", "localhost") or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Only a local AA HTTP endpoint is allowed")
        self.base_url = base_url.rstrip("/")

    def __call__(self, path, body=None):
        request = urllib.request.Request(
            self.base_url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"X-AA-Live": "1", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read(2 * 1024 * 1024))


def sample(api, binding):
    state = api("/api/status")
    if (state.get("instance_id"), state.get("generation")) != binding:
        return {"status": "END", "reason": "session_changed"}
    if state.get("source_kind") != "capture-card" or state.get("status") != "RUNNING":
        return {"status": "END", "reason": "capture_not_running"}
    if not state.get("payload"):
        return {"status": "SKIP", "reason": "no_fresh_payload"}
    config = api("/api/review/config")
    if not config.get("key_configured"):
        return {"status": "END", "reason": "key_cleared"}
    if config["calls_today"] >= config["daily_limit"]:
        return {"status": "END", "reason": "daily_limit"}
    if config.get("busy"):
        return {"status": "SKIP", "reason": "previous_request_busy"}
    record = api("/api/review/mark", {})
    issue = record["issue"]
    observed = issue["observation"]
    identifier = issue["issue_id"]
    if (observed.get("source_kind") != "capture-card"
            or (observed.get("instance_id"), observed.get("generation")) != binding):
        return {"status": "END", "reason": "source_changed_during_mark",
                "issue_id": identifier}
    report = api("/api/review/issues/" + identifier + "/ai", {"consent": True})
    return {"status": "SUBMITTED", "issue_id": identifier,
            "job_id": report["job_id"], "source_frame": observed["source_frame"],
            "image_sha256": issue["preview_sha256"], "model": report["model"],
            "trigger": "EXPLICITLY_AUTHORIZED_TIMED_LIVE_SAMPLE",
            "ai_candidate_only": True, "training_eligible": False}


def run_round(api, output, *, interval=30, maximum=20, consent=False,
              monotonic=time.monotonic, sleep=time.sleep):
    if not consent:
        raise ValueError("Explicit live image upload authorization is required")
    if interval < 30 or not 1 <= maximum <= 20:
        raise ValueError("Minimum interval is 30 seconds; maximum 20 requests")
    state = api("/api/status")
    if state.get("source_kind") != "capture-card" or state.get("status") != "RUNNING":
        raise ValueError("A live capture session must already be running")
    binding = (state.get("instance_id"), state.get("generation"))
    if not isinstance(binding[0], str) or type(binding[1]) is not int:
        raise ValueError("A bound server session is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    submitted, attempts, last_start = 0, 0, None
    log_path = output / "round.jsonl"
    with log_path.open("x", encoding="utf-8") as log:
        def emit(record):
            line = json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                               **record}, ensure_ascii=False)
            log.write(line + "\n")
            log.flush()

        emit({"status": "STARTED", "interval": interval, "maximum": maximum,
              "binding": binding, "real_images_authorized": True})
        while attempts < maximum and monotonic() - started <= maximum * interval:
            if last_start is not None:
                sleep(max(0, interval - (monotonic() - last_start)))
            last_start = monotonic()
            attempts += 1
            try:
                row = sample(api, binding)
            except Exception:
                # An uncertain POST may already be billed: stop, never retry it.
                emit({"status": "END", "reason": "request_failed_or_unconfirmed",
                      "submitted": submitted, "automatic_retry": False})
                return
            emit(row)
            if row["status"] == "SUBMITTED":
                submitted += 1
            elif row["status"] == "END":
                break
        emit({"status": "FINISHED", "submitted": submitted,
              "attempts": attempts, "last_result_may_still_be_running": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8777")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--maximum", type=int, default=20)
    parser.add_argument("--confirm-live-image-upload", action="store_true")
    args = parser.parse_args()
    run_round(LocalAPI(args.base_url), args.output, interval=args.interval,
              maximum=args.maximum, consent=args.confirm_live_image_upload)


if __name__ == "__main__":
    main()
