"""User-triggered local issue snapshots, separate from training or acceptance."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def save_issue(directory, evidence, note, category, table_rules):
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("问题说明最多 2000 字")
    if category not in {"cards", "amounts", "actor", "state", "other"}:
        raise ValueError("不支持的问题类别")
    snapshot, preview = evidence
    if snapshot.get("status") != "RUNNING" or not snapshot.get("payload"):
        raise ValueError("当前没有新鲜识别帧，请开始观察后再保存")
    if not preview:
        raise ValueError("当前帧预览不可用，未保存不完整证据")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    issue_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + (
        uuid.uuid4().hex[:12])
    target = root / issue_id
    target.mkdir(exist_ok=False)
    document = {
        "schema_version": 1, "issue_id": issue_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": note, "category": category, "observation": snapshot,
        "table_rules": table_rules, "user_triggered": True,
        "review_status": "UNREVIEWED", "training_eligible": False,
        "independent_acceptance": False,
        "preview_sha256": hashlib.sha256(preview).hexdigest(),
        "preview_encoding": "JPEG derived from the same processed BGR frame",
    }
    raw = json.dumps(document, ensure_ascii=False, indent=2,
                     allow_nan=False).encode("utf-8")
    for name, content in (("preview.jpg", preview), ("issue.json", raw)):
        with (target / name).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return {"issue_id": issue_id, "directory": str(target),
            "issue_sha256": hashlib.sha256(raw).hexdigest(),
            "review_status": "UNREVIEWED", "training_eligible": False}
