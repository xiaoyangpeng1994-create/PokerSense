"""Build a separate, traceable WPK video-review dataset without changing labels.

Sequential decoding assigns source indices; old normalized screenshots are
linked by exact decoded pixels, never by trusting a seek or filename timestamp.
All participation labels start UNKNOWN until explicit visual review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.capture_card_calibration.viewpoint import extract_evidence


def pixels_digest(image: np.ndarray, *, quick: bool = False) -> str:
    sampled = image[::12, ::12] if quick else image
    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(sampled.tobytes())
    return digest.hexdigest()


def frame_ranges(indices: list[int]) -> list[list[int]]:
    """Compress exact matches without pretending repeated frames are unique."""
    ranges: list[list[int]] = []
    for index in sorted(set(indices)):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ranges


def safe_reference_path(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("reference escapes normalized frame directory")
    return path


def read_image(path: Path) -> np.ndarray:
    picture = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        raise ValueError(f"cannot decode normalized reference: {path}")
    return picture


def probe_video(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_name,width,height,"
        "r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ], capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
    return json.loads(result.stdout)


def freeze_sources(dataset: Path) -> list[dict]:
    sources = []
    for session in ("session_001", "session_002"):
        path = dataset / "source/raw" / f"{session}.mkv"
        before = path.stat()
        print(json.dumps({"stage": "hash_source", "session": session}), flush=True)
        digest = sha256_file(path)
        probe = probe_video(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size, after.st_mtime_ns
        ):
            raise ValueError("source changed during inventory")
        sources.append({"session": session, "path": str(path.resolve()),
                        "sha256": digest, "size_bytes": after.st_size,
                        "mtime_ns": after.st_mtime_ns, "probe": probe})
    return sources


def load_references(dataset: Path, session: str):
    labels_path = dataset / "labels/frames.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8")
              .splitlines() if line.strip()]
    labels = [row for row in labels if row["session_id"] == session]
    manifest = json.loads((dataset / "normalized/manifest.json")
                          .read_text(encoding="utf-8"))
    metadata = {row["file"]: row for row in manifest["frames"]}
    references = []
    for label in labels:
        path = safe_reference_path(dataset / "normalized/frames", label["frame"])
        if sha256_file(path) != label["sha256"]:
            raise ValueError(
                f"legacy label image hash mismatch: {label['frame']}"
            )
        picture = read_image(path)
        references.append({
            "frame": label["frame"], "path": str(path),
            "legacy_hand_id": label["hand_id"],
            "nominal_source_frame": metadata.get(label["frame"], {}).get(
                "source_frame"),
            "pixel_sha256": pixels_digest(picture),
            "quick_sha256": pixels_digest(picture, quick=True),
            "matching_indices": [],
        })
    return references, sha256_file(labels_path)


def review_page(rows: list[dict], session: str) -> str:
    cards = []
    for row in rows:
        if row.get("exclude_preview"):
            picture = '<p>非牌桌手机页面：预览已隔离，原始素材保留。</p>'
        else:
            picture = ('<a href="' + html.escape(row["image"]) + '">'
                       '<img loading="lazy" src="' + html.escape(row["image"])
                       + '"></a>')
        state_text = (
            f"入座：{row['owner_presence']} / 本手：{row['hand_participation']} / "
            f"行动：{row['hero_turn']}"
        ) if row.get("review_state") == "agent_visual_review" else (
            "本人入座 / 本手在局 / 当前行动：均待复核"
        )
        cards.append(
            '<article>' + picture +
            f'<p>原始帧 {row["source_frame"]} · 容器 {row["container_pts_ms"]:.0f} ms</p>'
            '<p>' + html.escape(state_text) + '</p>'
            '<p>' + html.escape(row.get("review_notes", "")) + '</p></article>'
        )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>WPK 录像复核时间线</title><style>'
        'body{font:14px/1.5 system-ui;background:#eef1ef;padding:20px;color:#18271e}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));'
        'gap:16px}article{background:white;padding:10px;border-radius:8px}'
        'img{width:100%;height:auto}p{margin:8px 0}</style>'
        f'<h1>{html.escape(session)} · 连续录像复核</h1>'
        '<p>这些是时间线样本，尚未筛选成“本人正在打牌”的真值。'
        '逐点目检不等于相邻整个区间已确认。容器时间不等于实际录制时间；'
        '不使用按钮缺失推断旁观。点击图片可放大。</p>'
        '<section class="grid">' + ''.join(cards) + '</section></html>'
    )


def render_contact_sheets(directory: Path) -> list[Path]:
    """Compact human-review pages; never use thumbnails as model ground truth."""
    rows = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    target = directory / "contact-sheets"
    target.mkdir(exist_ok=True)
    paths = []
    for start in range(0, len(rows), 8):
        canvas = np.full((1120, 996, 3), 240, np.uint8)
        for offset, row in enumerate(rows[start:start + 8]):
            if row.get("exclude_preview"):
                image = np.full((540, 249, 3), 240, np.uint8)
                cv2.putText(image, "NON-POKER / EXCLUDED", (5, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                image = read_image(directory / row["image"])
                image = cv2.resize(image, (249, 540), interpolation=cv2.INTER_AREA)
            x, y = (offset % 4) * 249, (offset // 4) * 560
            canvas[y + 20:y + 560, x:x + 249] = image
            cv2.putText(canvas, f"frame {row['source_frame']}", (x + 4, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("contact sheet encoding failed")
        path = target / f"page-{start // 8:02d}.jpg"
        path.write_bytes(encoded.tobytes())
        paths.append(path)
    return paths


def apply_review_anchors(directory: Path) -> dict:
    """Attach source-bound visual review without promoting whole intervals."""
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    review = json.loads((directory / "review-anchors.json").read_text(encoding="utf-8"))
    if review["source_sha256"] != summary["source_sha256"]:
        raise ValueError("review source hash mismatch")
    by_frame = {row["source_frame"]: row for row in samples}
    seen = set()
    allowed = {
        "owner_presence": {"SEATED", "NOT_SEATED", "SPECTATING", "NOT_AT_TABLE",
                           "UNKNOWN"},
        "hand_participation": {"IN_HAND", "ALL_IN", "FOLDED", "WAITING", "ENDED",
                               "NOT_APPLICABLE", "UNKNOWN"},
        "hero_turn": {"HERO", "OTHER", "NOT_APPLICABLE", "UNKNOWN"},
    }
    for anchor in review["anchors"]:
        index = anchor["source_frame"]
        if index in seen or index not in by_frame:
            raise ValueError("duplicate or unknown review frame")
        seen.add(index)
        row = by_frame[index]
        if row["pixel_sha256"] != anchor["pixel_sha256"]:
            raise ValueError("review pixel hash mismatch")
        for name, choices in allowed.items():
            if anchor[name] not in choices:
                raise ValueError(f"unknown review value for {name}")
        if anchor["hero_turn"] == "HERO" and (
            anchor["owner_presence"] != "SEATED"
            or anchor["hand_participation"] != "IN_HAND"
        ):
            raise ValueError("hero decision requires seated and in-hand evidence")
    for anchor in review["anchors"]:
        row = by_frame[anchor["source_frame"]]
        row.update(anchor)
        row["review_state"] = "agent_visual_review"
        row["reviewer"] = review["reviewer"]
        row["decision_candidate"] = (
            row["owner_presence"] == "SEATED" and row["hand_participation"] == "IN_HAND"
            and row["hero_turn"] == "HERO" and row["scene"] == "TABLE"
        )
    (directory / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "review.html").write_text(
        review_page(samples, summary["session"]), encoding="utf-8")
    result = {"reviewed_sample_anchors": len(seen), "interval_review_complete": False,
              "owner_presence": dict(Counter(row["owner_presence"] for row in samples)),
              "hand_participation": dict(Counter(
                  row["hand_participation"] for row in samples)),
              "hero_turn": dict(Counter(row["hero_turn"] for row in samples)),
              "decision_candidates": sum(
                  row.get("decision_candidate", False) for row in samples),
              "reviewer": review["reviewer"], "golden_eligible": False}
    (directory / "participation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    render_contact_sheets(directory)
    return result


def build_hand_candidates(dataset: Path, directory: Path) -> list[dict]:
    """Old hand labels are hints for windows to review, not complete-hand truth."""
    references = json.loads(
        (directory / "reference-map.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    labels = {row["frame"]: row for line in (dataset / "labels/frames.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()
              for row in [json.loads(line)]}
    groups = defaultdict(list)
    for ref in references:
        if labels[ref["frame"]]["scene"] == "table" and ref["source_frame_ranges"]:
            groups[ref["legacy_hand_id"]].append(ref)
    candidates = []
    for hand, items in groups.items():
        low = min(span[0] for ref in items for span in ref["source_frame_ranges"])
        high = max(span[1] for ref in items for span in ref["source_frame_ranges"])
        streets = sorted({labels[ref["frame"]]["street"]["value"] for ref in items
                          if labels[ref["frame"]]["street"]["status"] == "VALID"})
        candidates.append({"legacy_hand_id": hand,
                           "matched_reference_count": len(items),
                           "reference_extent": [low, high],
                           "proposed_review_window": [
                               max(0, low - 150),
                               min(summary["decoded_frames"] - 1, high + 150)],
                           "legacy_street_hints": streets,
                           "complete_hand_verified": False,
                           "owner_participation_verified": False})
    candidates.sort(key=lambda row: row["reference_extent"][0])
    (directory / "hand-candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    return candidates


def scan_video(source: dict, dataset: Path, output: Path, step: int) -> dict:
    path = Path(source["path"])
    session = source["session"]
    references, labels_hash = load_references(dataset, session)
    by_quick = defaultdict(list)
    by_nominal = defaultdict(list)
    for ref in references:
        by_quick[ref["quick_sha256"]].append(ref)
        if ref["nominal_source_frame"] is not None:
            by_nominal[ref["nominal_source_frame"]].append(ref)
    config_path = dataset / "normalization/normalization.json"
    config = NormalizationConfig.from_json(config_path.read_text(encoding="utf-8"))
    directory = output / session
    directory.mkdir()
    (directory / "frames").mkdir()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source: {path}")
    declared_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    count = 0
    started = time.monotonic()
    last_pts = None
    nonmonotonic_pts = 0
    with (directory / "frame-index.jsonl").open("w", encoding="utf-8") as index_file:
        try:
            while True:
                ok, raw = capture.read()
                if not ok:
                    break
                picture = normalize(raw, config)
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if last_pts is not None and pts < last_pts:
                    nonmonotonic_pts += 1
                last_pts = pts
                possible = by_quick.get(pixels_digest(picture, quick=True), ())
                full_hash = (
                    pixels_digest(picture) if possible or count % step == 0 else None
                )
                matched = []
                for ref in possible:
                    if ref["pixel_sha256"] == full_hash:
                        ref["matching_indices"].append(count)
                        matched.append(ref["frame"])
                for ref in by_nominal.get(count, ()):
                    expected = read_image(Path(ref["path"]))
                    ref["nominal_pixels_equal"] = bool(
                        np.array_equal(expected, picture))
                    if expected.shape == picture.shape:
                        ref["nominal_mean_abs_difference"] = round(float(
                            np.abs(expected.astype(np.int16) - picture).mean()), 6)
                index_file.write(json.dumps({
                    "source_frame": count, "container_pts_ms": pts,
                    "exact_reference_matches": matched}) + "\n")
                if count % step == 0:
                    relative = f"frames/{count:06d}.jpg"
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not encoded_ok:
                        raise ValueError("preview encoding failed")
                    (directory / relative).write_bytes(encoded.tobytes())
                    evidence = extract_evidence(picture).to_dict()
                    rows.append({
                        "source_frame": count, "container_pts_ms": pts,
                        "pixel_sha256": full_hash, "image": relative,
                        "preview_sha256": sha256_file(directory / relative),
                        "owner_presence": "UNKNOWN",
                        "hand_participation": "UNKNOWN",
                        "hero_turn": "UNKNOWN", "review_state": "unreviewed",
                        "legacy_viewpoint_candidate": evidence.pop("verdict"),
                        "legacy_signals": evidence,
                    })
                count += 1
                if count % 1000 == 0:
                    elapsed = round(time.monotonic() - started, 1)
                    print(json.dumps({"stage": "decode", "session": session,
                                      "frames": count, "elapsed_s": elapsed}),
                          flush=True)
        finally:
            capture.release()
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
        raise ValueError("source changed while decoding")
    for ref in references:
        indices = ref.pop("matching_indices")
        ref["match_count"] = len(indices)
        ref["source_frame_ranges"] = frame_ranges(indices)
        ref["mapping_status"] = ("NOT_FOUND" if not indices else
                                 "UNIQUE" if len(indices) == 1 else "MULTIPLE")
        ref.pop("quick_sha256")
    summary = {
        "session": session, "source_sha256": source["sha256"],
        "labels_sha256": labels_hash, "normalization_sha256": sha256_file(config_path),
        "decoded_frames": count, "declared_frames": declared_count,
        "decode_count_matches": declared_count > 0 and count == declared_count,
        "pts_regressions": nonmonotonic_pts, "last_container_pts_ms": last_pts,
        "review_samples": len(rows), "sample_step_frames": step,
        "reference_status_counts": dict(Counter(
            ref["mapping_status"] for ref in references)),
        "nominal_reference_exact": sum(ref.get("nominal_pixels_equal", False)
                                       for ref in references),
        "elapsed_processing_s": round(time.monotonic() - started, 3),
        "participation_review_complete": False, "release_eligible": False,
    }
    for name, value in (("summary.json", summary), ("samples.json", rows),
                        ("reference-map.json", references)):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (directory / "review.html").write_text(review_page(rows, session), encoding="utf-8")
    print(json.dumps({"stage": "session_complete", **summary}), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session", choices=("session_001", "session_002"),
                        default="session_002")
    parser.add_argument("--sample-step", type=int, default=300)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output == dataset or output.is_relative_to(dataset):
        parser.error("output must be outside the original dataset")
    if args.sample_step <= 0:
        parser.error("sample-step must be positive")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = freeze_sources(dataset)
    inventory = {"created_at": datetime.now(timezone.utc).isoformat(),
                 "sources": sources, "excluded_variants": [
                     str(p) for p in sorted(
                         (dataset / "source/raw").glob("*_UNFINALIZED.mkv"))],
                 "exclusion_reason": (
                     "not treated as independent sessions; originals retained")}
    (output / "sources.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    source = next(row for row in sources if row["session"] == args.session)
    result = scan_video(source, dataset, output, args.sample_step)
    render_contact_sheets(output / args.session)
    write_sha256sums(output)
    return 0 if result["decode_count_matches"] and not result["pts_regressions"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
