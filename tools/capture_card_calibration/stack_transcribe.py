# -*- coding: utf-8 -*-
"""Stage F stack-value transcription for the capture-card platform.

The blocker stage H/I keeps hitting is that an ``OCCUPIED`` seat whose ``stack``
is still ``UNKNOWN`` de-qualifies the whole frame as a "stable positive", which
is why calibration/validation splits come out empty. This module turns the
readable stack pills into a human-transcribable worksheet and applies the
values you confirm, without ever guessing a single digit.

Philosophy (guide rule: a missing observation is never filled with a guess):

- The worksheet **renders, never writes**. It shows each target
  (``frame + slot_id``) with a zoomed crop of that seat's stack pill so the
  digits are legible, plus the frame context and the current label state.
- You fill the value in the accompanying CSV; a separate apply step promotes
  only the values you actually returned. An empty/blank cell is left alone —
  it is never auto-filled, never defaulted.
- The apply step keeps a timestamped backup of ``frames.jsonl`` before it
  writes anything, and re-reads the file so a concurrent edit cannot be
  silently clobbered.

This module is import-safe without OpenCV; the crop rendering imports cv2
lazily. Geometry is taken from ``seat_reader`` (``SLOT_LAYOUT_MULTI`` /
``SLOT_LAYOUT_S002``) — it is not reused from any other platform, and it stays
in the normalized canvas space (guide rules 1-2).
"""

from __future__ import annotations

import base64
import csv
import html
import io
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .dataset import read_frames_jsonl, write_frames_jsonl
from .schema import FieldValue, FrameLabel, LabelStatus, Occupancy
from .seat_reader import SLOT_LAYOUT_MULTI, SLOT_LAYOUT_S002

#: The stack pill is tiny on the canvas; we crop it and enlarge this factor so
#: the digits are legible on the worksheet without the viewer squinting.
_STACK_UPSCALE = 4

#: Crop is encoded as JPEG at this quality — legible digits, manageable size.
_CROP_JPEG_QUALITY = 92

#: Crop padding (as a multiple of the pill's own w/h) so the digit ROI is not
#: clipped at the pill edges.
_PAD_X = 0.5
_PAD_Y = 2.0


# --- gap collection -------------------------------------------------------

@dataclass(frozen=True)
class StackGap:
    """One readable-but-untitled stack: an OCCUPIED seat with UNKNOWN stack."""

    frame: str
    session_id: str
    hand_id: str
    timestamp_ms: int
    slot_id: int
    layout_key: str  # "multi" | "s002" — which slot layout the frame uses

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "session_id": self.session_id,
            "hand_id": self.hand_id,
            "timestamp_ms": self.timestamp_ms,
            "slot_id": self.slot_id,
            "layout_key": self.layout_key,
        }


def _layout_for_session(session_id: str) -> dict[int, dict[str, float]]:
    """Pick the per-session slot layout (hero pill sits lower on session_002)."""
    if session_id == "session_002":
        return SLOT_LAYOUT_S002
    return SLOT_LAYOUT_MULTI


def collect_stack_gaps(
    labels: Sequence[FrameLabel],
    *,
    session: str | None = None,
) -> tuple[StackGap, ...]:
    """Return the OCCUPIED-but-stack-UNKNOWN slots to transcribe.

    Only a seat that is confirmed ``OCCUPIED`` and whose ``stack`` is still
    ``UNKNOWN`` is a target: an empty seat genuinely has no stack number, and a
    ``CONFLICT`` needs re-reading, not transcription — it must never be filled
    with a typed guess.
    """
    gaps: list[StackGap] = []
    for label in labels:
        if session is not None and label.session_id != session:
            continue
        if not label.stable or label.scene.value != "table":
            continue
        layout_key = "s002" if label.session_id == "session_002" else "multi"
        for slot in label.slots:
            occ = slot.occupancy
            stack = slot.stack
            if (
                occ.status is LabelStatus.VALID
                and occ.value == Occupancy.OCCUPIED.value
                and stack.status is LabelStatus.UNKNOWN
            ):
                gaps.append(
                    StackGap(
                        frame=label.frame,
                        session_id=label.session_id,
                        hand_id=label.hand_id,
                        timestamp_ms=label.timestamp_ms,
                        slot_id=slot.slot_id,
                        layout_key=layout_key,
                    )
                )
    return tuple(gaps)


# --- worksheet rendering --------------------------------------------------

def _crop_stack_pill(
    img, slot_id: int, layout: dict[int, dict[str, float]]
):
    """Crop + upscale one slot's stack-pill region."""
    import cv2

    H, W = img.shape[:2]
    row = layout[slot_id]
    cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
    x0 = max(0, int((cx - w / 2) * W) - int(w * W * _PAD_X))
    x1 = min(W, int((cx + w / 2) * W) + int(w * W * _PAD_X))
    y0 = max(0, int((cy - h / 2) * H) - int(h * H * _PAD_Y))
    y1 = min(H, int((cy + h / 2) * H) + int(h * H * _PAD_Y))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=_STACK_UPSCALE, fy=_STACK_UPSCALE,
                      interpolation=cv2.INTER_CUBIC)
    # JPEG (not PNG) keeps the digits legible while keeping a 180-target
    # worksheet down from hundreds of MB to a browser-huggable size.
    ok, buf = cv2.imencode(".jpg", crop, [
        cv2.IMWRITE_JPEG_QUALITY, _CROP_JPEG_QUALITY,
    ])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


#: Frame thumbnail is only context for the labeller; encode it as a small JPEG
#: so a multi-hundred-row worksheet stays a page that a browser can hold.
_FRAME_THUMB_WIDTH = 720
_FRAME_JPEG_QUALITY = 70


def _encode_frame_uri(img) -> str:
    """Encode a frame as a compact JPEG thumbnail for the worksheet context."""
    import cv2

    H, W = img.shape[:2]
    if W > _FRAME_THUMB_WIDTH:
        scale = _FRAME_THUMB_WIDTH / W
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [
        cv2.IMWRITE_JPEG_QUALITY, _FRAME_JPEG_QUALITY,
    ])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def render_stack_worksheet(
    gaps: Sequence[StackGap],
    labels_by_frame: dict[str, FrameLabel],
    frames_dir: Path,
    *,
    title: str = "Stack value transcription",
    summary: str = "",
    include_images: bool = True,
) -> str:
    """Render a self-contained HTML worksheet: each target with a zoomed crop.

    The page is a *form*, not a result: it annotates what to read and leaves a
    CSV-style table for the labeller to fill. No value is written here.
    """
    # Frame-level image cache: a frame that hosts several targets is decoded
    # and encoded once, not once per target, or a 60-frame / 180-target sheet
    # would balloon into hundreds of megabytes.
    frame_cache: dict[str, tuple[str, str]] = {}
    rows = []
    for gap in gaps:
        layout = _layout_for_session(gap.session_id)
        label = labels_by_frame.get(gap.frame)

        frame_uri = ""
        crop_uri = ""
        if include_images:
            cached = frame_cache.get(gap.frame)
            if cached is not None:
                frame_uri, crop_uri = cached
            else:
                try:
                    import cv2

                    import numpy as np

                    img = cv2.imdecode(
                        np.frombuffer((frames_dir / gap.frame).read_bytes(),
                                      dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if img is None:
                        frame_cache[gap.frame] = ("", "")
                    else:
                        frame_uri = _encode_frame_uri(img)
                        crop_uri = (
                            _crop_stack_pill(img, gap.slot_id, layout) or ""
                        )
                        frame_cache[gap.frame] = (frame_uri, crop_uri)
                except Exception:  # pragma: no cover - defensive
                    frame_cache[gap.frame] = ("", "")
                    frame_uri, crop_uri = "", ""
        # Frame context: street and pot if readable (helps the labeller orient).
        street = ""
        pot = ""
        if label is not None:
            if label.street.status is LabelStatus.VALID:
                street = str(label.street.value)
            if label.pot.status is LabelStatus.VALID:
                pot = str(label.pot.value)
        rows.append(
            _row_html(
                gap, crop_uri, frame_uri, street, pot,
                index=len(rows),
            )
        )

    body = (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_CSS_STACK}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class=\"sub\">{html.escape(summary)}</p>"
        "<div class=\"toolbar\">"
        f"<span id=\"progress\">已填 0 / {len(gaps)}</span>"
        "<button id=\"download\" type=\"button\">下载填好的 CSV</button>"
        "<button id=\"clear\" type=\"button\">清空全部</button>"
        "</div>"
        "<p class=\"hint\">对着左边的<b>放大数字</b>，把整数填进对应的输入框。看不清就留空。"
        "填完点「下载填好的 CSV」，再用 <code>stack-apply --csv &lt;下载的文件&gt;</code> 写回。"
        "本页填了不会自动改任何文件，只有下载 + 命令行应用那一步才写数据集。</p>"
        "<table class=\"ws\"><thead><tr>"
        "<th>frame</th><th>slot</th><th>ctx</th><th>crop</th><th>value</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        f"<script>{_JS_STACK}</script>"
        "<p class=\"foot\">Private dataset — frames must never enter Git. "
        "An unconfirmed value is never auto-filled.</p>"
        "</body></html>"
    )
    return body


def _row_html(gap, crop_uri, frame_uri, street, pot, *, index: int) -> str:
    # Show the crop as the primary reading surface; show the full frame as a
    # small context thumbnail when available.
    ctx_bits = []
    if street:
        ctx_bits.append(f"street {html.escape(street)}")
    if pot:
        ctx_bits.append(f"pot {html.escape(pot)}")
    ctx = " · ".join(ctx_bits)
    frame_cell = (
        f"<div class=\"frame\">{html.escape(gap.frame)}</div>"
        f"<div class=\"ctx\">{ctx}</div>"
        f"<div class=\"seats\">seat {gap.slot_id}</div>"
        f"<div class=\"hand\">{html.escape(gap.hand_id)}</div>"
    )
    crop_cell = (
        f'<img class="crop" alt="slot {gap.slot_id} stack" '
        f'src="data:image/jpeg;base64,{crop_uri}">'
        if crop_uri
        else "<div class=\"noimg\">crop unavailable</div>"
    )
    thumb_cell = (
        f'<img class="thumb" alt="frame" src="data:image/jpeg;base64,{frame_uri}">'
        if frame_uri
        else ""
    )
    return (
        f"<tr><td class=\"frame-cell\">{frame_cell}</td>"
        f"<td class=\"slot-cell\">{gap.slot_id}</td>"
        f"<td class=\"ctx-cell\">{thumb_cell}</td>"
        f"<td class=\"crop-cell\">{crop_cell}</td>"
        f"<td class=\"value-cell\"><input type=\"text\" inputmode=\"numeric\" "
        f"data-index=\"{index}\" "
        f"data-frame=\"{html.escape(gap.frame)}\" "
        f"data-slot=\"{gap.slot_id}\" placeholder=\"?\"></td></tr>"
    )


def render_stack_csv(gaps: Sequence[StackGap]) -> str:
    """Render the fill-in CSV template (headers + one row per target)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["frame", "slot_id", "value"])
    for gap in gaps:
        writer.writerow([gap.frame, gap.slot_id, ""])
    return buf.getvalue()


# --- apply values (the ONLY writer) ---------------------------------------

@dataclass
class ApplyResult:
    applied: int
    blank: int
    unknown_frame: int
    not_unknown_stack: int
    backup_path: str
    total: int

    def summary_line(self) -> str:
        return (
            f"applied {self.applied} / blank {self.blank} / "
            f"unknown-frame {self.unknown_frame} / already-set {self.not_unknown_stack}"
        )


def apply_stack_values(
    labels_path: Path,
    values: Sequence[object],
    *,
    backup_dir: Path | None = None,
) -> ApplyResult:
    """Promote the labeller-confirmed stack values to VALID.

    ``values`` is an iterable of rows with ``frame``, ``slot_id``, ``value``
    (string or int). A blank/empty ``value`` is left untouched; a non-empty
    value that validates as a non-negative int promotes that slot's ``stack``
    to ``VALID``. Anything that does not match a known OCCUPIED-UNKNOWN target
    is skipped and counted, never invented.

    ``labels_path`` is re-read, backup created, then rewritten (lossless, one
    JSON object per line) — a concurrent edit is picked up rather than
    clobbered.
    """
    labels = read_frames_jsonl(labels_path)
    by_frame = {label.frame: label for label in labels}

    # Build index of OCCUPIED-UNKNOWN stack targets per (frame, slot_id).
    targets: dict[tuple[str, int], FrameLabel] = {}
    for label in labels:
        for slot in label.slots:
            if (
                slot.occupancy.status is LabelStatus.VALID
                and slot.occupancy.value == Occupancy.OCCUPIED.value
                and slot.stack.status is LabelStatus.UNKNOWN
            ):
                targets[(label.frame, slot.slot_id)] = label

    applied = blank = unknown_frame = not_unknown_stack = 0

    for row in values:
        if not isinstance(row, dict):
            continue
        frame = row.get("frame")
        raw_slot = row.get("slot_id")
        raw = row.get("value")
        # Blank cell -> leave alone (never auto-fill).
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            blank += 1
            continue
        if frame not in by_frame:
            unknown_frame += 1
            continue
        try:
            slot_id = int(raw_slot)
        except (TypeError, ValueError):
            unknown_frame += 1
            continue
        key = (frame, slot_id)
        label = by_frame[frame]
        if key not in targets:
            not_unknown_stack += 1
            continue
        try:
            value = int(str(raw).strip())
            if value < 0:
                raise ValueError
        except ValueError:
            blank += 1
            continue
        # Promote in-place the matched slot.
        _promote_stack(label, slot_id, value)
        applied += 1

    # Back up the original file before writing, timestamped.
    backup_path = ""
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = str(backup_dir / f"frames.jsonl.stack_{stamp}.bak")
        Path(backup_path).write_bytes(Path(labels_path).read_bytes())

    write_frames_jsonl(labels_path, labels)
    return ApplyResult(
        applied=applied,
        blank=blank,
        unknown_frame=unknown_frame,
        not_unknown_stack=not_unknown_stack,
        backup_path=backup_path,
        total=len(values),
    )


def _promote_stack(label: FrameLabel, slot_id: int, value: int) -> None:
    """Replace the given slot's stack with VALID(value), leaving rest intact."""
    slots = list(label.slots)
    for i, slot in enumerate(slots):
        if slot.slot_id == slot_id:
            slots[i] = _replace_slot_stack(slot, FieldValue.valid(value))
            break
    object.__setattr__(label, "slots", tuple(slots))


def _replace_slot_stack(slot, stack_field) -> object:
    return replace(slot, stack=stack_field)


# --- CSV / worksheet CSS --------------------------------------------------

_CSS_STACK = """
:root{--bg:#f5f6f8;--card:#fff;--ink:#1c1e21;--muted:#6a7280;--line:#e2e5ea;
  --gap:#c2410c;}
*{box-sizing:border-box;}
body{margin:0;padding:24px;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
h1{font-size:20px;margin:0 0 6px;}
.sub{color:var(--muted);margin:0 0 16px;}
.hint{margin:0 0 16px;padding:10px 12px;background:#fff7ed;color:#9a3412;
  border:1px solid #fed7aa;border-radius:8px;font-size:13px;}
table.ws{width:100%;border-collapse:collapse;background:var(--card);
  border:1px solid var(--line);border-radius:10px;overflow:hidden;}
table.ws th,table.ws td{padding:8px 10px;border-bottom:1px solid var(--line);
  text-align:left;vertical-align:top;}
table.ws th{color:var(--muted);font-weight:500;white-space:nowrap;}
.frame-cell{max-width:280px;}
.frame{font-size:12px;word-break:break-all;color:var(--muted);}
.ctx{font-size:12px;color:var(--ink);margin-top:2px;}
.seats{font-size:13px;font-weight:600;margin-top:4px;color:var(--gap);}
.hand{font-size:11px;color:var(--muted);margin-top:2px;}
.ctx-cell img.thumb{max-width:120px;border:1px solid var(--line);border-radius:4px;
  display:block;}
.crop-cell img.crop{max-width:280px;border:1px solid var(--line);border-radius:4px;
  display:block;background:#000;}
.noimg{color:var(--muted);font-size:12px;}
.value-cell input{width:90px;padding:6px 8px;font-size:15px;font-weight:600;
  border:1px solid var(--line);border-radius:6px;}
.value-cell input.filled{border-color:var(--gap);background:#fff7ed;
  color:#9a3412;}
.foot{margin-top:20px;color:var(--muted);font-size:12px;}
.toolbar{display:flex;align-items:center;gap:12px;margin:0 0 14px;flex-wrap:wrap;
  padding:12px 14px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;}
.toolbar #progress{font-weight:600;color:var(--gap);font-size:15px;
  min-width:110px;}
.toolbar button{font-size:14px;font-weight:600;padding:8px 16px;border-radius:8px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);
  cursor:pointer;}
.toolbar button:hover{background:#eef1f4;}
.toolbar button.primary{background:var(--gap);border-color:var(--gap);
  color:#fff;}
.toolbar button.primary:hover{background:#9a3412;}
"""

#: Client-side script that makes the worksheet a real fill-in form: it counts
#: filled cells, lets the user download the completed CSV (exactly the shape
#: ``stack-apply`` expects: frame,slot_id,value), and clear all in one click.
#: The page still never writes the dataset — only the downloaded CSV + the CLI
#: apply step do.
_JS_STACK = """
(function () {
  var inputs = Array.prototype.slice.call(
    document.querySelectorAll('.ws .value-cell input')
  );
  var progress = document.getElementById('progress');
  var download = document.getElementById('download');
  var clearBtn = document.getElementById('clear');
  var total = inputs.length;

  function filledCount() {
    var n = 0;
    inputs.forEach(function (i) {
      if (i.value.trim() !== '') n += 1;
    });
    return n;
  }

  function refresh() {
    var n = filledCount();
    progress.textContent = '已填 ' + n + ' / ' + total;
    inputs.forEach(function (i) {
      i.classList.toggle('filled', i.value.trim() !== '');
    });
  }

  function buildCsv() {
    var rows = ['frame,slot_id,value'];
    inputs.forEach(function (i) {
      var frame = i.getAttribute('data-frame').replace(/"/g, '""');
      var slot = i.getAttribute('data-slot');
      var value = i.value.trim();
      rows.push('"' + frame + '",' + slot + ',' + value);
    });
    return '\\ufeff' + rows.join('\\r\\n') + '\\r\\n';
  }

  function downloadCsv() {
    var blob = new Blob([buildCsv()], { type: 'text/csv;charset=utf-8;' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'stack-values.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      inputs.forEach(function (i) { i.value = ''; });
      refresh();
    });
  }
  if (download) {
    download.addEventListener('click', downloadCsv);
  }
  inputs.forEach(function (i) {
    i.addEventListener('input', refresh);
    i.addEventListener('keydown', function (e) {
      // Enter jumps to the next input for a fast transcription flow.
      if (e.key === 'Enter') {
        e.preventDefault();
        var idx = parseInt(i.getAttribute('data-index'), 10);
        var next = inputs[idx + 1];
        if (next) next.focus();
      }
    });
  });
  refresh();
})();
"""


__all__ = [
    "ApplyResult",
    "StackGap",
    "apply_stack_values",
    "collect_stack_gaps",
    "render_stack_csv",
    "render_stack_worksheet",
]
