"""Private contact sheets from already extracted AA8 frames; not predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def render(pool, layout_path, frames, output):
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if layout["canvas"] != [498, 1080] or len(layout["slots"]) != 8:
        raise ValueError("requires the explicit AA8 canvas and slots")
    manifest = json.loads((pool / "samples.json").read_text(encoding="utf-8"))
    rows = {r["global_frame"]: r for r in manifest["samples"]}
    if not frames or any(f not in rows for f in frames):
        raise ValueError("requested frame is outside extracted pool")
    output.mkdir(parents=True, exist_ok=False)
    evidence = []
    for page, start in enumerate(range(0, len(frames), 8)):
        selected = frames[start:start + 8]
        sheet = Image.new("RGB", (1000, 168 * len(selected)), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(selected):
            row = rows[frame]
            path = (pool / row["file"]).resolve()
            if not path.is_relative_to(pool.resolve()):
                raise ValueError("frame path escapes pool")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError("frame hash mismatch")
            with Image.open(path) as source:
                if source.size != (498, 1080):
                    raise ValueError("unexpected frame canvas")
                y = index * 168
                draw.text((4, y + 3), f"frame {frame}", fill="black")
                for slot in layout["slots"]:
                    x0, y0, w, h = slot["avatar"]
                    x = 90 + 82 * slot["slot"]
                    draw.text((x, y + 3), str(slot["slot"]), fill="black")
                    sheet.paste(source.crop((max(0, x0 - 2), y0 - 39,
                                             min(498, x0 + w + 2), y0 + 99)),
                                (x, y + 20))
                sheet.paste(source.crop((109, 472, 389, 553)).resize((240, 69)),
                            (752, y + 24))
                sheet.paste(source.crop((198, 278, 302, 301)), (752, y + 100))
            evidence.append({"global_frame": frame, "sha256": digest,
                             "source": str(path), "sheet": page})
        sheet.save(output / f"actions_{page:02d}.png")
    (output / "manifest.json").write_text(json.dumps({
        "purpose": "manual_review_only_not_predictions",
        "audit_sha256": manifest["audit_sha256"],
        "layout_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
        "frames": evidence}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pool", "layout", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, default=30)
    args = parser.parse_args()
    if args.step <= 0 or args.start > args.end:
        parser.error("positive step and ordered frame interval required")
    render(args.pool, args.layout, list(range(args.start, args.end + 1, args.step)),
           args.output)
