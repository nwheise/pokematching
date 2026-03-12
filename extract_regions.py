#!/usr/bin/env python3
"""Extract YOLO-labeled regions from frames and save as individual crops."""

from pathlib import Path
from PIL import Image, ImageDraw
import re

CLASSES = {0: "attached-energy", 1: "attached-item", 2: "card", 3: "multicard"}
CARD_CLASS = 2
OVERLAY_CLASSES = {0, 1}  # attached-energy, attached-item

labels_dir = Path("label_studio_export/labels")
frames_dir = Path("label_studio_export/images")
output_dir = Path("extracted_regions")
output_dir.mkdir(exist_ok=True)

total = 0
for label_file in sorted(labels_dir.glob("*.txt")):
    lines = label_file.read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    if not lines:
        continue

    match = re.search(r"(frame_\d+)", label_file.stem)
    if not match:
        print(f"Skipping {label_file.name}: can't parse frame number")
        continue
    frame_name = match.group(1)

    image_path = frames_dir / f"{label_file.stem}.png"
    if not image_path.exists():
        print(f"Missing image: {image_path}")
        continue

    img = Image.open(image_path)
    W, H = img.size

    boxes = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)
        boxes.append((class_id, x1, y1, x2, y2))

    for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
        crop = img.crop((x1, y1, x2, y2))
        if class_id == CARD_CLASS:
            draw = ImageDraw.Draw(crop)
            for oc_id, ox1, oy1, ox2, oy2 in boxes:
                if oc_id not in OVERLAY_CLASSES:
                    continue
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    draw.rectangle([ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1], fill=(0, 0, 0))
        class_name = CLASSES.get(class_id, str(class_id))
        out_path = output_dir / f"{frame_name}_{i}_{class_name}.png"
        crop.save(out_path)
        total += 1

print(f"Saved {total} crops to {output_dir}/")
