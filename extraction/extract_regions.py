#!/usr/bin/env python3
"""Extract YOLO-labeled regions from frames and save as individual crops."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from utils import CARD_CLASS, CLASSES, mask_overlapping_regions, parse_yolo_labels

if __name__ == "__main__":
    labels_dir = Path("data/labels")
    images_dir = Path("data/images")
    output_dir = Path("outputs/extracted_regions")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for label_file in sorted(labels_dir.glob("*.txt")):
        match = re.search(r"(frame_\d+)", label_file.stem)
        if not match:
            print(f"Skipping {label_file.name}: can't parse frame number")
            continue
        frame_name = match.group(1)

        image_path = images_dir / f"{label_file.stem}.png"
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        img = Image.open(image_path)
        W, H = img.size
        boxes = parse_yolo_labels(label_file, W, H)

        for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
            crop = img.crop((x1, y1, x2, y2))
            if class_id == CARD_CLASS:
                mask_overlapping_regions(crop, (x1, y1, x2, y2), boxes)
            class_name = CLASSES.get(class_id, str(class_id))
            out_path = output_dir / f"{frame_name}_{i}_{class_name}.png"
            crop.save(out_path)
            total += 1

    print(f"Saved {total} crops to {output_dir}/")
