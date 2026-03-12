#!/usr/bin/env python3
"""Extract YOLO-labeled regions from frames and save as individual crops."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageStat
import re

CLASSES = {0: "attached-energy", 1: "attached-item", 2: "card", 3: "multicard"}
CARD_CLASS = 2
OVERLAY_CLASSES = {0, 1}  # attached-energy, attached-item
BORDER_FRAC = 0.1  # halo width as a fraction of each intersection rect's smaller dimension

def mask_overlapping_regions(
    crop: Image.Image, crop_box: tuple, all_boxes: list
) -> Image.Image:
    """Fill attached-energy/item regions that overlap crop_box with the local mean color.

    Args:
        crop: PIL Image of the cropped card region
        crop_box: (x1, y1, x2, y2) pixel coords of the crop in frame space
        all_boxes: list of (cls, x1, y1, x2, y2) for all labeled regions in the frame
    Returns:
        The crop image with overlapping overlay regions filled with the local mean of
        surrounding pixels (mutates in place).
    """
    x1, y1, x2, y2 = crop_box
    W, H = crop.size

    # Pass 1: collect intersection rectangles in crop-local coordinates
    intersections = []
    for oc_id, ox1, oy1, ox2, oy2 in all_boxes:
        if oc_id not in OVERLAY_CLASSES:
            continue
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        if ix1 < ix2 and iy1 < iy2:
            intersections.append((ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1))

    if not intersections:
        return crop

    # Whole-crop mean as fallback when halo is entirely masked
    stat_mask_global = Image.new("L", crop.size, 255)
    global_mask_draw = ImageDraw.Draw(stat_mask_global)
    for rect in intersections:
        global_mask_draw.rectangle(rect, fill=0)
    whole_mean = tuple(int(v) for v in ImageStat.Stat(crop, mask=stat_mask_global).mean[:3])

    # Pass 2: compute all local means from the original (unmodified) crop, then fill
    orig = crop.copy()
    fill_colors = []
    for lx1, ly1, lx2, ly2 in intersections:
        # Border relative to this rect's smaller dimension
        border = max(1, int(min(lx2 - lx1, ly2 - ly1) * BORDER_FRAC))
        hx1, hy1 = max(0, lx1 - border), max(0, ly1 - border)
        hx2, hy2 = min(W, lx2 + border), min(H, ly2 + border)

        # Build mask: halo region white, all intersection rects black
        local_mask = Image.new("L", crop.size, 0)
        lm_draw = ImageDraw.Draw(local_mask)
        lm_draw.rectangle((hx1, hy1, hx2, hy2), fill=255)
        for rect in intersections:
            lm_draw.rectangle(rect, fill=0)

        if local_mask.getbbox() is None:
            fill_colors.append(whole_mean)
        else:
            fill_colors.append(tuple(int(v) for v in ImageStat.Stat(orig, mask=local_mask).mean[:3]))

    draw = ImageDraw.Draw(crop)
    for rect, color in zip(intersections, fill_colors):
        draw.rectangle(rect, fill=color)

    return crop


if __name__ == "__main__":
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
                mask_overlapping_regions(crop, (x1, y1, x2, y2), boxes)
            class_name = CLASSES.get(class_id, str(class_id))
            out_path = output_dir / f"{frame_name}_{i}_{class_name}.png"
            crop.save(out_path)
            total += 1

    print(f"Saved {total} crops to {output_dir}/")
