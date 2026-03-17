"""Shared constants and utilities for the pokematching pipeline."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageStat

# YOLO class definitions
CLASSES = {0: "attached-energy", 1: "attached-item", 2: "card", 3: "multicard"}
CARD_CLASS = 2
OVERLAY_CLASSES = {0, 1}  # attached-energy, attached-item
BORDER_FRAC = 0.1  # halo width as a fraction of each intersection rect's smaller dimension


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Parse a YOLO label file into pixel-coordinate bounding boxes.

    Args:
        label_path: Path to a YOLO-format .txt label file
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        List of (class_id, x1, y1, x2, y2) tuples in pixel coordinates.
    """
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, int((xc - bw / 2) * img_w))
        y1 = max(0, int((yc - bh / 2) * img_h))
        x2 = min(img_w, int((xc + bw / 2) * img_w))
        y2 = min(img_h, int((yc + bh / 2) * img_h))
        if x2 > x1 and y2 > y1:
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


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
