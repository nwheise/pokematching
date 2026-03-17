#!/usr/bin/env python3
"""Match extracted card crops against Pokemon TCG reference images using ORB feature matching."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matching.card_catalog import CARD_IMAGES_DIR, build_catalog, download_images

import cv2
import numpy as np
from tqdm import tqdm

EXTRACTED_DIR = Path("outputs/extracted_regions")
DESC_CACHE = Path("outputs/orb/card_descriptors.npz")
OUTPUT_FILE = Path("outputs/match_results/match_results.json")
REF_WIDTH = 245
ORB_FEATURES = 500
TOP_N = 5


# --- Load or compute ORB descriptor index ---

def _cached_image_ids():
    return sorted(p.stem for p in CARD_IMAGES_DIR.glob("*.png"))

def load_or_build_index(catalog):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    if DESC_CACHE.exists():
        data = np.load(DESC_CACHE, allow_pickle=True)
        cached_ids = data["image_ids"].tolist()
        if cached_ids == _cached_image_ids():
            print("Loaded ORB descriptor index from cache.")
            return orb, data["all_desc"], data["card_idx"].astype(np.int32), \
                   data["card_ids"].tolist(), data["card_names"].tolist()
        print("Descriptor cache stale — rebuilding...")

    catalog_map = {c["id"]: c["name"] for c in catalog}

    all_desc_list = []
    card_idx_list = []
    card_ids = []
    card_names = []
    missing = 0

    for path in tqdm(sorted(CARD_IMAGES_DIR.glob("*.png")), desc="Computing ORB descriptors"):
        card_id = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue
        _, des = orb.detectAndCompute(img, None)
        if des is None:
            continue
        idx = len(card_ids)
        card_ids.append(card_id)
        card_names.append(catalog_map.get(card_id, card_id))
        all_desc_list.append(des)
        card_idx_list.append(np.full(len(des), idx, dtype=np.int32))

    if missing:
        print(f"  Skipped {missing} unreadable images")

    all_desc = np.vstack(all_desc_list)
    card_idx = np.concatenate(card_idx_list)

    DESC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DESC_CACHE,
        all_desc=all_desc,
        card_idx=card_idx,
        card_ids=np.array(card_ids),
        card_names=np.array(card_names),
        image_ids=np.array(_cached_image_ids()),
    )
    print(f"Indexed {len(card_ids)} cards ({len(all_desc)} descriptors) — saved to {DESC_CACHE}")
    return orb, all_desc, card_idx, card_ids, card_names


# --- Match crops ---

CHUNK = 200_000

def match_crop(crop_path, orb, all_desc, card_idx, card_ids, card_names, bf):
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    new_h = int(h * REF_WIDTH / w)
    img = cv2.resize(img, (REF_WIDTH, new_h))

    _, des = orb.detectAndCompute(img, None)
    if des is None:
        return []

    scores = np.zeros(len(card_ids), dtype=np.int32)
    for start in range(0, len(all_desc), CHUNK):
        chunk_desc = all_desc[start:start + CHUNK]
        chunk_idx  = card_idx[start:start + CHUNK]
        pairs = bf.knnMatch(des, chunk_desc, k=2)
        good = [p for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if good:
            train_idx = np.array([p[0].trainIdx for p in good], dtype=np.int32)
            np.add.at(scores, chunk_idx[train_idx], 1)

    top_indices = np.argpartition(scores, -TOP_N)[-TOP_N:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        {"card_id": card_ids[i], "name": card_names[i], "score": int(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]


def group_crops_by_frame(crops):
    groups = defaultdict(list)
    for p in crops:
        m = re.match(r"(frame_\d+)", p.name)
        key = m.group(1) if m else "unknown"
        groups[key].append(p)
    return dict(sorted(groups.items()))


def main():
    catalog = build_catalog()
    download_images(catalog)
    orb, all_desc, card_idx, card_ids, card_names = load_or_build_index(catalog)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    crops = sorted(
        p for p in EXTRACTED_DIR.glob("*.png")
        if "_card" in p.name or "_multicard" in p.name
    )

    frames = group_crops_by_frame(crops)
    print(f"\nProcessing {len(crops)} crops across {len(frames)} frames...")

    results = []
    for frame_name, frame_crops in tqdm(frames.items(), desc="Frames"):
        print(f"\n[{frame_name}] — {len(frame_crops)} crop(s)")
        for crop_path in frame_crops:
            matches = match_crop(crop_path, orb, all_desc, card_idx, card_ids, card_names, bf)
            results.append({"crop": crop_path.name, "matches": matches})
            top = matches[0] if matches else None
            if top:
                print(f"  {crop_path.name:35s} -> {top['name']:30s} ({top['card_id']}, score={top['score']})")
            else:
                print(f"  {crop_path.name:35s} -> no match")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
