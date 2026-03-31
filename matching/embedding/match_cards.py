#!/usr/bin/env python3
"""
Embedding-based Pokemon TCG card identification.

Reference index: data/card_images/       (one image per card)
Query images:    data/images/ + data/labels/  (YOLO-labelled video frames)

Output: outputs/match_results/{model}/  — original frames annotated with bounding boxes and top-1 match labels

Run:
    python matching/embedding/match_cards.py
    python matching/embedding/match_cards.py --deck_file data/deck.txt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matching.card_catalog import CARD_IMAGES_DIR, build_catalog, download_images
from utils import OVERLAY_CLASSES, mask_overlapping_regions, parse_yolo_labels

import numpy as np
import timm
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from timm.data import resolve_data_config
from tqdm import tqdm

FRAMES_DIR = Path("data/images")
LABELS_DIR = Path("data/labels")
OUTPUT_BASE_DIR = Path("outputs/match_results")
EMBED_CACHE_DIR = Path("outputs/embeddings")

TOP_N = 5

BOX_COLOR = (50, 220, 50)       # green box outline
TEXT_BG   = (20, 20, 20)        # dark label background
TEXT_COLOR = (255, 255, 255)    # white text
BOX_WIDTH  = 3


# ---------------------------------------------------------------------------
# Card catalog helpers
# ---------------------------------------------------------------------------

def catalog_by_id(catalog: list[dict]) -> dict[str, dict]:
    """Convert catalog list to {card_id: {name, supertype, subtypes}} dict for lookups."""
    return {
        c["id"]: {"name": c["name"], "supertype": c["supertype"], "subtypes": c["subtypes"]}
        for c in catalog
    }


# ---------------------------------------------------------------------------
# Deck file parsing
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r'^(Pok[eé]mon|Trainer|Energy):\s*\d+', re.IGNORECASE)
TOTAL_RE  = re.compile(r'^Total Cards:\s*\d+', re.IGNORECASE)
CARD_RE   = re.compile(r'^(\d+)\s+(.+?)\s+([A-Z0-9-]+)\s+(\d+)\s*$')


def parse_deck_file(deck_path: Path) -> set[str]:
    """
    Parse a PTCGL-format deck list and return a set of resolved card IDs.

    Expected format per card line:
        <count> <name> <SET_CODE> <number>
    e.g.:
        3 Dunsparce TEF 128
    """
    sets_path = Path("pokemon-tcg-data/sets/en.json")
    if not sets_path.exists():
        raise SystemExit(f"Error: sets file not found: {sets_path}")

    with open(sets_path) as f:
        sets_data = json.load(f)

    ptcgo_to_id: dict[str, str] = {}
    for s in sets_data:
        code = s.get("ptcgoCode", "")
        if code:
            ptcgo_to_id[code.upper()] = s["id"]

    card_ids: set[str] = set()
    unresolved_codes: set[str] = set()
    parse_failures = 0

    with open(deck_path) as f:
        for line in f:
            line = line.rstrip()
            if not line or HEADER_RE.match(line) or TOTAL_RE.match(line):
                continue
            m = CARD_RE.match(line)
            if not m:
                print(f"  WARNING: could not parse deck line: {line!r}")
                parse_failures += 1
                continue
            count, _name, set_code, number = m.groups()
            if int(count) <= 0:
                continue
            set_id = ptcgo_to_id.get(set_code.upper())
            if set_id is None:
                unresolved_codes.add(set_code)
                continue
            card_ids.add(f"{set_id}-{number}")

    for code in sorted(unresolved_codes):
        print(f"  WARNING: set code not found in pokemon-tcg-data: {code!r}")
    if parse_failures:
        print(f"  WARNING: {parse_failures} line(s) could not be parsed")

    print(f"  Deck loaded: {len(card_ids)} unique cards from {deck_path}")
    return card_ids


# ---------------------------------------------------------------------------
# Class-based filtering
# ---------------------------------------------------------------------------

CLASS_FILTERS = {
    0: lambda meta: meta.get("supertype") == "Energy",
    1: lambda meta: meta.get("supertype") == "Trainer" and "Pokémon Tool" in meta.get("subtypes", []),
}


def build_class_masks(card_ids: list[str], catalog: dict[str, dict]) -> dict[int, np.ndarray]:
    """Return {class_id: bool mask [N]} for classes with restricted search spaces."""
    masks = {}
    for cls, predicate in CLASS_FILTERS.items():
        masks[cls] = np.array([predicate(catalog.get(cid, {})) for cid in card_ids], dtype=bool)
    return masks


# ---------------------------------------------------------------------------
# Preprocessing transform
# ---------------------------------------------------------------------------

def _make_transform(model):
    cfg = resolve_data_config(model.pretrained_cfg)
    input_size = cfg["input_size"]
    h, w = input_size[1], input_size[2]
    mean = torch.tensor(list(cfg["mean"]), dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(list(cfg["std"]),  dtype=torch.float32).view(3, 1, 1)

    def transform(img: Image.Image) -> torch.Tensor:
        orig_w, orig_h = img.size
        new_h = max(1, int(round(orig_h * w / orig_w)))
        img = TF.resize(img, [new_h, w])
        if img.height >= h:
            img = img.crop((0, 0, w, h))
        else:
            padded = Image.new("RGB", (w, h), (0, 0, 0))
            padded.paste(img, (0, 0))
            img = padded
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return (tensor - mean) / std

    return transform


# ---------------------------------------------------------------------------
# Frame crop loading
# ---------------------------------------------------------------------------

def load_frame_crops(frames_dir: Path, labels_dir: Path) -> list[dict]:
    """
    Parse YOLO labels and crop card-class bounding boxes from each frame.
    Returns list of dicts: {frame_stem, frame_path, cls, box (x0,y0,x1,y1), crop}.
    """
    results = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        img_path = frames_dir / (label_path.stem + ".png")
        if not img_path.exists():
            continue
        try:
            frame = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        W, H = frame.size
        frame_boxes = parse_yolo_labels(label_path, W, H)

        for cls, x0, y0, x1, y1 in frame_boxes:
            crop = frame.crop((x0, y0, x1, y1))
            if cls not in OVERLAY_CLASSES:
                mask_overlapping_regions(crop, (x0, y0, x1, y1), frame_boxes)
            results.append({
                "frame_stem": label_path.stem,
                "frame_path": img_path,
                "cls": cls,
                "box": (x0, y0, x1, y1),
                "crop": crop,
            })

    return results


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    def __init__(self, name: str, model_id: str):
        self.name = name
        print(f"Loading {name} ({model_id})...")
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model.eval()
        self.transform = _make_transform(self.model)

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.transform(img) for img in images])
        feats = np.array(self.model(tensors).tolist(), dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return feats / norms

    @torch.no_grad()
    def embed_single(self, img: Image.Image) -> tuple[np.ndarray, float]:
        tensor = self.transform(img).unsqueeze(0)
        t0 = time.perf_counter()
        feats = np.array(self.model(tensor).tolist(), dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        norm = np.linalg.norm(feats)
        return feats[0] / (norm if norm > 0 else 1.0), elapsed_ms


# ---------------------------------------------------------------------------
# Index building (with disk cache)
# ---------------------------------------------------------------------------

def _cache_key(card_paths: list[Path], model_name: str) -> str:
    return model_name + "|" + ",".join(p.stem for p in card_paths)


def _embed_cache_path(model_name: str) -> Path:
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    return EMBED_CACHE_DIR / f"ref_embeddings_{safe_name}.npz"


def build_index(
    model: EmbeddingModel, card_paths: list[Path], batch_size: int = 64
) -> tuple[np.ndarray, list[str]]:
    cache_path = _embed_cache_path(model.name)
    current_key = _cache_key(card_paths, model.name)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if str(data["cache_key"]) == current_key:
            print(f"  Loaded reference embeddings from cache ({cache_path})")
            return data["embeddings"], data["card_ids"].tolist()
        print("  Embedding cache stale — rebuilding...")
    else:
        print("  No embedding cache found — building...")

    all_embeddings = []
    card_ids = [p.stem for p in card_paths]

    for start in tqdm(range(0, len(card_paths), batch_size), desc=f"  Indexing [{model.name}]"):
        batch = card_paths[start : start + batch_size]
        images = []
        for p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        all_embeddings.append(model.embed_batch(images))

    embeddings = np.vstack(all_embeddings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, card_ids=np.array(card_ids), cache_key=current_key)
    print(f"  Saved reference embeddings to {cache_path}")
    return embeddings, card_ids


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_crop(
    query_emb: np.ndarray,
    ref_embeddings: np.ndarray,
    card_ids: list[str],
    top_n: int = TOP_N,
    valid_mask: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    if valid_mask is not None and valid_mask.any():
        idx = np.where(valid_mask)[0]
        sims_sub = ref_embeddings[idx] @ query_emb
        k = min(top_n, len(idx))
        top_local = np.argpartition(sims_sub, -k)[-k:]
        top_local = top_local[np.argsort(sims_sub[top_local])[::-1]]
        return [(card_ids[idx[i]], float(sims_sub[i])) for i in top_local]
    sims = ref_embeddings @ query_emb
    top_idx = np.argpartition(sims, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    return [(card_ids[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Annotated image output
# ---------------------------------------------------------------------------

def _load_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def draw_results(frame_path: Path, detections: list[dict], catalog: dict[str, dict], out_path: Path) -> None:
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        card_id, sim = det["top"][0]
        card_name = catalog.get(card_id, {}).get("name", card_id)
        label = f"{card_name}  [{card_id}]  {sim:.3f}"

        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        lx = x0
        ly = max(0, y0 - th - pad * 2 - BOX_WIDTH)
        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=TEXT_BG)
        draw.text((lx + pad, ly + pad), label, fill=TEXT_COLOR, font=font)

    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Match card crops to Pokemon TCG cards using embeddings.")
    parser.add_argument(
        "--deck_file",
        metavar="PATH",
        help="Path to a PTCGL-format deck list .txt file. When provided, matching is restricted to only cards in this deck.",
    )
    args = parser.parse_args()

    for d in (CARD_IMAGES_DIR, FRAMES_DIR, LABELS_DIR):
        if not d.is_dir():
            raise SystemExit(f"Error: directory not found: {d}")

    print("Building card catalog from pokemon-tcg-data...")
    catalog_list = build_catalog()
    download_images(catalog_list)
    catalog = catalog_by_id(catalog_list)
    print(f"  {len(catalog)} cards loaded")

    card_paths = sorted(
        p for p in CARD_IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not card_paths:
        raise SystemExit(f"No images found in {CARD_IMAGES_DIR}")
    print(f"Reference cards : {len(card_paths)}")

    print("\nLoading query crops from label studio frames...")
    crops = load_frame_crops(FRAMES_DIR, LABELS_DIR)
    if not crops:
        raise SystemExit("No card crops found — check FRAMES_DIR / LABELS_DIR paths")
    print(f"Query crops     : {len(crops)}")

    # model = EmbeddingModel("DINOv2-ViT-S/14", "vit_small_patch14_dinov2.lvd142m")
    model = EmbeddingModel("MobileNetV4", "mobilenetv4_conv_small.e2400_r224_in1k")
    safe_name = model.name.replace("/", "_").replace(" ", "_")
    output_dir = OUTPUT_BASE_DIR / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 1] Building reference index...")
    ref_embs, card_ids = build_index(model, card_paths)

    class_masks = build_class_masks(card_ids, catalog)
    n_energy = int(class_masks[0].sum())
    n_tool   = int(class_masks[1].sum())
    print(f"  Class 0 (energy): {n_energy} valid cards, Class 1 (item): {n_tool} valid cards")
    if n_energy == 0:
        print("  WARNING: no Energy cards found in index — class 0 will fall back to full search")
    if n_tool == 0:
        print("  WARNING: no Pokémon Tool cards found in index — class 1 will fall back to full search")

    deck_mask = None
    if args.deck_file:
        print(f"\nLoading deck from {args.deck_file}...")
        deck_card_ids = parse_deck_file(Path(args.deck_file))
        deck_mask = np.array([cid in deck_card_ids for cid in card_ids], dtype=bool)
        n_found = int(deck_mask.sum())
        n_missing = len(deck_card_ids) - n_found
        print(f"  Deck mask: {n_found}/{len(deck_card_ids)} deck cards found in index")
        if n_missing > 0:
            missing = deck_card_ids - set(card_ids)
            print(f"  WARNING: {n_missing} deck card(s) not in reference index: {missing}")

    print("\n[Phase 2] Matching crops...")
    inf_ms = []

    frames: dict[str, list[dict]] = {}
    for entry in crops:
        frames.setdefault(entry["frame_stem"], []).append(entry)

    for frame_stem, detections in tqdm(frames.items(), desc="  Matching frames"):
        for det in detections:
            emb, ms = model.embed_single(det["crop"])
            inf_ms.append(ms)
            class_mask = class_masks.get(det["cls"])
            if deck_mask is not None:
                valid_mask = deck_mask & class_mask if class_mask is not None else deck_mask
            else:
                valid_mask = class_mask
            det["top"] = match_crop(emb, ref_embs, card_ids, valid_mask=valid_mask)

        out_path = output_dir / f"{frame_stem}_annotated.png"
        draw_results(detections[0]["frame_path"], detections, catalog, out_path)

    print("\n" + "=" * 55)
    print(f"  TIMING SUMMARY  ({model.name})")
    print("=" * 55)
    print(f"  Median: {np.median(inf_ms):.1f} ms   Mean: {np.mean(inf_ms):.1f} ms")
    print(f"  Total crops matched: {len(crops)}")
    print("=" * 55)
    print(f"\nAnnotated frames saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
