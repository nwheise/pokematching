#!/usr/bin/env python3
"""
Embedding-based Pokemon TCG card identification — proof of concept.

Reference index: card_images/          (one image per card, from match_cards.py)
Query images:    label_studio_export/  (YOLO-labelled video frames; crops class 2=card, 3=multicard)

Output: embedding_poc/output/  — original frames annotated with bounding boxes and top-1 match labels

Run:
    python embedding_poc/evaluate.py
"""

import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from timm.data import resolve_data_config
from tqdm import tqdm

CARD_IMAGES_DIR = Path("card_images")
FRAMES_DIR = Path("label_studio_export/images")
LABELS_DIR = Path("label_studio_export/labels")
TCG_CARDS_DIR = Path("pokemon-tcg-data/cards/en")
OUTPUT_DIR = Path("embedding_poc/output")
EMBED_CACHE = Path("embedding_poc/ref_embeddings.npz")

TOP_N = 5

BOX_COLOR = (50, 220, 50)       # green box outline
TEXT_BG   = (20, 20, 20)        # dark label background
TEXT_COLOR = (255, 255, 255)    # white text
BOX_WIDTH  = 3


# ---------------------------------------------------------------------------
# Card catalog from pokemon-tcg-data
# ---------------------------------------------------------------------------

def build_card_catalog() -> dict[str, dict]:
    """Return {card_id: {"name": ..., "supertype": ..., "subtypes": [...]}} for all cards."""
    catalog = {}
    for json_path in TCG_CARDS_DIR.glob("*.json"):
        try:
            for card in json.loads(json_path.read_text()):
                catalog[card["id"]] = {
                    "name": card.get("name", card["id"]),
                    "supertype": card.get("supertype", ""),
                    "subtypes": card.get("subtypes", []),
                }
        except Exception:
            pass
    return catalog


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
    """Preprocessing callable from timm's data config.
    Avoids all torchvision PIL→tensor helpers that break under numpy 2.x."""
    cfg = resolve_data_config(model.pretrained_cfg)
    input_size = cfg["input_size"]          # (C, H, W)
    h, w = input_size[1], input_size[2]
    crop_pct = cfg.get("crop_pct", 0.875)
    scale_size = int(round(min(h, w) / crop_pct))
    mean = torch.tensor(list(cfg["mean"]), dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(list(cfg["std"]),  dtype=torch.float32).view(3, 1, 1)

    def transform(img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, scale_size)
        img = TF.center_crop(img, (h, w))
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
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x0 = max(0,    int((xc - bw / 2) * W))
            y0 = max(0,    int((yc - bh / 2) * H))
            x1 = min(W,    int((xc + bw / 2) * W))
            y1 = min(H,    int((yc + bh / 2) * H))
            if x1 <= x0 or y1 <= y0:
                continue
            results.append({
                "frame_stem": label_path.stem,
                "frame_path": img_path,
                "cls": cls,
                "box": (x0, y0, x1, y1),
                "crop": frame.crop((x0, y0, x1, y1)),
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
        """Embed a list of PIL images → (N, D) float32 L2-normalised."""
        tensors = torch.stack([self.transform(img) for img in images])
        feats = np.array(self.model(tensors).tolist(), dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return feats / norms

    @torch.no_grad()
    def embed_single(self, img: Image.Image) -> tuple[np.ndarray, float]:
        """Embed one image; return ((D,) embedding, inference_ms)."""
        tensor = self.transform(img).unsqueeze(0)
        t0 = time.perf_counter()
        feats = np.array(self.model(tensor).tolist(), dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        norm = np.linalg.norm(feats)
        return feats[0] / (norm if norm > 0 else 1.0), elapsed_ms


# ---------------------------------------------------------------------------
# Index building (with disk cache)
# ---------------------------------------------------------------------------

def _cache_key(card_paths: list[Path]) -> str:
    """A fingerprint of the current card image set — sorted stems joined."""
    return ",".join(p.stem for p in card_paths)


def build_index(
    model: EmbeddingModel, card_paths: list[Path], batch_size: int = 64
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings [N,D], card_ids [N]), loading from cache when valid."""
    current_key = _cache_key(card_paths)

    if EMBED_CACHE.exists():
        data = np.load(EMBED_CACHE, allow_pickle=True)
        if str(data["cache_key"]) == current_key:
            print(f"  Loaded reference embeddings from cache ({EMBED_CACHE})")
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
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_CACHE, embeddings=embeddings, card_ids=np.array(card_ids), cache_key=current_key)
    print(f"  Saved reference embeddings to {EMBED_CACHE}")
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
    """Return top-N (card_id, cosine_similarity), optionally restricted to valid_mask."""
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
    """Draw bounding boxes + top-1 match labels onto a copy of the frame and save."""
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        card_id, sim = det["top"][0]
        card_name = catalog.get(card_id, {}).get("name", card_id)
        label = f"{card_name}  [{card_id}]  {sim:.3f}"

        # Bounding box
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)

        # Label background + text just above the box
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
    for d in (CARD_IMAGES_DIR, FRAMES_DIR, LABELS_DIR, TCG_CARDS_DIR):
        if not d.is_dir():
            raise SystemExit(f"Error: directory not found: {d}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building card catalog from pokemon-tcg-data...")
    catalog = build_card_catalog()
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

    # Load model + build index
    model = EmbeddingModel("MobileNetV4", "mobilenetv4_conv_small.e2400_r224_in1k")

    print("\n[Phase 1] Building reference index...")
    ref_embs, card_ids = build_index(model, card_paths)

    # Build per-class masks and report counts
    class_masks = build_class_masks(card_ids, catalog)
    n_energy = int(class_masks[0].sum())
    n_tool   = int(class_masks[1].sum())
    print(f"  Class 0 (energy): {n_energy} valid cards, Class 1 (item): {n_tool} valid cards")
    if n_energy == 0:
        print("  WARNING: no Energy cards found in index — class 0 will fall back to full search")
    if n_tool == 0:
        print("  WARNING: no Pokémon Tool cards found in index — class 1 will fall back to full search")

    # Run inference on crops
    print("\n[Phase 2] Matching crops...")
    inf_ms = []

    # Group crops by frame so we can draw all boxes onto each frame at once
    frames: dict[str, list[dict]] = {}
    for entry in crops:
        frames.setdefault(entry["frame_stem"], []).append(entry)

    for frame_stem, detections in tqdm(frames.items(), desc="  Matching frames"):
        for det in detections:
            emb, ms = model.embed_single(det["crop"])
            inf_ms.append(ms)
            valid_mask = class_masks.get(det["cls"])  # None for cls 2/3
            det["top"] = match_crop(emb, ref_embs, card_ids, valid_mask=valid_mask)

        out_path = OUTPUT_DIR / f"{frame_stem}_annotated.png"
        draw_results(detections[0]["frame_path"], detections, catalog, out_path)

    # Timing summary
    print("\n" + "=" * 55)
    print("  TIMING SUMMARY  (MobileNetV4)")
    print("=" * 55)
    print(f"  Median: {np.median(inf_ms):.1f} ms   Mean: {np.mean(inf_ms):.1f} ms")
    print(f"  Total crops matched: {len(crops)}")
    print("=" * 55)
    print(f"\nAnnotated frames saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
