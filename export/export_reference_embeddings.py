#!/usr/bin/env python3
"""Export precomputed reference card embeddings for the full standard-legal catalog.

Reuses the existing EmbeddingModel and build_index from match_cards.py,
including the disk cache in outputs/embeddings/.

Outputs:
  <out>/reference_embeddings.npz        — {embeddings: float32 [N, D], card_ids: str[N]}
  <out>/reference_embeddings_meta.json  — {card_ids: [...], cards: {id: {name, supertype, subtypes}}}

Usage:
    python export/export_reference_embeddings.py
    python export/export_reference_embeddings.py --out outputs/export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matching.card_catalog import CARD_IMAGES_DIR, build_catalog, download_images
from matching.embedding.match_cards import EmbeddingModel, build_index

DEFAULT_MODEL_ID = "mobilenetv4_conv_small.e2400_r224_in1k"


def main():
    parser = argparse.ArgumentParser(description="Export reference card embeddings")
    parser.add_argument("--out", default="outputs/export",
                        help="Output directory")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="timm model ID (must match the model used for ONNX export)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build catalog and ensure reference images are present
    catalog = build_catalog(standard_only=True)
    download_images(catalog)

    card_paths = sorted(
        p for c in catalog
        if (p := CARD_IMAGES_DIR / f"{c['id']}.png").exists()
    )
    if not card_paths:
        print("Error: no card images found in data/card_images/", file=sys.stderr)
        sys.exit(1)

    # Build or load embedding index (reuses cache in outputs/embeddings/)
    model = EmbeddingModel("MobileNetV4", args.model_id)
    embeddings, card_ids = build_index(model, card_paths)

    # Save embeddings
    npz_dest = out_dir / "reference_embeddings.npz"
    np.savez(npz_dest, embeddings=embeddings, card_ids=np.array(card_ids))
    print(f"Saved {npz_dest} — shape {embeddings.shape}")

    # Save metadata (id → name, supertype, subtypes) for JS-side class filtering
    meta_by_id = {
        c["id"]: {"name": c["name"], "supertype": c["supertype"], "subtypes": c["subtypes"]}
        for c in catalog
    }
    meta = {
        "card_ids": card_ids,
        "cards": {cid: meta_by_id[cid] for cid in card_ids if cid in meta_by_id},
    }
    meta_path = out_dir / "reference_embeddings_meta.json"
    meta_path.write_text(json.dumps(meta))
    print(f"Saved {meta_path} — {len(card_ids)} cards")


if __name__ == "__main__":
    main()
