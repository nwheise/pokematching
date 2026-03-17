"""Shared card catalog and reference image downloading for matching approaches."""

import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

CARDS_DIR = Path("pokemon-tcg-data/cards/en")
SETS_FILE = Path("pokemon-tcg-data/sets/en.json")
CARD_IMAGES_DIR = Path("data/card_images")


def build_catalog(standard_only: bool = True) -> list[dict]:
    """Build a catalog of cards from pokemon-tcg-data JSON files.

    Args:
        standard_only: If True, only include Standard-legal cards from Standard-legal sets.

    Returns:
        List of card dicts with keys: id, name, small_url, supertype, subtypes.
    """
    sets = json.loads(SETS_FILE.read_text())

    if standard_only:
        set_ids = {
            s["id"] for s in sets
            if s.get("legalities", {}).get("standard") == "Legal"
        }
    else:
        set_ids = {s["id"] for s in sets}

    catalog = []
    for set_id in set_ids:
        json_file = CARDS_DIR / f"{set_id}.json"
        if not json_file.exists():
            continue
        for card in json.loads(json_file.read_text()):
            if standard_only and card.get("legalities", {}).get("standard") != "Legal":
                continue
            small_url = card.get("images", {}).get("small")
            if not small_url:
                continue
            catalog.append({
                "id": card["id"],
                "name": card.get("name", card["id"]),
                "small_url": small_url,
                "supertype": card.get("supertype", ""),
                "subtypes": card.get("subtypes", []),
            })

    print(f"Catalog: {len(catalog)} cards" + (" (standard-legal)" if standard_only else ""))
    return catalog


def download_images(catalog: list[dict], dest_dir: Path = CARD_IMAGES_DIR) -> None:
    """Download card images that aren't already cached.

    Args:
        catalog: List of card dicts (must have 'id' and 'small_url' keys).
        dest_dir: Directory to save images to.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    to_download = [c for c in catalog if not (dest_dir / f"{c['id']}.png").exists()]
    if not to_download:
        print("All reference images already cached.")
        return

    min_interval = 1.0 / 10  # 10 requests per second max
    session = requests.Session()
    last_request = 0.0
    for card in tqdm(to_download, desc="Downloading card images"):
        dest = dest_dir / f"{card['id']}.png"
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        try:
            resp = session.get(card["small_url"], timeout=10)
            last_request = time.monotonic()
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as e:
            last_request = time.monotonic()
            print(f"  Warning: failed to download {card['id']}: {e}", file=sys.stderr)
