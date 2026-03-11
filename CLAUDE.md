# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pipeline for identifying Pokemon TCG cards from video frames using computer vision. The workflow is:

1. **Label Studio** annotates video frames with YOLO bounding boxes for card regions
2. **`extract_regions.py`** crops those regions from frames into individual images
3. **`match_cards.py`** matches the crops against reference card images using ORB feature matching

## Running the Pipeline

```bash
# Step 1: Extract labeled regions from frames
python extract_regions.py
# Reads: label-studio-export/labels/*.txt + frames/*.png
# Writes: extracted/*.png

# Step 2: Match crops to Pokemon TCG cards
python match_cards.py
# Downloads reference card images to card_images/ on first run
# Caches ORB descriptors to card_descriptors.npz
# Writes: match_results.json
```

## Label Studio

```bash
source ~/.venvs/label-studio/bin/activate
label-studio start
# UI at http://localhost:8080
```

Uses Python 3.11.9 via pyenv (avoids `pkgutil.find_loader` error in Python 3.14). Label Studio has its own separate venv at `~/.venvs/label-studio` (install: `~/.pyenv/versions/3.11.9/bin/python -m venv ~/.venvs/label-studio && source ~/.venvs/label-studio/bin/activate && pip install label-studio`).

## Setup

Create and activate the project virtualenv using Python 3.11.9 via pyenv:

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv ~/.venvs/pokematching-venv
source ~/.venvs/pokematching-venv/bin/activate
pip install -r requirements.txt
```

## Key Directories (runtime-generated, not in git)

- `frames/` — source video frame PNGs
- `label-studio-export/labels/` — YOLO-format `.txt` label files from Label Studio export
- `extracted/` — cropped card regions output by `extract_regions.py`
- `card_images/` — downloaded reference card images (cached)
- `card_descriptors.npz` — cached ORB descriptor index (invalidated when `card_images/` changes)
- `match_results.json` — final match output

## Architecture Notes

**`extract_regions.py`** is a simple script; frame filenames must contain `frame_\d+` to match label files to images. YOLO label format: `class_id x_center y_center width height` (normalized 0–1). Classes: `{0: attached-energy, 1: attached-item, 2: card, 3: multicard}`.

**`match_cards.py`** has four phases:
1. **Catalog** — reads `pokemon-tcg-data/` submodule JSON to find Standard-legal cards with image URLs
2. **Download** — fetches card images at ≤10 req/s, cached in `card_images/`
3. **Index** — computes ORB descriptors for all reference images, stacked into a single matrix for batch matching; cached in `card_descriptors.npz`
4. **Match** — for each crop, runs BFMatcher with Lowe's ratio test (0.75) against the descriptor index in 200k-row chunks; returns top-5 matches by inlier count

The submodule `pokemon-tcg-data` provides card/set JSON data from `git@github.com:PokemonTCG/pokemon-tcg-data.git`.
