# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pipeline for identifying Pokemon TCG cards from video frames using computer vision. The pipeline stages are:

1. **Labeling** — Label Studio annotates video frames with YOLO bounding boxes
2. **Extraction** (`extraction/`) — Crops labeled regions from frames into individual images
3. **Detection** (`detection/`) — Train and run YOLO object detection models (ultralytics)
4. **Matching** (`matching/`) — Match detected card crops against reference card images
5. **Results** — Output in `outputs/`

## Directory Structure

```
extraction/            # Crop YOLO-labeled regions from frames
detection/             # YOLO model training, inference, dataset prep
  prepare_dataset.py   # Split labels into train/val
  train.py             # Ultralytics YOLO training
  detect.py            # Run trained model on new frames
  dataset.yaml         # Ultralytics dataset config
matching/              # Card identification approaches
  card_catalog.py      # Shared catalog building and reference image downloading
  orb/                 # ORB feature matching
  embedding/           # DINOv2/MobileNetV4 embedding matching
utils/                 # Shared constants and utilities
  common.py            # CLASSES, parse_yolo_labels, mask_overlapping_regions

data/                  # Shared input data (gitignored)
  frames/              # Source video frame PNGs
  labels/              # YOLO-format .txt label files
  card_images/         # Downloaded reference card images

outputs/               # All pipeline outputs (gitignored)
  extracted_regions/   # Crops from extraction stage
  detection/           # Training runs, weights, predictions
  match_results/       # Match outputs, ORB descriptor cache
  embeddings/          # Cached embedding .npz files

pokemon-tcg-data/      # Git submodule: card/set JSON data
```

## Setup

Create and activate the project virtualenv using Python 3.11.9 via pyenv:

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Label Studio

```bash
source venv/bin/activate
label-studio start
# UI at http://localhost:8080
```

For the ML backend to resolve local image paths, Label Studio must be started with:

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
```

## ML Backend (YOLO pre-annotation)

Serves the trained YOLO model to Label Studio for automatic pre-annotation of new frames.

```bash
source venv/bin/activate
label-studio-ml start ml_backend/ --port 9090
# Connect at http://localhost:9090 via Label Studio > Settings > Machine Learning
```

To use a different model checkpoint, set the `YOLO_MODEL_PATH` env var:

```bash
YOLO_MODEL_PATH=outputs/detection/train/weights/best.pt label-studio-ml start ml_backend/ --port 9090
```

Default model path: `outputs/detection/train/weights/best.pt`

**Label Studio labeling config must have `RectangleLabels` with these exact value strings:**
`attached-energy`, `attached-item`, `card`, `multicard`

## Running the Pipeline

```bash
# Step 1: Export labels from Label Studio into data/frames/ and data/labels/

# Step 2: Extract labeled regions from frames
python extraction/extract_regions.py

# Step 3a: Prepare dataset for training
python detection/prepare_dataset.py

# Step 3b: Train YOLO detector
python detection/train.py

# Step 3c: Run detector on new frames
python detection/detect.py

# Step 4: Match crops to Pokemon TCG cards (ORB approach)
python matching/orb/match_cards.py

# Step 4 alt: Match using embeddings
python matching/embedding/evaluate.py
```


## Architecture Notes

**`utils/common.py`** contains shared constants (`CLASSES`, `CARD_CLASS`, `OVERLAY_CLASSES`) and utilities (`parse_yolo_labels`, `mask_overlapping_regions`) used across pipeline stages.

**YOLO label format:** `class_id x_center y_center width height` (normalized 0–1). Classes: `{0: attached-energy, 1: attached-item, 2: card, 3: multicard}`.

**`extraction/extract_regions.py`** — frame filenames must contain `frame_\d+` to match label files to images.

**`matching/card_catalog.py`** — shared module for building the card catalog from `pokemon-tcg-data/` JSON and downloading reference card images to `data/card_images/` (rate-limited to ≤10 req/s). Used by both matching approaches.

**`matching/orb/match_cards.py`** — ORB feature matching: builds descriptor index (cached in `outputs/match_results/card_descriptors.npz`), then matches crops via BFMatcher with Lowe's ratio test (0.75) in 200k-row chunks, returning top-5 by inlier count.

**`matching/embedding/evaluate.py`** — proof-of-concept using DINOv2 or MobileNetV4 embeddings with cosine similarity matching and class-based filtering masks.

**`detection/`** — uses ultralytics YOLOv8. `prepare_dataset.py` splits data into train/val. `train.py` and `detect.py` are minimal wrappers around the ultralytics API.

The submodule `pokemon-tcg-data` provides card/set JSON data from `git@github.com:PokemonTCG/pokemon-tcg-data.git`.
