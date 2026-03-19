# Pokematching

A computer vision pipeline for identifying Pokemon TCG cards from video frames. This repo is the ML/CV backend intended to power the card detection and identification pipeline used by [ptcgl-deck-tracker](https://github.com/nwheise/ptcgl-deck-tracker), a tool for tracking decks in Pokemon TCG Live gameplay footage.

The pipeline takes raw video frames as input and produces identified card names as output, by combining YOLO-based object detection with image matching against a reference card catalog.

## Pipeline Overview

```
Video frames
     │
     ▼
1. Labeling      — Annotate frames in Label Studio (bounding boxes)
     │
     ▼
2. Extraction    — Crop labeled regions into individual card images
     │
     ▼
3. Detection     — Train and run a YOLO model to detect cards in new frames
     │
     ▼
4. Matching      — Match detected card crops against a reference card catalog
     │
     ▼
Identified card names
```

## Setup

Requires Python 3.11.9 via pyenv (avoids `pkgutil.find_loader` error in Python 3.14):

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Initialize the `pokemon-tcg-data` submodule (provides card/set JSON used to build the reference catalog):

```bash
git submodule update --init
```

## Directory Structure

```
extraction/            # Crop YOLO-labeled regions from frames
detection/             # YOLO model training, inference, dataset prep
matching/              # Card identification approaches
  orb/                 # ORB feature matching
  embedding/           # DINOv2/MobileNetV4 embedding matching
  card_catalog.py      # Shared catalog builder + reference image downloader
ml_backend/            # Label Studio ML backend for YOLO pre-annotation
utils/                 # Shared constants and utilities

data/                  # Input data (gitignored)
  frames/              # Source video frame PNGs
  labels/              # YOLO-format .txt label files
  card_images/         # Downloaded reference card images

outputs/               # All pipeline outputs (gitignored)
  extracted_regions/   # Card crops from extraction stage
  detection/           # YOLO training runs, weights, predictions
  match_results/       # Match outputs, ORB descriptor cache
  embeddings/          # Cached embedding .npz files

pokemon-tcg-data/      # Submodule: card/set JSON from PokemonTCG/pokemon-tcg-data
```

## Running the Pipeline

### Setup
```bash
~/.pyenv/versions/3.11.9/bin/python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Labeling (Label Studio)

```bash
source venv/bin/activate
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
# UI at http://localhost:8080
```

> `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` is required for the ML backend to resolve local image paths.

Annotate video frames with bounding boxes and export in YOLO format. Place the exported files in:

- Images → `data/frames/`
- Labels → `data/labels/`

YOLO label format: `class_id x_center y_center width height` (normalized 0–1)

Classes:
- `0`: `attached-energy`
- `1`: `attached-item`
- `2`: `card`
- `3`: `multicard`

#### ML Backend (YOLO pre-annotation)

To speed up labeling, connect the trained YOLO model to Label Studio as a pre-annotation backend:

```bash
source venv/bin/activate
label-studio-ml start ml_backend/ --port 9090
# Connect at http://localhost:9090 via Label Studio > Settings > Machine Learning
```

To use a specific model checkpoint, set `YOLO_MODEL_PATH` (default: `outputs/detection/train/weights/best.pt`).

### 2. Extraction

Crops the labeled bounding box regions from frames into individual images. Useful for testing card matching independently of the detection model.

```bash
python extraction/extract_regions.py
# Output: outputs/extracted_regions/
```

### 3. Detection

Train a YOLOv8 model to detect cards in unlabeled frames.

```bash
python detection/prepare_dataset.py    # split labels into train/val
python detection/train.py              # train YOLO model
python detection/detect.py             # run model on new frames
# Output: outputs/detection/
```

### 4. Matching

Match detected card crops against the Pokemon TCG reference catalog to identify card names.

```bash
# ORB feature matching
python matching/orb/match_cards.py

# Embedding-based matching (experimental: DINOv2 or MobileNetV4)
python matching/embedding/evaluate.py

# Output: outputs/match_results/
```

The first run will download reference card images to `data/card_images/` (rate-limited to ≤10 req/s).
