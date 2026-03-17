# Pokematching

## Setup

Create the project virtualenv using Python 3.11.9 via pyenv (avoids `pkgutil.find_loader` error in Python 3.14):

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Labeling (Label Studio)

```bash
label-studio start
# UI at http://localhost:8080
```

Export annotations in YOLO format and place them in the shared data directories:

- Images → `data/frames/`
- Labels → `data/labels/`

YOLO bounding box format: `class_id x_center y_center width height` (all values normalized 0–1).

Classes:
- `0`: attached-energy
- `1`: attached-item
- `2`: card
- `3`: multicard

### 2. Extraction

Extraction will just extract the ROIs from the labeled images directly. This is useful for testing card matching without relying on the outputs of a detection model.

```bash
python extraction/extract_regions.py
```

### 3. Detection

```bash
python detection/prepare_dataset.py    # split into train/val
python detection/train.py              # train YOLO model
python detection/detect.py             # run on new frames
```

### 4. Matching

```bash
# ORB feature matching
python matching/orb/match_cards.py

# Embedding-based matching (experimental)
python matching/embedding/evaluate.py
```
