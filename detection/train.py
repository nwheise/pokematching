#!/usr/bin/env python3
"""Train a YOLO object detection model on labeled Pokemon TCG card frames.

Usage:
    python detection/train.py [--model yolo11n.pt] [--epochs 100] [--imgsz 640]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "outputs/detection/dataset/dataset.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs/detection"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(OUTPUT_DIR),
        name="train",
    )


if __name__ == "__main__":
    main()
