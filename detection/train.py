#!/usr/bin/env python3
"""Train a YOLO object detection model on labeled Pokemon TCG card frames.

Usage:
    python detection/train.py [--model yolov8n.pt] [--epochs 100] [--imgsz 640]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("detection/dataset.yaml")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="outputs/detection",
        name="train",
    )


if __name__ == "__main__":
    main()
