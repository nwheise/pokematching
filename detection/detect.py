#!/usr/bin/env python3
"""Run a trained YOLO detector on video frames to produce bounding box predictions.

Usage:
    python detection/detect.py [--model outputs/detection/train/weights/best.pt] [--source data/frames/]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on frames")
    parser.add_argument("--model", default="outputs/detection/train/weights/best.pt", help="Trained model weights")
    parser.add_argument("--source", default="data/frames/", help="Directory of frames to run on")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        save=True,
        save_txt=True,
        project="outputs/detection",
        name="predictions",
    )


if __name__ == "__main__":
    main()
