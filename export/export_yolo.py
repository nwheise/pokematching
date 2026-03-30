#!/usr/bin/env python3
"""Export trained YOLOv11n to ONNX format with a JSON sidecar.

Outputs:
  <out>/yolo.onnx
  <out>/yolo.json

Usage:
    python export/export_yolo.py
    python export/export_yolo.py --weights outputs/detection/train/weights/best.pt --out outputs/export
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import CLASSES


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--weights", default="outputs/detection/train/weights/best.pt",
                        help="Path to trained YOLO weights (.pt)")
    parser.add_argument("--out", default="outputs/export",
                        help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size for export")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"Error: weights not found at {weights_path}", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(str(weights_path))

    # Attempt export with NMS baked in; fall back to nms=False if it fails
    nms_baked = True
    try:
        print("Exporting to ONNX (nms=True)...")
        onnx_path = Path(model.export(format="onnx", nms=True, imgsz=args.imgsz))
    except Exception as e:
        print(f"  nms=True failed ({e}), retrying with nms=False...")
        nms_baked = False
        onnx_path = Path(model.export(format="onnx", nms=False, imgsz=args.imgsz))

    dest = out_dir / "yolo.onnx"
    shutil.copy(onnx_path, dest)
    print(f"Saved {dest}")

    sidecar = {
        "input_size": [args.imgsz, args.imgsz],
        "class_names": [CLASSES[i] for i in sorted(CLASSES)],
        "nms_baked": nms_baked,
        "model_id": "yolo11n",
    }
    sidecar_path = out_dir / "yolo.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Saved {sidecar_path} (nms_baked={nms_baked})")


if __name__ == "__main__":
    main()
