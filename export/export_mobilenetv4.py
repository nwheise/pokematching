#!/usr/bin/env python3
"""Export MobileNetV4 embedding model to ONNX format with a JSON sidecar.

Performs a parity check between PyTorch and ONNX outputs (asserts max diff < 1e-4).

Outputs:
  <out>/mobilenetv4.onnx
  <out>/mobilenetv4.json

Usage:
    python export/export_mobilenetv4.py
    python export/export_mobilenetv4.py --model_id mobilenetv4_conv_small.e2400_r224_in1k --out outputs/export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL_ID = "mobilenetv4_conv_small.e2400_r224_in1k"


def main():
    parser = argparse.ArgumentParser(description="Export MobileNetV4 to ONNX")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="timm model ID")
    parser.add_argument("--out", default="outputs/export",
                        help="Output directory")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}...")
    model = timm.create_model(args.model_id, pretrained=True, num_classes=0)
    model.eval()

    # Resolve canonical preprocessing params — same path as _make_transform in match_cards.py
    cfg = resolve_data_config(model.pretrained_cfg)
    input_size = cfg["input_size"]  # (3, H, W)
    h, w = input_size[1], input_size[2]
    crop_pct = float(cfg.get("crop_pct", 0.875))
    scale_size = int(round(min(h, w) / crop_pct))
    mean = list(cfg["mean"])
    std = list(cfg["std"])

    print(f"  input_size={h}x{w}, scale_size={scale_size}, crop_pct={crop_pct}")
    print(f"  mean={mean}, std={std}")

    dummy_input = torch.zeros(1, 3, h, w)

    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
    embedding_dim = pt_out.shape[1]
    print(f"  embedding_dim={embedding_dim}")

    onnx_path = out_dir / "mobilenetv4.onnx"
    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Saved {onnx_path}")

    # Parity check: compare PyTorch and ONNX outputs on the same dummy input
    print("Running parity check...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_input.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"  max |PT - ORT| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Parity check failed: max diff {max_diff:.2e} >= 1e-4"
    print("  Parity check passed.")

    sidecar = {
        "model_id": args.model_id,
        "input_size": [h, w],
        "scale_size": scale_size,
        "crop_pct": crop_pct,
        "mean": mean,
        "std": std,
        "embedding_dim": embedding_dim,
    }
    sidecar_path = out_dir / "mobilenetv4.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Saved {sidecar_path}")


if __name__ == "__main__":
    main()
