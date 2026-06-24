#!/usr/bin/env python3
"""Export a Depth-Anything-V2 *metric* checkpoint to a fixed-shape ONNX model.

The public `onnx-community` DAv2 export is the *relative* model (no real-world
scale). For correct 3D reconstruction we need metric depth (meters), which has
no official ONNX export — so we trace the transformers checkpoint here.

Output is a fixed `1x3xSIZE xSIZE` input / `1xSIZE xSIZE` metric-depth output,
loadable by ge-depth's OrtDepth via `--model-path`. Same ImageNet preprocessing
as the relative model (DPTImageProcessor mean/std), so no pipeline change.

This is OFFLINE tooling — it never runs on the real-time path.

Usage:
    python3 tools/export_metric_onnx.py [MODEL_ID] [SIZE] [OUT.onnx]
Defaults: Depth-Anything-V2-Metric-Indoor-Small-hf, 392, models/dav2_metric_indoor_392.onnx
"""
import sys

import torch
from transformers import AutoModelForDepthEstimation

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else (
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
)
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 392
OUT = sys.argv[3] if len(sys.argv) > 3 else f"models/dav2_metric_indoor_{SIZE}.onnx"

assert SIZE % 14 == 0, "input size must be divisible by the 14px ViT patch size"


class DepthOnly(torch.nn.Module):
    """Expose just the metric depth map (drops the structured output)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth


def main() -> None:
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    wrapper = DepthOnly(model)
    dummy = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            OUT,
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=17,
            dynamo=False,
        )
    print(f"exported {OUT} (fixed 1x3x{SIZE}x{SIZE}, metric meters)")


if __name__ == "__main__":
    main()
