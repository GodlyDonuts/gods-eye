#!/usr/bin/env python3
"""Freeze a Depth-Anything-V2 ONNX export to a fixed input shape.

The public `onnx-community/depth-anything-v2-small` export has dynamic
`batch_size`/`height`/`width`. ONNX Runtime's CoreML execution provider
recompiles the CoreML graph on every call when the shape is dynamic, which makes
it ~50x slower than CPU. Fixing the shape lets CoreML compile once at session
creation and run on the ANE/GPU.

This is OFFLINE tooling — it never runs on the real-time path.

Usage:
    python3 tools/export_fixed_onnx.py IN.onnx OUT.onnx SIZE
"""
import sys

import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed


def main() -> None:
    src, dst, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    model = onnx.load(src)
    make_dim_param_fixed(model.graph, "batch_size", 1)
    make_dim_param_fixed(model.graph, "height", size)
    make_dim_param_fixed(model.graph, "width", size)
    fix_output_shapes(model)
    onnx.save(model, dst)
    print(f"saved {dst} with fixed input 1x3x{size}x{size}")


if __name__ == "__main__":
    main()
