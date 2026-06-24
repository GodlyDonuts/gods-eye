# Gods Eye — Benchmarks

Measured numbers that ground the design. **All figures are provisional** until
re-run on an idle machine (see caveat).

## M0 depth-latency spike

- **Machine:** Apple M1 (base), 8-core GPU, 16-core ANE, 8 GB unified, macOS 26.5.
- **Model:** `onnx-community/depth-anything-v2-small` (ViT-S / DPT). This is the
  *relative* checkpoint; it shares the exact backbone with the metric variant,
  so its inference latency is a faithful proxy. (A metric-checkpoint ONNX export
  is a separate offline-tooling task tracked for correctness.)
- **Runtime:** `ort` 2.0.0-rc.12 (ONNX Runtime), fp32, synthetic NCHW input.
- **Harness:** `cargo run -p ge-cli --features coreml -- bench-depth ...`
  (see `crates/ge-depth/src/lib.rs`).

> ⚠️ **Caveat — read `min`, not `median`.** These were captured while the Mac
> was in active use and under memory pressure (≈3 GB swap in use at run time).
> On an 8 GB machine, pressure spills to SSD swap and inflates latency, hitting
> the tail (`median`/`p95`) far more than the floor. `min_ms` (the fastest,
> least-contended iteration) is the best estimate of true compute time. Re-run
> idle for clean numbers.

| EP | input | min ms | median ms | fps (from min) |
|----|-------|-------:|----------:|---------------:|
| CPU    | 252² | 95.2  | 101.0 | 10.5 |
| CPU    | 392² | 215.5 | 226.4 | 4.6  |
| CPU    | 518² | 441.6 | 478.2 | 2.3  |
| CoreML | 252² | 96.8  | 100.6 | 10.3 |
| CoreML | 392² | 420.0 | 524.7 | 2.4  |

## Findings

1. **Resolution dominates.** CPU cost scales ~quadratically with input size
   (252→518 ≈ 4.6×). 252² gives ~10 fps; 518² (native) is ~2 fps.
2. **Dynamic input shape makes ORT's CoreML EP unusable** — it recompiles the
   CoreML graph on every call (~12 s/frame). Fixed via
   `tools/export_fixed_onnx.py` (freezes `batch/height/width`).
3. **Even fixed-shape, ORT→CoreML is not faster than CPU** (≈equal at 252²,
   *slower* at 392²). ORT's CoreML EP partitions a DPT/ViT graph and falls back
   to CPU for many ops, paying copy costs at each boundary; fp32 is also not
   ANE-friendly. **Routing the model through ORT→CoreML buys no acceleration on
   this M1.**

## Implications / next

- A first working pipeline is viable today at **252² ≈ 10 fps** on CPU alone.
- Real Apple acceleration needs a different path than ORT's CoreML EP:
  - a native CoreML `.mlpackage` via `coremltools` (fp16, fixed shape), or
  - `candle` with its Metal backend (run the ViT on the M1 GPU directly).
  Benchmark `candle`-Metal first (pure-Rust, no extra runtime).
- Re-benchmark on an idle machine; capture peak RSS alongside latency.
- The project's *own* GPU compute (TSDF fusion + marching cubes via `wgpu`) is a
  separate, later concern from depth-model acceleration.
