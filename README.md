# Gods Eye

**Real-time monocular 3D reconstruction — a single camera in, a live triangle mesh out.**

Gods Eye turns an ordinary RGB camera feed into a live, incrementally-built
triangle mesh of the space around it. No LiDAR, no depth sensor, no IMU — pure
vision, inspired by the geometry-first spirit of Tesla's occupancy stack, but
built to run **locally and open-source on modest hardware**.

The trick is adaptive level-of-detail: a flat wall costs as little as two
triangles, complex regions get more, and detail is added *progressively over
multiple frames* — so compute follows information instead of being spent
uniformly.

> **Status: early.** Scaffolding + design. The architecture is settled
> (see [`docs/design/`](docs/design/)); the pipeline stages are being built one
> milestone at a time. It does not reconstruct anything yet.

## How it works

```
camera ─▶ metric depth ─▶ depth-assisted visual odometry ─▶
         sparse-hash TSDF fusion ─▶ dirty-block mesh extraction ─▶ live viewer
```

- **Depth** — Depth Anything V2 Metric-Small via [`ort`](https://github.com/pykeio/ort) (ONNX Runtime), with CoreML / CUDA / DirectML acceleration optional behind a trait and a CPU floor everywhere.
- **Pose** — depth-assisted RGB-D-style direct visual odometry (the learned depth turns one camera into RGB-D, recovering scale).
- **Fusion** — a sparse voxel-hashed TSDF integrated on the GPU via [`wgpu`](https://github.com/gfx-rs/wgpu) (Metal / Vulkan / DX12 from one codebase).
- **Mesh** — dirty-block marching cubes with a CPU surface-nets fallback, then RANSAC planar simplification + progressive refinement for the adaptive LOD.

See [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) for the full
design, [`ROADMAP.md`](docs/design/ROADMAP.md) for milestones, and
[`RESEARCH.md`](docs/design/RESEARCH.md) / [`PROPOSALS.md`](docs/design/PROPOSALS.md)
for the analysis behind the choices.

## Honest scope

Gods Eye aims for **locally-metric, room-scale-coherent** geometry that is great
for bounded indoor spaces. Without IMU/LiDAR, a monocular system *will* drift
over long free-walking sessions — it is not survey-grade SLAM, and it produces
no object labels (pure geometry, by design). Indoor is the first target;
outdoor/drone is a later, separate milestone.

## Build

Requires a recent Rust toolchain (1.96+). No `cmake` needed for the core.

```sh
cargo build --workspace
cargo run -p ge-cli -- --frames 30   # M0 smoke run (synthetic source)
```

## Platforms

Apple Silicon (Metal), Linux + NVIDIA/AMD/Intel (Vulkan), Windows (DX12).
A pure-Rust/CPU path runs everywhere with no native ML toolchain.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option. Neural-network weights are downloaded at first run and carry their
own licenses — see [NOTICE](NOTICE).
