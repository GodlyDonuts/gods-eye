# Gods Eye — Roadmap, Risks & Open Questions

## The low-poly plan (2026-07-01) — current milestone ladder

**Pivot:** the world is represented as clean primitives — planes first — with
dense TSDF meshing demoted to *residual* geometry that planes cannot explain
(`ge-prim` is the pivot's foundation). The M-series below predates this pivot;
its stages survive as components (M0 spine = shipped, M1 VO = L2, M2 drift =
L3, TSDF+mesh = L5) and its risk register still applies, but this ladder
supersedes its ordering.

**Progress:** L1 + L2 shipped (2026-07-01). L1: planes render as true polygons
with crisp plane–plane edges. L2: full 6-DoF depth-assisted VO wired live, with
an offline drift harness (0.2% clean, 1.6% under depth breathing after the
joint pose+scale mitigation). Next up: L3 (planes as landmarks / drift-free
return) and L4 (dynamic-object detection, red).

**Standing today** (post-L2): live webcam → DAv2 metric depth (CPU, 252px) →
**full 6-DoF depth-assisted VO** (frame-to-keyframe point-to-plane ICP + joint
per-frame depth-scale alignment) → CAPE-style plane detection → moment-fused
world plane registry → each confirmed plane as its **true outline polygon** with
crisp plane–plane edges in the live viewer. The tracker is validated offline
against ground truth (`ge_slam::sim`); real handheld capture is the next test.

### L1 — The room looks architected (stays in the rotation-only regime) — SHIPPED
Planes render as true polygons, not bounding rectangles, and adjacent planes
meet in crisp edges.
- [x] Per-plane 2D footprint rasterized into a plane-local occupancy grid (fed
  by cell centroids), morphological-close to bridge sparse gaps, hole-fill,
  largest-component boundary trace → Douglas–Peucker → ear-clip triangulation
  (`ge-prim/src/polygon.rs::footprint_polygon`/`triangulate`). Concave outlines
  (L-shaped floors) preserved; oriented-rect fallback for sparse footprints.
- [x] Plane–plane intersection snapping: each polygon is Sutherland–Hodgman
  clipped to a neighbour's intersection line, **sliver-guarded** (never removes
  >35% of area) so a real partition is left intact — walls meet floors on a
  shared 3D edge (`polygon.rs::snap_to_line`, wired in `registry.rs::to_mesh`).
- [ ] Footprint hysteresis so polygons grow smoothly instead of popping (defer:
  do together with L2's per-observation temporal smoothing).
- [ ] Optional Manhattan regularization (snap near-orthogonal normals) — not yet
  needed; revisit if walls look skewed on real captures.

**On screen:** pan around a room → floor, walls, ceiling as crisp polygons
with sharp corners — an "architected" room from one webcam. Validated offline:
`image_to_planes` on a real photo yields clean multi-vertex polygons (7 planes →
21 tris) with correct per-plane normals and no degenerate geometry.

### L2 — Walk around (full 6-DoF pose) — SHIPPED
Wire `RgbdVoTracker` (frame-to-keyframe point-to-plane ICP) into the live
pipeline, replacing rotation-only tracking.
- [x] FIRST (M1 discipline): deterministic offline drift harness — a known room
  + known closed-loop trajectory, tracker run against it, end-to-end drift
  reported as a number, in CI (`ge-slam/src/sim.rs`). No COLMAP needed; a
  record/replay real-capture source is a later add for real-data validation.
- [x] Frame-to-keyframe tracking (was frame-to-frame): **6× lower drift on clean
  depth** (1.3% → 0.2% of path length over a 1.9 m loop), 4× on noisy.
- [x] Per-frame depth **scale** alignment, co-estimated jointly with the pose in
  one 7-DoF solve (risk #1 mitigation): **cuts drift under 5% depth breathing
  3.1×** (5.0% → 1.6%), and is harmless on clean depth (scale stays ≈1). Shift
  was dropped deliberately — measured ~98% collinear with scale at room scale,
  so a joint scale+shift solve is ill-posed.
- [x] Scale-observability handled by the Tikhonov prior in the joint solve: when
  scale is degenerate with pose-Z (single frontal wall) the prior keeps scale at
  1 and lets pose absorb the motion, instead of the two fighting. Keyframe
  promotion by motion magnitude + inlier-support drop.
- [ ] Per-keyframe affine applied to the *fused* geometry (vs. just the pose) —
  lands with L5 residual fusion, where depth consistency matters for the mesh.
- [ ] Record/replay real-capture source + COLMAP ground-truth on real footage.

**On screen:** walk through the room; planes stay put; new walls appear as you
enter them. Drift is measured, not vibes: an end-to-end integration test
(`ge-slam/tests/pipeline.rs`) walks two laps and confirms the floor and front
wall fuse to their true world positions with the camera home to <15 cm.

### L3 — It stays put (planes are the landmarks)
Use the persistent plane registry as the map that corrects the pose —
plane-SLAM-lite, no sparse-feature machinery.
- After VO, refine the pose against confirmed world planes (point-to-plane
  against the registry = a drift brake on every wall, textured or not).
- Small keyframe pose graph (argmin/nalgebra) with plane-observation edges;
  loop closure via plane-configuration signatures (normal triads + offsets).
- Registry re-fuse (moments re-lifted) when the graph updates.

**On screen:** walk a loop around the apartment; on return the walls line up
instead of double-walling.

### L4 — What moves is painted red (and kept out of the map)
Moving objects — people, pets, roombas — detected **geometrically, by motion,
not by classification** (stays pure-geometry: no labels, no extra neural net),
rendered as red blobs in the live view, and masked out of tracking and fusion.
- Ego-motion-compensated model residual: render the static model's predicted
  depth (plane registry; later + residual TSDF) from the current pose; live
  pixels that disagree beyond a depth-adaptive threshold are *unexplained*.
- Temporal rigidity check: warp the previous keyframe's depth through the
  relative pose; unexplained pixels that also violate the rigid-static
  hypothesis frame-over-frame are *moving*. Per-cell dynamicity score with
  hysteresis separates "moving" from "static but not yet mapped".
- Cluster moving pixels (image + depth connectivity) → unproject → render as
  red blobs (viewer gains per-vertex color if it lacks it); nearest-centroid
  blob tracking for frame-to-frame stability.
- Feed the mask back: dynamic pixels are excluded from ICP tracking, plane
  detection, and registry observation. This is the correctness half — and why
  dynamics lands BEFORE residual fusion (L5): fusing without the mask bakes
  walking people into the mesh as ghosts.
- Known limit, documented not hidden: an object that stops moving decays to
  static after hysteresis and gets mapped (correct — a parked roomba IS
  furniture); when it moves again its stale geometry must be retired
  (registry demotion / fusion cleanup).

**On screen:** someone walks through the scene as a red blob; the room behind
them never bends, and no ghost trail is left in the map.

### L5 — Everything planes can't explain (residual geometry)
The adaptive-LOD promise: planes cost 2 triangles; everything else gets a real
mesh under a hard budget.
- Residual mask = depth pixels not claimed by confirmed planes → existing
  `ge-fusion` TSDF (dense CPU grid first, sparse-hash later) → `ge-mesh`
  surface-nets → QEM decimation to a hard triangle budget.
- Objects anchor to their supporting plane (sit ON the floor, not near it).

**On screen:** chairs, desks, clutter as compact low-poly meshes standing on
crisp planes.

### L6 — Beautiful and usable (appearance + robotics API)
- Plane texturing from the best keyframe per plane (least-oblique, closest)
  into a simple atlas; stylized flat-shaded palette as the zero-cost fallback.
- glTF/GLB export of the whole scene (plane polygons + residual meshes).
- `ge-py` query API: confirmed planes (normal, offset, polygon), floor
  extraction, a simple occupancy grid for navigation.

**Deliverable:** a textured low-poly room you can open in Blender or hand to a
motion planner.

### Cross-cutting (cheap, any time, high leverage)
- CoreML EP spike (open questions 1–2): depth at 392px and/or 2×+ fps headroom.
- Threaded `ge-core` pipeline (capture/depth/track/fuse stages) once L2 lands —
  depth running below camera rate must not stall tracking.
- Keep `image_to_planes` (single image → planes, offline) as the regression
  fixture for detection quality.

---

## M0 — Smallest end-to-end loop that puts a LIVE triangle mesh on screen AND builds via `cargo build` on >1 platform (macOS arm64 + Linux x86_64), while empirically measuring base-M1 depth latency so no fps number is committed blind. Deliberately fixed-resolution, identity-or-trivial pose, NO adaptivity, NO loop closure — prove the spine and benchmark depth.

**Effort:** 4-6 weeks for one experienced Rust+GPU engineer. The WGSL marching-cubes + workgroup-scan port and the cross-backend (Metal/Vulkan) validation are the time sinks; the depth spike is days but gates everything; fast-surface-nets-first de-risks the mesh path so M0 can land even if the GPU kernel slips.

**Steps:**
1. Scaffold the cargo workspace (root Cargo.toml [workspace] + [workspace.dependencies] pinning EXACT wgpu and ort versions) with the 10 member crates as stubs; dual MIT/Apache LICENSE files + NOTICE; cargo-deny.toml gating GPL/CC-BY-NC.
2. ge-camera: implement a FILE/VIDEO CaptureSource as the primary dev input (decouples the spine from camera flakiness) plus a nokhwa webcam source with a frame format/byte-size assertion at the boundary.
3. ge-depth: OrtBackend running DAv2-Metric-Small ONNX. WEEK-1 SPIKE: benchmark CoreML EP and CPU/XNNPACK at 256/392/518px on the actual M1 box, log raw wall-clock + wgpu-timestamp latency; verify the CoreML build does NOT trigger a from-source cmake build (fall back to tract if it does). First-run hf-hub downloader with sha256 + license prompt + --model-path.
4. ge-fusion: minimal sparse-hash TSDF wgpu integrate kernel; unproject depth->points using fixed/known intrinsics; integrate with a trivial pose (identity or a scripted path) — no VO yet.
5. ge-mesh: fixed-resolution dirty-block extraction. Ship the fast-surface-nets CPU mesher FIRST as the guaranteed-portable path, then the GPU MC + workgroup-scan kernel; double-buffer vertex/index to the viewer. Validate the workgroup-scan on Metal AND Vulkan (NOT GL).
6. ge-viewer: rerun out-of-process (gRPC) showing the live updating triangle mesh; set memory_limit aggressively low.
7. ge-core: wire the 6 stages with bounded crossbeam channels + object pools; ge-cli binary runs the loop on a recorded video file.
8. CI: GitHub Actions matrix ubuntu(+libv4l-dev) + macos-14(arm64), default job pure cargo (build + clippy + cargo-deny), no cmake; assert the binary builds and runs headless on a test video on both.

**Files to create:**

| Path | Purpose |
|---|---|
| `Cargo.toml` | workspace root: [workspace] members + [workspace.dependencies] pinning exact wgpu/ort/nokhwa/nalgebra/glam/meshopt/fast-surface-nets/crossbeam/rerun versions |
| `cargo-deny.toml` | license gate denying GPL/AGPL/CC-BY-NC, allowing MIT/Apache/BSD |
| `LICENSE-MIT` | project MIT license |
| `LICENSE-APACHE` | project Apache-2.0 license |
| `NOTICE` | LICENSE-THIRD-PARTY: per-model weight licenses + the DAv2-Small data-provenance caveat (issue #320) |
| `.github/workflows/ci.yml` | GitHub Actions matrix ubuntu+macos-14(arm64), pure-cargo default job (build/clippy/test/cargo-deny), opt-in accel-EP jobs |
| `crates/ge-backend-trait/src/lib.rs` | DepthBackend/PoseEstimator/ComputeBackend/CaptureSource trait definitions |
| `crates/ge-camera/src/lib.rs` | file/video CaptureSource (primary dev input) + nokhwa webcam source with format assertion |
| `crates/ge-depth/src/lib.rs` | OrtBackend DAv2-Metric-Small + latency-spike benchmark harness + EP/feature gating |
| `crates/ge-fusion/src/lib.rs` | minimal sparse-hash TSDF wgpu integrate + unproject (trivial pose) |
| `crates/ge-fusion/src/shaders/integrate.wgsl` | WGSL TSDF integration compute kernel |
| `crates/ge-mesh/src/lib.rs` | fast-surface-nets CPU mesher (portable floor) + GPU MC entry point, double-buffered output |
| `crates/ge-mesh/src/shaders/marching_cubes.wgsl` | WGSL 3-kernel MC (active-detect/workgroup-scan/vertex-emit), non-subgroup fallback |
| `crates/ge-viewer/src/lib.rs` | rerun out-of-process live mesh logging with memory_limit |
| `crates/ge-core/src/lib.rs` | 6-stage crossbeam pipeline + object pools + return channels |
| `crates/ge-cli/src/main.rs` | clap binary: first-run downloader, --model-path, runs the loop on a video file |

**Acceptance criteria:** `cargo build` succeeds with NO cmake on BOTH macOS arm64 and Linux x86_64 (verified in CI); running the ge-cli binary on a recorded indoor video produces a LIVE, incrementally-updating fixed-resolution triangle mesh in the rerun viewer; the M0 spike has logged measured DAv2-Small depth latency at 256/392/518px on the actual base-M1 box (CoreML + CPU) and that measured number — not an interpolation — is recorded as the basis for the latency budget; the CoreML build is confirmed to NOT pull cmake (or tract fallback is wired); cargo-deny passes with zero GPL/CC-BY-NC; peak resident memory under macOS measured and under the ~4GB free-RAM ceiling with rerun out-of-process.

## Next milestones
- M1 (research-grade, front-loaded): hand-rolled depth-assisted direct VO as a DEDICATED milestone with an OFFLINE accuracy harness comparing against COLMAP ground-truth on recorded sequences BEFORE wiring live. Add the per-keyframe affine (scale+shift) depth-alignment stage and a scale-observability gate (freeze scale + flag low-confidence on rotation-dominant/low-texture motion). Re-scope to frame-to-keyframe tracking only; budget 2-4x the naive estimate.
- M2 (drift/backend): keyframe pose-graph (argmin/nalgebra, optional cxx-GTSAM) + loop closure promoted to first-class (Rust HNSW + learned global descriptor). Topology-change override so loop-closure re-meshing is allowed despite never-coarsen. Multi-minute drift metric on a loop sequence. Honest deliverable: locally-metric, room-scale-coherent, globally-drifting.
- M3 (adaptivity, crack-free): ship QEM SimplifyLockBorder-ONLY adaptivity first (mature, crack-free same-res) + ErrorAbsolute world-unit thresholds + observation-driven monotonic shrinking target_error with hysteresis. THEN gate RANSAC 2-tri planar quads behind a validated plane-quad/voxel-patch seam-stitch pass + plane-membership hysteresis. Variance-adaptive (MrHash) LOD as a feature-flagged research spike only.
- M4 (portability/packaging hardening): full CI matrix incl. Windows/DirectML + opt-in CUDA/TensorRT jobs; out-of-core block streaming validated on a discrete-VRAM box; cargo-deny + first-run downloader + NOTICE finalized; three-d shipped viewer.
- M5 (drone, separate + unproven): DJI Mini H.264/RTMP decode (retina/openh264) behind CaptureSource; online intrinsics/auto-exposure handling, rolling-shutter + motion-blur robustness; Outdoor checkpoint validation on aerial footage. Do NOT assume the indoor pipeline transfers for free.

## Biggest risks (ranked)
1. [HIGH] Monocular metric-scale + pose/scale drift without IMU/LiDAR: per-frame learned depth is affine-inconsistent (a wall breathes 5-20% with exposure/focus/angle), so fused into a TSDF it produces warp/thickening/ghost-doubling that pose-graph scalar scale-consistency cannot fix (the error is non-rigid). Degenerate motion (slow pan, hover, textureless walls — exactly the 2-tri-wall showcase) starves parallax and the photometric VO simultaneously. The 'metric model removes scale ambiguity / no drift' framing is FALSE. This is the project's central correctness risk and the single hardest, least-crate-supported stage. MITIGATION: per-keyframe affine depth alignment before fusion, depth-confidence-weighted integration, scale-observability gate, first-class early loop closure, scope to small bounded spaces, offline ground-truth harness before live.
2. [HIGH] Depth latency is interpolated, not measured: base-M1 DAv2-Small is realistically 60-110ms (verified ~13ms RTX4080 vs ~98ms datacenter), so real throughput is ~9-15fps not 18-25. The whole budget rests on an admittedly-interpolated number. MITIGATION: M0 hard empirical spike at 256/392/518px (CoreML + CPU) before any number is committed; stop quoting 18-25fps; design for depth below camera rate with multi-frame fusion.
3. [HIGH] Greenfield concentration: ~60-70% of the differentiating code (direct VO, keyframing, plane RANSAC/region-grow, hashed-TSDF fusion, WGSL prefix-scan, progressive mesh, BoW loop closure) has NO mature Rust crate, and the two hardest problems (scale/drift and VO) are the SAME unbuilt stage. Summed schedule is optimistic by 2-4x for one maintainer. MITIGATION: sequence ruthlessly (M0 spine+benchmark, M1 VO research milestone, adaptivity only after the spine is correct); reuse permissive algorithm blueprints (Open3D, RTAB-Map, DPVO); keep a cxx-GTSAM FFI escape hatch.
4. [MEDIUM] Plane-quad / voxel-mesh boundary is a self-contradictory seam: RANSAC 2-tri quads bypass the voxel grid so they cannot share a locked border with the neighboring voxel-meshed chunk — guaranteed cracks at the 'wall meets detail' showcase boundary; Transvoxel does NOT solve that boundary and is itself unbuilt in WGSL. Plus RANSAC plane membership oscillates frame-to-frame on noisy depth (quad popping). MITIGATION: defer plane-quad replacement until a real boundary-stitch exists; ship QEM-LockBorder-only first; add plane-MEMBERSHIP hysteresis + minimum-support gating; geomorph or document popping.
5. [MEDIUM] wgpu compute portability cliff: WGSL prefix-scan/atomics/subgroups are full only on Vulkan, conditional on Metal/DX12, absent on GL — 'one WGSL source everywhere' is FALSE for the compute path. MITIGATION: drop GL from the compute matrix; fast-surface-nets CPU mesher is the documented fallback there; require a non-subgroup workgroup-scan; keep zero correctness on any compute shader; CI-test the CPU path on every platform.
6. [MEDIUM] DAv2-Small weight provenance: Apache-2.0 on paper but unresolved training-data license (issue #320) means commercial-OSS shipping is NOT cleared. MITIGATION: reclassify as 'permissive label, unverified provenance'; NOTICE caveat; track #320; identify a clean-provenance swap-in; --model-path for cleared weights.
7. [MEDIUM] ort CoreML-without-cmake unverified (MS ships no prebuilt CoreML): may trigger a from-source cmake build, breaking the no-cmake guarantee. MITIGATION: week-1 spike on the M1; fall back to tract/candle if it pulls cmake; CPU/XNNPACK is the guaranteed floor.
8. [MEDIUM] Memory undercounts: ort+CoreML co-resident graphs (budget 500-700MB not 260) + growing viewer buffers + degeneration to non-planar path can push real peak to 2.0-2.8GB. The in-process rerun dev path needs 16GB. MITIGATION: hard non-planar cell budget + LRU eviction; share ONE wgpu device; rerun out-of-process; document dev needs 16GB, shipped three-d fits 8GB.
9. [LOW] nokhwa AVFoundation/NV12 bugs on M1 (the baseline box) masquerade as pipeline bugs. MITIGATION: file/video source as primary dev input; pin known-good nokhwa; format/byte-size assertion at the capture boundary.
10. [LOW] Drone transfer (M5): rolling shutter, autofocus-varying intrinsics, motion blur break the fixed-focal scale assumption and stress direct VO; depth models are indoor-biased. MITIGATION: treat as a separate unproven milestone with its own validation; do not let drone claims imply the indoor pipeline transfers for free.

## Open questions
1. What is the ACTUAL measured DAv2-Small depth latency on your specific base-M1 box at 256/392/518px under CoreML and CPU/XNNPACK? Every fps number is gated on this M0 spike — can we run it week 1 before locking the budget?
2. Does enabling ort's CoreML EP trigger a from-source cmake build on your M1, or is there a usable prebuilt? (cmake is confirmed absent; clang is present.) If it pulls cmake, do you accept tract/candle CPU as the macOS floor until resolved?
3. What is the primary deployment scene — indoor rooms (Hypersim/20m checkpoint) or outdoor/drone (VKITTI/80m)? They are NOT interchangeable and this picks the shipped checkpoint and the scale-correction assumptions.
4. Is COMMERCIAL use a goal? If so, the DAv2-Small weight data-provenance question (issue #320) needs upstream/legal sign-off, and we should identify a clean-provenance swap-in now. If non-commercial framing is acceptable, this de-risks the whole license axis.
5. Are the camera intrinsics known/calibrated for the target camera? Known intrinsics materially improve the focal-bias scale correction; unknown-intrinsics (drone) is significantly harder and may need online intrinsics estimation.
6. What session length and accuracy bar defines success — a 30-second bounded-room demo, or multi-minute free-walk? This sets whether loop closure must be in M2 and whether 'locally-metric, globally-drifting' is acceptable.
7. Is the 8GB base M1 the hard target for the LIVE shipped experience, or only for the portable floor? The rerun DEV loop realistically needs 16GB; is a 16GB dev box acceptable while the shipped three-d path targets 8GB?
8. Single-maintainer or team? The schedule (M0 ~4-6wk, then research-grade VO/SLAM as the dominant multi-month risk) assumes one experienced Rust+GPU engineer; team size changes sequencing and whether candle metric-head porting is worth funding.

### Resolved (2026-06-24)
- **Scene:** start **indoor** (Hypersim/20m), but checkpoint is a swappable config so outdoor/drone (VKITTI/80m) slots in later.
- **Licensing:** **open-source / permissive** (dual MIT-OR-Apache-2.0); document DAv2-Small data-provenance caveat (issue #320) in NOTICE; keep `--model-path` for a cleared-weight swap.
- **Language/stack:** Rust + wgpu + ort, decided.
