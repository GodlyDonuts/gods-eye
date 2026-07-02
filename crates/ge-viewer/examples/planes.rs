//! Live low-poly plane reconstruction.
//!
//! camera → metric depth → 6-DoF depth-assisted VO (frame-to-keyframe
//! point-to-plane ICP + per-frame depth-scale alignment) → CAPE-style plane
//! detection → world plane registry (fused across frames, posed by the tracked
//! camera) → each confirmed plane rendered as its true outline polygon. The room
//! becomes a handful of clean planes instead of a noisy dense mesh.
//!
//! Full 6-DoF: walk through the room — planes stay put and new ones appear as you
//! enter them. The tracker is validated offline against ground truth (see
//! `ge_slam::sim`); live handheld capture is the real test. Run in release:
//!   cargo run --release -p ge-viewer --example planes --features planes -- 1

use ge_backend_trait::{DepthBackend, DepthMap, Frame, Intrinsics};

fn assumed_intrinsics(width: u32, height: u32, hfov_deg: f32) -> Intrinsics {
    let fx = (width as f32 * 0.5) / (hfov_deg.to_radians() * 0.5).tan();
    Intrinsics {
        fx,
        fy: fx,
        cx: width as f32 * 0.5,
        cy: height as f32 * 0.5,
        width,
        height,
    }
}

/// Downsample the full-res depth to the working resolution used for tracking +
/// detection, returning (working depth, intrinsics).
fn working(frame: &Frame, dm: &DepthMap, target_w: u32, fov: f32) -> (DepthMap, Intrinsics) {
    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let (dw, dh) = (dm.width as usize, dm.height as usize);
    let tw = (target_w as usize).max(1);
    let th = (fh * tw / fw).max(1);
    let mut depth = vec![0.0f32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            depth[ty * tw + tx] = dm.depth_m[(ty * dh / th) * dw + (tx * dw / tw)];
        }
    }
    let intr = assumed_intrinsics(tw as u32, th as u32, fov);
    (
        DepthMap {
            width: tw as u32,
            height: th as u32,
            depth_m: depth,
            confidence: None,
        },
        intr,
    )
}

fn main() -> anyhow::Result<()> {
    let cam = std::env::args().nth(1);
    let model = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/dav2_metric_indoor_252.onnx".to_string());

    if let Ok(devices) = ge_viewer::list_cameras() {
        println!("available cameras:");
        for d in &devices {
            println!("  [{}] {}", d.index, d.name);
        }
    }
    let mut camera = match cam.as_deref() {
        None => ge_viewer::WebcamSource::open(0)?,
        Some(s) => match s.parse::<u32>() {
            Ok(index) => ge_viewer::WebcamSource::open(index)?,
            Err(_) => ge_viewer::WebcamSource::open_named(s)?,
        },
    };

    println!("loading metric depth model: {model}");
    let mut depth = ge_depth::OrtDepth::new_with_size(&model, ge_depth::Accel::Cpu, 252)?;

    let det = ge_prim::DetectParams::default();
    let mut registry = ge_prim::WorldPlaneRegistry::new(ge_prim::RegistryParams::default());
    // 6-DoF tracker, created on the first frame once the working intrinsics are
    // known (they depend on the camera's frame size).
    let mut tracker: Option<ge_slam::RgbdVoTracker> = None;
    let mut frame_i = 0usize;

    println!("reconstructing planes — move through the room slowly; orbit with the mouse…");
    ge_viewer::view_meshes(
        move || {
            let frame = match camera.next_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let dm_full = depth.infer(&frame)?;
            let (dm, intr) = working(&frame, &dm_full, 160, 65.0);

            // Full 6-DoF depth-assisted VO → camera-to-world pose.
            let tracker = tracker.get_or_insert_with(|| ge_slam::RgbdVoTracker::new(intr));
            let predicted = tracker.track(&dm);

            let segments = ge_prim::detect_planes(&dm, &intr, &det);

            // Frame-to-map plane registration: re-anchor the pose to the
            // persistent map (drift brake / loop closure), feeding the correction
            // back to the tracker. A near-no-op while VO agrees with the map.
            let cam_to_world = match registry.refine_pose(&predicted, &segments) {
                Some(corrected) => {
                    tracker.set_pose(corrected);
                    corrected
                }
                None => predicted,
            };
            registry.observe(&segments, &cam_to_world);

            frame_i += 1;
            if frame_i % 3 == 0 {
                Ok(Some(registry.to_mesh()))
            } else {
                Ok(None)
            }
        },
        "Gods Eye — low-poly planes",
    )
}
