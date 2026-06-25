//! Live 3D reconstruction with visual odometry.
//!
//! camera → metric depth → frame-to-frame VO (point-to-plane ICP) → a persistent
//! TSDF fused at the tracked pose → surface mesh, shown live. Turn or move and
//! the room accumulates in one fixed world frame; turn back and the earlier
//! geometry is still there (rotated into view). VO is dead-reckoning for now, so
//! it drifts over long runs — best for "recent" backtracking.
//!
//! Run in release (depth + fusion are CPU-heavy), and export the metric model
//! first (`python3 tools/export_metric_onnx.py '' 252`):
//!   cargo run --release -p ge-viewer --example reconstruct --features reconstruct -- 1

use ge_backend_trait::{DepthBackend, DepthMap, Intrinsics};

/// Pinhole intrinsics from an assumed horizontal FOV (the webcam reports none).
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

/// Downsample a depth map to `target_w` (nearest) with matching assumed
/// intrinsics — keeps VO + fusion fast regardless of camera resolution.
fn downsample(src: &DepthMap, target_w: u32, hfov_deg: f32) -> (DepthMap, Intrinsics) {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let tw = (target_w as usize).max(1);
    let th = (sh * tw / sw).max(1);
    let mut depth_m = vec![0.0f32; tw * th];
    for ty in 0..th {
        let sy = ty * sh / th;
        for tx in 0..tw {
            let sx = tx * sw / tw;
            depth_m[ty * tw + tx] = src.depth_m[sy * sw + sx];
        }
    }
    let intr = assumed_intrinsics(tw as u32, th as u32, hfov_deg);
    (
        DepthMap {
            width: tw as u32,
            height: th as u32,
            depth_m,
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

    // Persistent world TSDF: a ~4 x 4 x 4 m volume in front of the start point.
    let voxel = 0.04f32;
    let mut tsdf = ge_fusion::Tsdf::new([100, 100, 98], voxel, [-2.0, -2.0, 0.1], 4.0 * voxel);
    let mut vo: Option<ge_slam::RgbdVoTracker> = None;
    let mut frame_i = 0usize;

    println!("reconstructing — move/turn slowly; orbit the result with the mouse…");
    ge_viewer::view_meshes(
        move || {
            let frame = match camera.next_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let dm_full = depth.infer(&frame)?;
            let (dm, intr) = downsample(&dm_full, 160, 65.0);
            let tracker = vo.get_or_insert_with(|| ge_slam::RgbdVoTracker::new(intr));
            let pose = tracker.track(&dm);
            tsdf.integrate(&dm, &intr, &pose);
            frame_i += 1;
            // Re-mesh on a slower cadence than tracking (extraction is the
            // expensive step); the TSDF keeps accumulating every frame.
            if frame_i % 4 == 0 {
                Ok(Some(tsdf.extract_mesh()))
            } else {
                Ok(None)
            }
        },
        "Gods Eye — live 3D (VO)",
    )
}
