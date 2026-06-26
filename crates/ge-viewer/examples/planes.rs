//! Live low-poly plane reconstruction.
//!
//! camera → metric depth → CAPE-style plane detection → world plane registry
//! (fused across frames, posed by the 2D-tracking-derived camera rotation) →
//! each confirmed plane rendered as one oriented rectangle (2 triangles). The
//! room becomes a handful of clean planes instead of a noisy dense mesh.
//!
//! Camera rotation only (panorama/near-static regime) — pan/tilt/roll slowly to
//! reveal the room; walking around is a later milestone. Run in release:
//!   cargo run --release -p ge-viewer --example planes --features planes -- 1

use ge_backend_trait::{DepthBackend, DepthMap, Frame, Intrinsics};
use glam::{Affine3A, Quat, Vec2};

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

/// Downsample the full-res depth + frame to a common working resolution,
/// returning (working depth, intrinsics, grayscale for tracking).
fn working(
    frame: &Frame,
    dm: &DepthMap,
    target_w: u32,
    fov: f32,
) -> (DepthMap, Intrinsics, Vec<f32>) {
    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let (dw, dh) = (dm.width as usize, dm.height as usize);
    let tw = (target_w as usize).max(1);
    let th = (fh * tw / fw).max(1);
    let mut depth = vec![0.0f32; tw * th];
    let mut gray = vec![0.0f32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            depth[ty * tw + tx] = dm.depth_m[(ty * dh / th) * dw + (tx * dw / tw)];
            let si = ((ty * fh / th) * fw + (tx * fw / tw)) * 3;
            gray[ty * tw + tx] = (0.299 * frame.rgb[si] as f32
                + 0.587 * frame.rgb[si + 1] as f32
                + 0.114 * frame.rgb[si + 2] as f32)
                / 255.0;
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
        gray,
    )
}

/// Convert a per-frame 2D rigid image transform (prev_from_cur) into a 3D
/// relative camera rotation under a rotation-about-centre (zero translation)
/// assumption: image pan→yaw/pitch, image rotation→roll.
fn rel_rotation(rel: glam::Affine2, center: Vec2, fx: f32, fy: f32) -> Quat {
    let t = rel.transform_point2(center) - center;
    let roll = rel.matrix2.x_axis.y.atan2(rel.matrix2.x_axis.x);
    let yaw = -t.x / fx;
    let pitch = -t.y / fy;
    Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch) * Quat::from_rotation_z(roll)
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
    let mut prev_gray: Option<(Vec<f32>, usize, usize)> = None;
    let mut world_rot = Quat::IDENTITY;
    let mut frame_i = 0usize;

    println!("reconstructing planes — pan/tilt/roll slowly; orbit with the mouse…");
    ge_viewer::view_meshes(
        move || {
            let frame = match camera.next_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let dm_full = depth.infer(&frame)?;
            let (dm, intr, gray) = working(&frame, &dm_full, 160, 65.0);
            let (tw, th) = (dm.width as usize, dm.height as usize);

            // Camera rotation from 2D tracking (rotation-only world pose).
            if let Some((pg, pw, ph)) = prev_gray.as_ref() {
                if *pw == tw && *ph == th {
                    let rel = ge_slam::estimate_rigid2d(pg, &gray, tw, th);
                    let center = Vec2::new(tw as f32 * 0.5, th as f32 * 0.5);
                    let moved = (rel.transform_point2(center) - center).length();
                    if moved < tw as f32 * 0.4 {
                        world_rot =
                            (world_rot * rel_rotation(rel, center, intr.fx, intr.fy)).normalize();
                    }
                }
            }
            prev_gray = Some((gray, tw, th));

            let cam_to_world = Affine3A::from_quat(world_rot);
            let segments = ge_prim::detect_planes(&dm, &intr, &det);
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
