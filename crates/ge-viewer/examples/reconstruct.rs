//! Live single-view 3D reconstruction from a camera.
//!
//! camera → metric depth (DAv2 metric ONNX) → per-pixel unprojected surface
//! mesh, shown live in a 3D window. Single viewpoint with identity pose (no
//! cross-frame fusion yet — that needs visual odometry), so it's a live, metric
//! "depth relief" of whatever the camera sees. Orbit with the mouse.
//!
//! Run in release for smooth depth, and export the metric model first:
//!   python3 tools/export_metric_onnx.py            # writes models/dav2_metric_indoor_392.onnx
//!   python3 tools/export_metric_onnx.py '' 252     # faster 252 variant
//!   cargo run --release -p ge-viewer --example reconstruct --features reconstruct
//!   cargo run --release -p ge-viewer --example reconstruct --features reconstruct -- 1   # camera index 1 (e.g. iPhone)

use ge_backend_trait::{DepthBackend, DepthMap, Intrinsics};
use ge_mesh::Mesh;

/// Pinhole intrinsics from an assumed horizontal field of view (square pixels).
/// The webcam doesn't report intrinsics, so x/y scale is approximate; depth (z)
/// is metric from the model.
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

/// Unproject a subsampled metric depth map into a triangle surface mesh. Quads
/// spanning a large depth discontinuity are dropped to avoid stretched
/// triangles across occlusion edges.
fn depth_to_mesh(depth: &DepthMap, intr: &Intrinsics, stride: usize, max_depth: f32) -> Mesh {
    let w = depth.width as usize;
    let h = depth.height as usize;
    let cols = (w / stride).max(1);
    let rows = (h / stride).max(1);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut idx = vec![u32::MAX; cols * rows];
    let mut zs = vec![0.0f32; cols * rows];
    for gv in 0..rows {
        for gu in 0..cols {
            let (u, v) = (gu * stride, gv * stride);
            let d = depth.depth_m[v * w + u];
            if d.is_finite() && d > 0.1 && d < max_depth {
                let p = intr.unproject(u as f32, v as f32, d);
                idx[gv * cols + gu] = positions.len() as u32;
                zs[gv * cols + gu] = d;
                positions.push([p.x, p.y, p.z]);
            }
        }
    }

    let mut indices: Vec<u32> = Vec::new();
    for gv in 0..rows.saturating_sub(1) {
        for gu in 0..cols.saturating_sub(1) {
            let a = idx[gv * cols + gu];
            let b = idx[gv * cols + gu + 1];
            let c = idx[(gv + 1) * cols + gu];
            let e = idx[(gv + 1) * cols + gu + 1];
            if a == u32::MAX || b == u32::MAX || c == u32::MAX || e == u32::MAX {
                continue;
            }
            let za = zs[gv * cols + gu];
            let zb = zs[gv * cols + gu + 1];
            let zc = zs[(gv + 1) * cols + gu];
            let ze = zs[(gv + 1) * cols + gu + 1];
            let (zmax, zmin) = (za.max(zb).max(zc).max(ze), za.min(zb).min(zc).min(ze));
            if zmax - zmin > 0.05 * zmax {
                continue; // occlusion edge
            }
            indices.extend_from_slice(&[a, c, b, b, c, e]);
        }
    }

    Mesh {
        positions,
        normals: Vec::new(),
        indices,
    }
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
    // Accept a numeric index (e.g. `1` for the iPhone) or a name substring.
    let mut camera = match cam.as_deref() {
        None => ge_viewer::WebcamSource::open(0)?,
        Some(s) => match s.parse::<u32>() {
            Ok(index) => ge_viewer::WebcamSource::open(index)?,
            Err(_) => ge_viewer::WebcamSource::open_named(s)?,
        },
    };

    println!("loading metric depth model: {model}");
    let mut depth = ge_depth::OrtDepth::new_with_size(&model, ge_depth::Accel::Cpu, 252)?;
    println!("reconstructing — orbit with the mouse…");

    ge_viewer::view_meshes(
        move || match camera.next_frame()? {
            Some(frame) => {
                let dm = depth.infer(&frame)?;
                let intr = assumed_intrinsics(frame.width, frame.height, 65.0);
                let stride = (frame.width as usize / 256).max(1);
                Ok(Some(depth_to_mesh(&dm, &intr, stride, 10.0)))
            }
            None => Ok(None),
        },
        "Gods Eye — live 3D",
    )
}
