//! Offline single-image → low-poly planes — our reproducible "basis".
//!
//! One stock image in; metric depth, the unprojected point cloud, and the
//! detected low-poly planes out (as PLY), plus diagnostics on stdout. No camera,
//! no tracking, no live noise — so we can see exactly what the depth model and
//! plane detector do on a fixed input and iterate.
//!
//!   cargo run --release -p ge-viewer --example image_to_planes --features image-recon -- path/to/image.jpg
//!
//! Writes out/pointcloud.ply (raw depth geometry) and out/planes.ply (low-poly).

use ge_backend_trait::{DepthBackend, DepthMap, Frame, Intrinsics};
use ge_mesh::Mesh;
use glam::Affine3A;

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

/// Downsample a depth map to a working width with matching assumed intrinsics.
fn working_depth(dm: &DepthMap, target_w: u32, fov: f32) -> (DepthMap, Intrinsics) {
    let (dw, dh) = (dm.width as usize, dm.height as usize);
    let tw = (target_w as usize).max(1);
    let th = (dh * tw / dw).max(1);
    let mut depth = vec![0.0f32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            depth[ty * tw + tx] = dm.depth_m[(ty * dh / th) * dw + (tx * dw / tw)];
        }
    }
    (
        DepthMap {
            width: tw as u32,
            height: th as u32,
            depth_m: depth,
            confidence: None,
        },
        assumed_intrinsics(tw as u32, th as u32, fov),
    )
}

fn main() -> anyhow::Result<()> {
    let image_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: image_to_planes <image> [model.onnx]"))?;
    let model = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/dav2_metric_indoor_252.onnx".to_string());

    // Decode the image into an RGB frame.
    let img = image::ImageReader::open(&image_path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame {
        width: w,
        height: h,
        timestamp_ns: 0,
        rgb: img.into_raw(),
    };
    println!("image {image_path}: {w}x{h}");

    // Metric depth, then a working-resolution copy + intrinsics.
    let mut depth = ge_depth::OrtDepth::new_with_size(&model, ge_depth::Accel::Cpu, 252)?;
    let dm_full = depth.infer(&frame)?;
    let (dm, intr) = working_depth(&dm_full, 200, 65.0);
    let (tw, th) = (dm.width as usize, dm.height as usize);

    // Depth diagnostics.
    let valid: Vec<f32> = dm
        .depth_m
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d > 0.05)
        .collect();
    if valid.is_empty() {
        anyhow::bail!("no valid depth");
    }
    let (mut dmin, mut dmax, mut dsum) = (f32::MAX, f32::MIN, 0.0f64);
    for &d in &valid {
        dmin = dmin.min(d);
        dmax = dmax.max(d);
        dsum += d as f64;
    }
    println!(
        "depth: {} valid px, min {:.3} max {:.3} mean {:.3} m  (intr fx {:.1})",
        valid.len(),
        dmin,
        dmax,
        dsum / valid.len() as f64,
        intr.fx
    );

    std::fs::create_dir_all("out")?;

    // Raw point cloud (what the depth model actually produced).
    let mut pc = Mesh::default();
    for v in 0..th {
        for u in 0..tw {
            let d = dm.depth_m[v * tw + u];
            if d.is_finite() && d > 0.05 {
                let p = intr.unproject(u as f32, v as f32, d);
                pc.positions.push([p.x, p.y, p.z]);
            }
        }
    }
    pc.write_ply(std::path::Path::new("out/pointcloud.ply"))?;
    println!("wrote out/pointcloud.ply — {} points", pc.positions.len());

    // Plane detection.
    let det = ge_prim::DetectParams::default();
    let segments = ge_prim::detect_planes(&dm, &intr, &det);
    println!("detected {} planar segment(s):", segments.len());
    for (i, s) in segments.iter().enumerate() {
        println!(
            "  [{i}] normal ({:+.2},{:+.2},{:+.2}) offset {:+.2} cells {} flatness {:.2e}",
            s.plane.normal.x,
            s.plane.normal.y,
            s.plane.normal.z,
            s.plane.offset,
            s.cell_count,
            s.flatness
        );
    }

    // Low-poly mesh (single frame → confirm immediately).
    let mut reg = ge_prim::WorldPlaneRegistry::new(ge_prim::RegistryParams {
        confirm_after: 1,
        ..Default::default()
    });
    reg.observe(&segments, &Affine3A::IDENTITY);
    let mesh = reg.to_mesh();
    mesh.write_ply(std::path::Path::new("out/planes.ply"))?;
    println!(
        "wrote out/planes.ply — {} planes, {} triangles",
        reg.confirmed_count(),
        mesh.triangle_count()
    );
    Ok(())
}
