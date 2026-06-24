//! Live mesh viewer.
//!
//! Two implementations are planned behind the [`Viewer`] trait: `rerun`
//! out-of-process for development (rich, but needs a 16 GB dev box), and
//! `three-d` + `egui` in-process for the shipped experience (fits 8 GB).
//!
//! Today this crate ships the headless [`Viewer`] sinks (always available), a
//! PLY loader, and [`view_mesh`] — an interactive `three-d` window behind the
//! `window` feature.

use std::path::Path;

use ge_mesh::Mesh;

/// A sink for live mesh updates.
pub trait Viewer {
    /// Log/replace the current mesh. Called once per extraction tick.
    fn log_mesh(&mut self, mesh: &Mesh) -> anyhow::Result<()>;
}

/// Discards everything. For headless runs where no view is needed.
pub struct NullViewer;

impl Viewer for NullViewer {
    fn log_mesh(&mut self, _mesh: &Mesh) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Records how many times it was called and the last triangle count — used by
/// headless CI to assert the spine actually produced mesh updates.
#[derive(Default)]
pub struct CountingViewer {
    pub updates: usize,
    pub last_triangles: usize,
}

impl Viewer for CountingViewer {
    fn log_mesh(&mut self, mesh: &Mesh) -> anyhow::Result<()> {
        self.updates += 1;
        self.last_triangles = mesh.triangle_count();
        Ok(())
    }
}

/// Load a mesh from an ASCII PLY file produced by [`Mesh::write_ply`].
///
/// Supports `float x/y/z` (+ optional `nx/ny/nz`) vertices and triangle faces.
pub fn load_ply(path: &Path) -> anyhow::Result<Mesh> {
    use std::io::{BufRead, BufReader};

    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut line = String::new();
    let (mut n_verts, mut n_faces, mut has_normals) = (0usize, 0usize, false);

    loop {
        line.clear();
        anyhow::ensure!(
            reader.read_line(&mut line)? > 0,
            "unexpected EOF in PLY header"
        );
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("element vertex ") {
            n_verts = rest.trim().parse()?;
        } else if let Some(rest) = l.strip_prefix("element face ") {
            n_faces = rest.trim().parse()?;
        } else if l == "property float nx" {
            has_normals = true;
        } else if l == "end_header" {
            break;
        }
    }

    let mut positions = Vec::with_capacity(n_verts);
    let mut normals = Vec::with_capacity(if has_normals { n_verts } else { 0 });
    for _ in 0..n_verts {
        line.clear();
        anyhow::ensure!(
            reader.read_line(&mut line)? > 0,
            "unexpected EOF in PLY vertices"
        );
        let v: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect::<Result<_, _>>()?;
        anyhow::ensure!(v.len() >= 3, "vertex line has fewer than 3 floats");
        positions.push([v[0], v[1], v[2]]);
        if has_normals {
            anyhow::ensure!(v.len() >= 6, "expected normals but vertex line is short");
            normals.push([v[3], v[4], v[5]]);
        }
    }

    let mut indices = Vec::with_capacity(n_faces * 3);
    for _ in 0..n_faces {
        line.clear();
        anyhow::ensure!(
            reader.read_line(&mut line)? > 0,
            "unexpected EOF in PLY faces"
        );
        let f: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()?;
        anyhow::ensure!(
            f.len() >= 4 && f[0] == 3,
            "only triangle faces are supported"
        );
        indices.extend_from_slice(&f[1..4]);
    }

    Ok(Mesh {
        positions,
        normals,
        indices,
    })
}

/// Open an interactive window showing `mesh`, with orbit/zoom controls.
///
/// Blocks until the window is closed. Requires the `window` feature.
#[cfg(feature = "window")]
pub fn view_mesh(mesh: &Mesh, title: &str) -> anyhow::Result<()> {
    use three_d::{
        degrees, vec3, AmbientLight, Camera, ClearState, CpuMaterial, CpuMesh, DirectionalLight,
        FrameOutput, Gm, Indices, Light, OrbitControl, PhysicalMaterial, Positions, Srgba, Vec3,
        Window, WindowSettings,
    };

    anyhow::ensure!(!mesh.positions.is_empty(), "mesh has no vertices");

    let window = Window::new(WindowSettings {
        title: title.to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let context = window.gl();

    // Axis-aligned bounds for framing the camera.
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for p in &mesh.positions {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    let center = vec3(
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    );
    let dx = mx[0] - mn[0];
    let dy = mx[1] - mn[1];
    let dz = mx[2] - mn[2];
    let extent = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-3);

    let positions: Vec<Vec3> = mesh
        .positions
        .iter()
        .map(|p| vec3(p[0], p[1], p[2]))
        .collect();
    let mut cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(mesh.indices.clone()),
        ..Default::default()
    };
    if mesh.normals.len() == mesh.positions.len() {
        cpu_mesh.normals = Some(
            mesh.normals
                .iter()
                .map(|n| vec3(n[0], n[1], n[2]))
                .collect(),
        );
    } else {
        cpu_mesh.compute_normals();
    }

    let model = Gm::new(
        three_d::Mesh::new(&context, &cpu_mesh),
        PhysicalMaterial::new_opaque(
            &context,
            &CpuMaterial {
                albedo: Srgba::new(180, 185, 195, 255),
                roughness: 0.7,
                metallic: 0.1,
                ..Default::default()
            },
        ),
    );

    let mut camera = Camera::new_perspective(
        window.viewport(),
        center + vec3(extent * 0.6, -extent * 0.6, -extent * 1.4),
        center,
        vec3(0.0, -1.0, 0.0),
        degrees(45.0),
        extent * 0.01,
        extent * 10.0,
    );
    let mut control = OrbitControl::new(center, extent * 0.1, extent * 5.0);
    let light = DirectionalLight::new(&context, 2.0, Srgba::WHITE, vec3(-0.5, -1.0, -0.7));
    let ambient = AmbientLight::new(&context, 0.4, Srgba::WHITE);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);
        control.handle_events(&mut camera, &mut frame_input.events);
        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.08, 0.09, 0.11, 1.0, 1.0))
            .render(
                &camera,
                &model,
                &[&light as &dyn Light, &ambient as &dyn Light],
            );
        FrameOutput::default()
    });

    Ok(())
}

/// Open a window that displays a live RGB frame stream from any capture source
/// (webcam, image file, synthetic). Blocks until the window is closed.
/// Requires the `window` feature.
#[cfg(feature = "window")]
pub fn view_frames<F>(mut next_frame: F, title: &str) -> anyhow::Result<()>
where
    F: FnMut() -> anyhow::Result<Option<ge_backend_trait::Frame>> + 'static,
{
    use three_d::{
        degrees, vec2, Camera, ClearState, ColorMaterial, CpuTexture, FrameOutput, Gm, Rectangle,
        Srgba, Texture2DRef, TextureData, Window, WindowSettings,
    };

    let window = Window::new(WindowSettings {
        title: title.to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let context = window.gl();

    let mut quad: Option<Gm<Rectangle, ColorMaterial>> = None;

    window.render_loop(move |frame_input| {
        // Pull the latest frame; keep the previous one on transient errors.
        if let Ok(Some(frame)) = next_frame() {
            if frame.rgb.len() == frame.pixel_count() * 3 {
                // three-d's 2D origin is bottom-left while camera rows are
                // top-first, so flip vertically to keep the image upright.
                let w = frame.width as usize;
                let mut data: Vec<[u8; 3]> = Vec::with_capacity(frame.pixel_count());
                for row in (0..frame.height as usize).rev() {
                    let base = row * w * 3;
                    for x in 0..w {
                        let p = base + x * 3;
                        data.push([frame.rgb[p], frame.rgb[p + 1], frame.rgb[p + 2]]);
                    }
                }
                let cpu = CpuTexture {
                    data: TextureData::RgbU8(data),
                    width: frame.width,
                    height: frame.height,
                    ..Default::default()
                };
                let vp = frame_input.viewport;
                let rect = Rectangle::new(
                    &context,
                    vec2(vp.width as f32 * 0.5, vp.height as f32 * 0.5),
                    degrees(0.0),
                    vp.width as f32,
                    vp.height as f32,
                );
                quad = Some(Gm::new(
                    rect,
                    ColorMaterial {
                        texture: Some(Texture2DRef::from_cpu_texture(&context, &cpu)),
                        color: Srgba::WHITE,
                        ..Default::default()
                    },
                ));
            }
        }

        let camera = Camera::new_2d(frame_input.viewport);
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.0, 0.0, 0.0, 1.0, 1.0));
        if let Some(q) = &quad {
            screen.render(&camera, q, &[]);
        }
        FrameOutput::default()
    });

    Ok(())
}

/// A live webcam capture source (nokhwa). Requires the `camera` feature.
#[cfg(feature = "camera")]
pub struct WebcamSource {
    camera: nokhwa::Camera,
}

#[cfg(feature = "camera")]
impl WebcamSource {
    /// Open the camera at `index` (0 = default) and start streaming.
    ///
    /// On macOS this triggers a camera-permission prompt; the process must be
    /// granted access or capture fails.
    pub fn open(index: u32) -> anyhow::Result<Self> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
        use nokhwa::Camera;

        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(CameraIndex::Index(index), requested)
            .map_err(|e| anyhow::anyhow!("open camera {index}: {e}"))?;
        camera
            .open_stream()
            .map_err(|e| anyhow::anyhow!("open stream: {e}"))?;
        Ok(Self { camera })
    }
}

#[cfg(feature = "camera")]
impl WebcamSource {
    /// Capture the next frame as an RGB8 [`ge_backend_trait::Frame`].
    pub fn next_frame(&mut self) -> anyhow::Result<Option<ge_backend_trait::Frame>> {
        use nokhwa::pixel_format::RgbFormat;
        let buffer = self
            .camera
            .frame()
            .map_err(|e| anyhow::anyhow!("capture frame: {e}"))?;
        let img = buffer
            .decode_image::<RgbFormat>()
            .map_err(|e| anyhow::anyhow!("decode frame: {e}"))?;
        let (width, height) = (img.width(), img.height());
        Ok(Some(ge_backend_trait::Frame {
            width,
            height,
            timestamp_ns: 0,
            rgb: img.into_raw(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_round_trips_through_loader() {
        let mut m = Mesh::quad([[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        m.normals = vec![[0.0, 0.0, 1.0]; 4];
        let path = std::env::temp_dir().join(format!("ge-viewer-ply-{}.ply", std::process::id()));
        m.write_ply(&path).unwrap();

        let loaded = load_ply(&path).unwrap();
        assert_eq!(loaded.vertex_count(), 4);
        assert_eq!(loaded.triangle_count(), 2);
        assert_eq!(loaded.normals.len(), 4);
        assert_eq!(loaded.positions[1], [1.0, 0.0, 0.0]);
        let _ = std::fs::remove_file(path);
    }
}
