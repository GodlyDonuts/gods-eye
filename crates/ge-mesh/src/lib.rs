//! Mesh extraction and adaptive level-of-detail.
//!
//! Pipeline (lands incrementally):
//! 1. Fixed-resolution dirty-block extraction — GPU marching cubes (WGSL,
//!    `shaders/marching_cubes.wgsl`) with a `fast-surface-nets` CPU mesher as
//!    the guaranteed-portable floor.
//! 2. Adaptive LOD as a *post-process*: RANSAC plane detection emits large flat
//!    surfaces as 2-triangle quads; `meshopt` simplifies the rest with
//!    crack-free locked borders; detail accrues over frames (observation-driven
//!    refinement). See `docs/design/ARCHITECTURE.md`.

use std::io::Write;
use std::path::Path;

/// A triangle mesh handed to the viewer. Positions/normals are world-space.
#[derive(Clone, Default)]
pub struct Mesh {
    /// `xyz` positions.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals (may be empty until computed).
    pub normals: Vec<[f32; 3]>,
    /// Triangle list; each consecutive triple indexes `positions`.
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// An axis-aligned quad (two triangles) — the minimal representation a large
    /// flat surface collapses to. Winding is CCW around `+normal`.
    pub fn quad(corners: [[f32; 3]; 4]) -> Mesh {
        Mesh {
            positions: corners.to_vec(),
            normals: Vec::new(),
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    /// Write the mesh as an ASCII PLY file (openable in MeshLab, Blender,
    /// macOS Quick Look, etc.). Normals are included when present.
    pub fn write_ply(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let has_normals = self.normals.len() == self.positions.len();
        writeln!(f, "ply")?;
        writeln!(f, "format ascii 1.0")?;
        writeln!(f, "comment Gods Eye mesh export")?;
        writeln!(f, "element vertex {}", self.positions.len())?;
        writeln!(f, "property float x")?;
        writeln!(f, "property float y")?;
        writeln!(f, "property float z")?;
        if has_normals {
            writeln!(f, "property float nx")?;
            writeln!(f, "property float ny")?;
            writeln!(f, "property float nz")?;
        }
        writeln!(f, "element face {}", self.triangle_count())?;
        writeln!(f, "property list uchar int vertex_indices")?;
        writeln!(f, "end_header")?;
        for (i, p) in self.positions.iter().enumerate() {
            if has_normals {
                let n = self.normals[i];
                writeln!(f, "{} {} {} {} {} {}", p[0], p[1], p[2], n[0], n[1], n[2])?;
            } else {
                writeln!(f, "{} {} {}", p[0], p[1], p[2])?;
            }
        }
        for t in self.indices.chunks_exact(3) {
            writeln!(f, "3 {} {} {}", t[0], t[1], t[2])?;
        }
        Ok(())
    }
}

/// Synthetic signed-distance fields meshed with the CPU floor — used to validate
/// the extraction stage against known ground truth before real depth is fused.
pub mod demo {
    use super::Mesh;
    use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
    use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};

    /// Grid resolution per axis (including the 1-voxel border surface-nets needs).
    pub const N: u32 = 64;
    type GridShape = ConstShape3u32<64, 64, 64>;

    /// Mesh a sphere of `radius` (in normalized units, grid spans ~[-1, 1]).
    pub fn sphere_mesh(radius: f32) -> Mesh {
        let half = (N as f32) / 2.0;
        let center = half - 0.5;
        let sdf: Vec<f32> = (0..GridShape::SIZE)
            .map(|i| {
                let [x, y, z] = GridShape::delinearize(i);
                let px = (x as f32 - center) / half;
                let py = (y as f32 - center) / half;
                let pz = (z as f32 - center) / half;
                (px * px + py * py + pz * pz).sqrt() - radius
            })
            .collect();

        let mut buffer = SurfaceNetsBuffer::default();
        surface_nets(&sdf, &GridShape {}, [0; 3], [N - 1; 3], &mut buffer);
        Mesh {
            positions: buffer.positions,
            normals: buffer.normals,
            indices: buffer.indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_is_two_triangles() {
        let m = Mesh::quad([[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4);
    }

    #[test]
    fn sphere_mesh_is_nonempty_and_closed_ish() {
        let m = demo::sphere_mesh(0.7);
        assert!(
            m.vertex_count() > 100,
            "sphere should produce many vertices"
        );
        assert!(
            m.triangle_count() > 100,
            "sphere should produce many triangles"
        );
        assert_eq!(m.normals.len(), m.positions.len(), "normals per vertex");
    }
}
