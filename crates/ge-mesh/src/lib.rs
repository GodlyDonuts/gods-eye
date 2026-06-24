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
}
