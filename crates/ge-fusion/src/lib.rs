//! Truncated signed-distance (TSDF) fusion.
//!
//! Design target: a sparse hash of 8³ voxel blocks allocated only near
//! surfaces, depth-confidence-weighted integration, and LRU streaming, with the
//! integrate step running as a `wgpu` compute kernel. This module currently
//! holds a **dense CPU grid** — correct and simple — so the
//! depth → unproject → fuse → mesh chain can be validated end-to-end. The sparse
//! hash + GPU kernel replace the dense grid next, behind the same surface.

use ge_backend_trait::{DepthMap, Intrinsics, Pose};
use ge_mesh::Mesh;
use glam::Vec3;

/// TSDF parameters (block/sparse design constants, retained for the GPU path).
#[derive(Clone, Copy, Debug)]
pub struct TsdfConfig {
    pub voxel_size_m: f32,
    pub truncation_m: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.02,
            truncation_m: 0.08,
        }
    }
}

/// Voxel blocks are 8×8×8 (for the future sparse-hash layout).
pub const BLOCK_DIM: i32 = 8;

/// Map a world-space voxel coordinate to its containing block coordinate.
#[inline]
pub fn block_of(voxel: glam::IVec3) -> glam::IVec3 {
    glam::IVec3::new(
        voxel.x.div_euclid(BLOCK_DIM),
        voxel.y.div_euclid(BLOCK_DIM),
        voxel.z.div_euclid(BLOCK_DIM),
    )
}

/// A dense TSDF volume on a regular grid.
///
/// Voxel `(x,y,z)` covers world point
/// `origin + (xyz + 0.5) * voxel_size`. `tsdf` holds the signed distance to the
/// nearest surface, normalized to `[-1, 1]` by `trunc` (negative = behind the
/// surface / inside, positive = free space toward the camera). `weight` is the
/// running observation count for the weighted average.
pub struct Tsdf {
    pub dims: [u32; 3],
    pub voxel_size: f32,
    pub origin: Vec3,
    pub trunc: f32,
    tsdf: Vec<f32>,
    weight: Vec<f32>,
}

impl Tsdf {
    pub fn new(dims: [u32; 3], voxel_size: f32, origin: [f32; 3], trunc: f32) -> Self {
        let n = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        Self {
            dims,
            voxel_size,
            origin: Vec3::from_array(origin),
            trunc,
            tsdf: vec![1.0; n],
            weight: vec![0.0; n],
        }
    }

    #[inline]
    fn linear(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.dims[0] + z * self.dims[0] * self.dims[1]) as usize
    }

    pub fn voxel_count(&self) -> usize {
        self.tsdf.len()
    }

    /// Number of voxels that received at least one observation.
    pub fn observed_voxels(&self) -> usize {
        self.weight.iter().filter(|&&w| w > 0.0).count()
    }

    /// Integrate one depth frame, given its camera intrinsics and the
    /// camera-to-world pose (projective TSDF integration).
    pub fn integrate(&mut self, depth: &DepthMap, intr: &Intrinsics, cam_to_world: &Pose) {
        let world_to_cam = cam_to_world.inverse();
        let half = 0.5 * self.voxel_size;
        let (dw, dh) = (depth.width as f32, depth.height as f32);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let world = self.origin
                        + Vec3::new(x as f32, y as f32, z as f32) * self.voxel_size
                        + Vec3::splat(half);
                    let c = world_to_cam.transform_point3(world);
                    if c.z <= 1e-4 {
                        continue;
                    }
                    let u = intr.fx * c.x / c.z + intr.cx;
                    let v = intr.fy * c.y / c.z + intr.cy;
                    if u < 0.0 || v < 0.0 || u >= dw || v >= dh {
                        continue;
                    }
                    let d = depth.depth_m[(v as usize) * depth.width as usize + (u as usize)];
                    if d <= 0.0 || d.is_nan() {
                        continue;
                    }
                    // Signed distance along the ray: + in front of the surface.
                    let sdf = d - c.z;
                    if sdf < -self.trunc {
                        continue; // occluded / behind surface beyond truncation
                    }
                    let val = (sdf / self.trunc).clamp(-1.0, 1.0);
                    let i = self.linear(x, y, z);
                    let w = self.weight[i];
                    self.tsdf[i] = (self.tsdf[i] * w + val) / (w + 1.0);
                    self.weight[i] = w + 1.0;
                }
            }
        }
    }

    /// Extract a world-space triangle mesh from the current volume.
    pub fn extract_mesh(&self) -> Mesh {
        let mut mesh = ge_mesh::surface_nets_mesh(&self.tsdf, self.dims);
        for p in mesh.positions.iter_mut() {
            p[0] = self.origin.x + p[0] * self.voxel_size;
            p[1] = self.origin.y + p[1] * self.voxel_size;
            p[2] = self.origin.z + p[2] * self.voxel_size;
        }
        mesh
    }
}

/// Synthetic depth scenes for validating the fusion + meshing chain without a
/// camera or model.
pub mod scenes {
    use ge_backend_trait::{DepthMap, Intrinsics};

    /// A frontal wall at `wall_z` metres with a nearer square panel at
    /// `panel_z` in the centre — mimics "a wall with a raised painting/shelf".
    /// Returns the depth map and matching pinhole intrinsics (60° FOV).
    pub fn wall_with_panel(size: u32) -> (DepthMap, Intrinsics) {
        let f = (size as f32) / 2.0 / (60.0f32.to_radians() / 2.0).tan();
        let intr = Intrinsics {
            fx: f,
            fy: f,
            cx: size as f32 / 2.0,
            cy: size as f32 / 2.0,
            width: size,
            height: size,
        };
        let (wall_z, panel_z) = (2.5f32, 1.8f32);
        let (lo, hi) = (size / 3, 2 * size / 3);
        let mut depth_m = vec![0.0f32; (size * size) as usize];
        for v in 0..size {
            for u in 0..size {
                let in_panel = u >= lo && u < hi && v >= lo && v < hi;
                depth_m[(v * size + u) as usize] = if in_panel { panel_z } else { wall_z };
            }
        }
        (
            DepthMap {
                width: size,
                height: size,
                depth_m,
            },
            intr,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn block_of_handles_negative_coords() {
        assert_eq!(block_of(IVec3::new(0, 0, 0)), IVec3::new(0, 0, 0));
        assert_eq!(block_of(IVec3::new(7, 7, 7)), IVec3::new(0, 0, 0));
        assert_eq!(block_of(IVec3::new(8, -1, -8)), IVec3::new(1, -1, -1));
    }

    #[test]
    fn integrate_wall_yields_surface_near_expected_depth() {
        let (depth, intr) = scenes::wall_with_panel(128);
        // Volume spanning the frustum; z brackets both wall (2.5) and panel (1.8).
        let voxel = 0.03;
        let origin = [-1.8, -1.8, 1.2];
        let dims = [120, 120, 60];
        let mut tsdf = Tsdf::new(dims, voxel, origin, 4.0 * voxel);
        tsdf.integrate(&depth, &intr, &Pose::IDENTITY);
        assert!(tsdf.observed_voxels() > 0, "some voxels observed");
        let mesh = tsdf.extract_mesh();
        assert!(mesh.triangle_count() > 100, "wall produces a real surface");
        // Mean vertex depth should sit between the panel and wall planes.
        let mean_z = mesh.positions.iter().map(|p| p[2]).sum::<f32>() / mesh.positions.len() as f32;
        assert!(
            (1.6..2.7).contains(&mean_z),
            "surface depth {mean_z} should be near the wall/panel"
        );
    }
}
