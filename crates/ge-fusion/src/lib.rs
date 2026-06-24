//! Sparse voxel-hashed TSDF fusion.
//!
//! Design: a flat hash table of 8³ voxel blocks allocated only in occupied,
//! near-surface space (integer hashing, WGSL-native), depth-confidence-weighted
//! integration, and out-of-core block streaming with LRU eviction. Integration
//! runs as a `wgpu` compute kernel (`shaders/integrate.wgsl`). This module
//! currently holds the config + voxel-math; the GPU kernel lands next.

/// TSDF fusion parameters.
#[derive(Clone, Copy, Debug)]
pub struct TsdfConfig {
    /// Edge length of one voxel, in meters.
    pub voxel_size_m: f32,
    /// Truncation distance for the signed-distance field, in meters.
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

/// Voxel blocks are 8×8×8.
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
}
