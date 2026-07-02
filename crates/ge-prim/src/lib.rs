//! Low-poly primitive reconstruction.
//!
//! Gods Eye represents the world as a small set of clean geometric primitives
//! (planes/quads) rather than a dense, noisy triangle mesh. Fitting a primitive
//! to many noisy depth samples **averages out** the noise, so a wall becomes a
//! single clean plane instead of thousands of jittery triangles.
//!
//! This crate starts with the irreducible core — robust least-squares plane
//! fitting via geometric moments — and grows detection, persistence, and
//! polygonization on top.

pub mod detect;
mod eigen;
pub mod moments;
pub mod polygon;
pub mod registry;

pub use detect::{detect_planes, DetectParams, Segment};
pub use moments::Moments;
pub use polygon::{footprint_polygon, snap_to_line, triangulate, PolyParams};
pub use registry::{RegistryParams, WorldPlaneRegistry};

use glam::Vec3;

/// An (infinite) plane `{ p : normal·p = offset }` with a unit `normal`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    /// Signed distance from `p` to the plane (along `normal`).
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.offset
    }

    /// Orthogonally project `p` onto the plane.
    #[inline]
    pub fn project(&self, p: Vec3) -> Vec3 {
        p - self.normal * self.signed_distance(p)
    }

    /// An orthonormal in-plane basis `(u, v)` perpendicular to `normal`.
    pub fn basis(&self) -> (Vec3, Vec3) {
        let a = if self.normal.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let u = self.normal.cross(a).normalize();
        let v = self.normal.cross(u).normalize();
        (u, v)
    }

    /// World point on the plane for in-plane coordinates `(a, b)` measured along
    /// `u`/`v` from the world origin (so `a = p·u`, `b = p·v` for `p` on-plane).
    #[inline]
    pub fn point_from_uv(&self, a: f32, b: f32, u: Vec3, v: Vec3) -> Vec3 {
        u * a + v * b + self.normal * self.offset
    }
}

/// Fit a plane to `points` by total least squares (the covariance eigenvector
/// with the smallest eigenvalue). Robust to per-point noise because it minimises
/// squared orthogonal distance over all points. Convenience wrapper over
/// [`Moments`]; the per-cell/per-segment paths accumulate moments directly
/// rather than building a point `Vec`.
pub fn fit_plane(points: &[Vec3]) -> Option<Plane> {
    let mut m = Moments::new();
    for p in points {
        m.add_point(1.0, *p);
    }
    m.fit().map(|(plane, _flatness)| plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic [-1, 1] pseudo-noise (no Math.random / Date in scope).
    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 9) as f32 / 8_388_608.0 - 1.0
    }

    #[test]
    fn fit_plane_recovers_known_plane_under_noise() {
        // Plane z = 0.5x + 0.3y + 1  ⇒  normal ∝ (0.5, 0.3, -1).
        let mut seed = 12345u32;
        let mut pts = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (x, y) = (i as f32 * 0.1, j as f32 * 0.1);
                let z = 0.5 * x + 0.3 * y + 1.0 + lcg(&mut seed) * 0.01; // ~1 cm noise
                pts.push(Vec3::new(x, y, z));
            }
        }
        let plane = fit_plane(&pts).expect("plane");
        let true_n = Vec3::new(0.5, 0.3, -1.0).normalize();
        assert!(
            plane.normal.dot(true_n).abs() > 0.999,
            "normal off: {:?}",
            plane.normal
        );
        let max_resid = pts
            .iter()
            .map(|p| plane.signed_distance(*p).abs())
            .fold(0.0f32, f32::max);
        assert!(max_resid < 0.05, "max residual {max_resid} too large");
    }

    #[test]
    fn project_lands_on_plane() {
        let pl = Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            offset: 2.0,
        };
        let q = pl.project(Vec3::new(3.0, -4.0, 7.0));
        assert!(pl.signed_distance(q).abs() < 1e-5);
        assert!((q.z - 2.0).abs() < 1e-5);
    }
}
