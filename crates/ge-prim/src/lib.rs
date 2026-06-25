//! Low-poly primitive reconstruction.
//!
//! Gods Eye represents the world as a small set of clean geometric primitives
//! (planes/quads) rather than a dense, noisy triangle mesh. Fitting a primitive
//! to many noisy depth samples **averages out** the noise, so a wall becomes a
//! single clean plane instead of thousands of jittery triangles.
//!
//! This crate starts with the irreducible core — robust least-squares plane
//! fitting — and grows detection, persistence, and polygonization on top.

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
}

/// Fit a plane to `points` by total least squares (PCA): the plane through the
/// centroid whose normal is the covariance eigenvector with the smallest
/// eigenvalue. Robust to per-point noise because it minimises squared
/// orthogonal distance over all points. Returns `None` for < 3 points or a
/// degenerate (collinear) set.
pub fn fit_plane(points: &[Vec3]) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let (mut cx, mut cy, mut cz) = (0.0f64, 0.0, 0.0);
    for p in points {
        cx += p.x as f64;
        cy += p.y as f64;
        cz += p.z as f64;
    }
    cx /= n;
    cy /= n;
    cz /= n;

    // Symmetric covariance.
    let (mut c00, mut c01, mut c02, mut c11, mut c12, mut c22) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy, dz) = (p.x as f64 - cx, p.y as f64 - cy, p.z as f64 - cz);
        c00 += dx * dx;
        c01 += dx * dy;
        c02 += dx * dz;
        c11 += dy * dy;
        c12 += dy * dz;
        c22 += dz * dz;
    }

    // Smallest eigenvector of C = largest eigenvector of M = kI − C
    // (k chosen so M is well-conditioned). Found by power iteration.
    let k = c00 + c11 + c22 + 1.0;
    let m = [
        [k - c00, -c01, -c02],
        [-c01, k - c11, -c12],
        [-c02, -c12, k - c22],
    ];
    let mut v = [0.41f64, 0.31, -0.86]; // arbitrary, not axis-aligned
    for _ in 0..128 {
        let nv = [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ];
        let len = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        if len < 1e-12 {
            return None;
        }
        let nv = [nv[0] / len, nv[1] / len, nv[2] / len];
        let delta = (nv[0] - v[0]).abs() + (nv[1] - v[1]).abs() + (nv[2] - v[2]).abs();
        v = nv;
        if delta < 1e-10 {
            break;
        }
    }

    let normal = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32).normalize();
    let centroid = Vec3::new(cx as f32, cy as f32, cz as f32);
    Some(Plane {
        normal,
        offset: normal.dot(centroid),
    })
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
