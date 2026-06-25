//! Running geometric moments — the currency of primitive fusion.
//!
//! A plane is summarised by its zeroth/first/second moments `N`, `S = Σ w·p`,
//! `Q = Σ w·p·pᵀ`. These are **additive** (fuse two observations by summing
//! moments) and **transform linearly** under rigid motion (so a per-frame
//! observation can be lifted into world space without storing any points).
//! Fitting a plane from moments — centroid `μ = S/N`, covariance `C = Q/N − μμᵀ`,
//! normal = smallest-eigenvalue eigenvector of `C` — is the one code path shared
//! by per-cell detection, per-segment refit, and cross-frame world fusion.

use glam::{Affine3A, Vec3};

use crate::eigen::smallest_eigen;
use crate::Plane;

/// Precision-weighted raw moments in some fixed frame. `q` is the upper triangle
/// of the symmetric `Σ w·p·pᵀ` as `[xx, xy, xz, yy, yz, zz]`.
#[derive(Clone, Debug, Default)]
pub struct Moments {
    pub n: f64,
    pub s: [f64; 3],
    pub q: [f64; 6],
}

impl Moments {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate a point with weight `w` (e.g. depth confidence / z²).
    pub fn add_point(&mut self, w: f64, p: Vec3) {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        self.n += w;
        self.s[0] += w * x;
        self.s[1] += w * y;
        self.s[2] += w * z;
        self.q[0] += w * x * x;
        self.q[1] += w * x * y;
        self.q[2] += w * x * z;
        self.q[3] += w * y * y;
        self.q[4] += w * y * z;
        self.q[5] += w * z * z;
    }

    /// Fuse another observation (of the same surface) into this one.
    pub fn merge(&mut self, o: &Moments) {
        self.n += o.n;
        for i in 0..3 {
            self.s[i] += o.s[i];
        }
        for i in 0..6 {
            self.q[i] += o.q[i];
        }
    }

    /// Rigidly transform the moments by `t` (e.g. camera→world). Exact: no
    /// points are re-projected.
    pub fn transform(&self, t: &Affine3A) -> Moments {
        let m = t.matrix3;
        // r[row][col] (matrix3 columns are x_axis/y_axis/z_axis).
        let r = [
            [m.x_axis.x as f64, m.y_axis.x as f64, m.z_axis.x as f64],
            [m.x_axis.y as f64, m.y_axis.y as f64, m.z_axis.y as f64],
            [m.x_axis.z as f64, m.y_axis.z as f64, m.z_axis.z as f64],
        ];
        let tr = [
            t.translation.x as f64,
            t.translation.y as f64,
            t.translation.z as f64,
        ];

        // S' = R·S + N·t
        let rs = [
            r[0][0] * self.s[0] + r[0][1] * self.s[1] + r[0][2] * self.s[2],
            r[1][0] * self.s[0] + r[1][1] * self.s[1] + r[1][2] * self.s[2],
            r[2][0] * self.s[0] + r[2][1] * self.s[1] + r[2][2] * self.s[2],
        ];
        let s2 = [
            rs[0] + self.n * tr[0],
            rs[1] + self.n * tr[1],
            rs[2] + self.n * tr[2],
        ];

        // Q' = R·Q·Rᵀ + (R·S)·tᵀ + t·(R·S)ᵀ + N·t·tᵀ
        let qm = [
            [self.q[0], self.q[1], self.q[2]],
            [self.q[1], self.q[3], self.q[4]],
            [self.q[2], self.q[4], self.q[5]],
        ];
        let mut rq = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rq[i][j] = r[i][0] * qm[0][j] + r[i][1] * qm[1][j] + r[i][2] * qm[2][j];
            }
        }
        let mut q2 = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let rqrt = rq[i][0] * r[j][0] + rq[i][1] * r[j][1] + rq[i][2] * r[j][2];
                q2[i][j] = rqrt + rs[i] * tr[j] + tr[i] * rs[j] + self.n * tr[i] * tr[j];
            }
        }
        Moments {
            n: self.n,
            s: s2,
            q: [q2[0][0], q2[0][1], q2[0][2], q2[1][1], q2[1][2], q2[2][2]],
        }
    }

    /// Centroid `S/N`, or `None` if empty.
    pub fn centroid(&self) -> Option<Vec3> {
        if self.n <= 0.0 {
            return None;
        }
        let inv = 1.0 / self.n;
        Some(Vec3::new(
            (self.s[0] * inv) as f32,
            (self.s[1] * inv) as f32,
            (self.s[2] * inv) as f32,
        ))
    }

    /// Fit a plane to the accumulated moments, returning `(plane, flatness)`
    /// where `flatness` is the smallest covariance eigenvalue (≈ mean squared
    /// orthogonal residual — small means truly planar).
    pub fn fit(&self) -> Option<(Plane, f32)> {
        if self.n < 3.0 {
            return None;
        }
        let inv = 1.0 / self.n;
        let mu = [self.s[0] * inv, self.s[1] * inv, self.s[2] * inv];
        let c00 = self.q[0] * inv - mu[0] * mu[0];
        let c01 = self.q[1] * inv - mu[0] * mu[1];
        let c02 = self.q[2] * inv - mu[0] * mu[2];
        let c11 = self.q[3] * inv - mu[1] * mu[1];
        let c12 = self.q[4] * inv - mu[1] * mu[2];
        let c22 = self.q[5] * inv - mu[2] * mu[2];
        let (lam, v) = smallest_eigen(c00, c01, c02, c11, c12, c22);
        let normal = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32).normalize();
        let mu_v = Vec3::new(mu[0] as f32, mu[1] as f32, mu[2] as f32);
        Some((
            Plane {
                normal,
                offset: normal.dot(mu_v),
            },
            lam.max(0.0) as f32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{EulerRot, Quat};

    fn planar_points() -> Vec<Vec3> {
        let mut seed = 999u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 9) as f32 / 8_388_608.0 - 1.0
        };
        let mut v = Vec::new();
        for i in 0..15 {
            for j in 0..15 {
                let (x, y) = (i as f32 * 0.1, j as f32 * 0.1);
                v.push(Vec3::new(x, y, 0.4 * x - 0.2 * y + 1.0 + noise() * 0.008));
            }
        }
        v
    }

    #[test]
    fn moments_are_additive() {
        let pts = planar_points();
        let (mut a, mut b, mut all) = (Moments::new(), Moments::new(), Moments::new());
        for (i, p) in pts.iter().enumerate() {
            all.add_point(1.0, *p);
            if i % 2 == 0 {
                a.add_point(1.0, *p);
            } else {
                b.add_point(1.0, *p);
            }
        }
        a.merge(&b);
        let (pa, _) = a.fit().unwrap();
        let (pall, _) = all.fit().unwrap();
        assert!(pa.normal.dot(pall.normal).abs() > 0.99999);
        assert!((pa.offset - pall.offset).abs() < 1e-4);
    }

    #[test]
    fn moments_transform_invariance() {
        let pts = planar_points();
        let mut m = Moments::new();
        for p in &pts {
            m.add_point(1.0, *p);
        }
        let (p0, _) = m.fit().unwrap();

        let t = Affine3A::from_rotation_translation(
            Quat::from_euler(EulerRot::XYZ, 0.3, -0.5, 0.2),
            Vec3::new(1.0, 2.0, -0.5),
        );
        let (p1, _) = m.transform(&t).fit().unwrap();

        // Normal rotates with R; a point on plane0 maps onto plane1.
        let exp_n = t.transform_vector3(p0.normal).normalize();
        assert!(p1.normal.dot(exp_n).abs() > 0.999, "normal {:?}", p1.normal);
        let q0 = p0.normal * p0.offset; // a point on plane0
        let q1 = t.transform_point3(q0);
        assert!(p1.signed_distance(q1).abs() < 2e-3, "off plane");
    }
}
