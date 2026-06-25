//! Closed-form symmetric 3×3 eigensolver (Cardano), used for plane fitting from
//! covariance/moment matrices. Hand-rolled f64, no dependencies — matching the
//! project's no-cmake, no-linalg-crate ethos.

/// Smallest eigenvalue and its unit eigenvector of the symmetric matrix
/// `[[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]]`. For a covariance matrix this is
/// the surface normal and the (variance) flatness residual.
pub fn smallest_eigen(
    a00: f64,
    a01: f64,
    a02: f64,
    a11: f64,
    a12: f64,
    a22: f64,
) -> (f64, [f64; 3]) {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    let lambda_min = if p1 < 1e-20 {
        a00.min(a11).min(a22)
    } else {
        let q = (a00 + a11 + a22) / 3.0;
        let p2 = (a00 - q).powi(2) + (a11 - q).powi(2) + (a22 - q).powi(2) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let (b00, b11, b22) = ((a00 - q) / p, (a11 - q) / p, (a22 - q) / p);
        let (b01, b02, b12) = (a01 / p, a02 / p, a12 / p);
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);
        let r = (det_b / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;
        // Three eigenvalues; the one at phi + 2π/3 is the smallest.
        q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos()
    };

    // Eigenvector of (A − λI): cross product of two of its rows.
    let lam = lambda_min;
    let r0 = [a00 - lam, a01, a02];
    let r1 = [a01, a11 - lam, a12];
    let r2 = [a02, a12, a22 - lam];
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let mag = |c: [f64; 3]| c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    let candidates = [cross(r0, r1), cross(r1, r2), cross(r2, r0)];
    let mut best = candidates[0];
    let mut best_mag = mag(candidates[0]);
    for &c in &candidates[1..] {
        let m = mag(c);
        if m > best_mag {
            best = c;
            best_mag = m;
        }
    }
    if best_mag < 1e-30 {
        return (lam, [0.0, 0.0, 1.0]); // degenerate (e.g. isotropic)
    }
    let inv = 1.0 / best_mag.sqrt();
    (lam, [best[0] * inv, best[1] * inv, best[2] * inv])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix() {
        // diag(3, 2, 1): smallest eigenvalue 1 with eigenvector ±z.
        let (lam, v) = smallest_eigen(3.0, 0.0, 0.0, 2.0, 0.0, 1.0);
        assert!((lam - 1.0).abs() < 1e-9, "lambda {lam}");
        assert!(v[2].abs() > 0.999, "vec {v:?}");
    }

    #[test]
    fn flat_in_z_plane() {
        // Covariance of points spread in x,y but flat in z: normal ≈ z.
        let (lam, v) = smallest_eigen(5.0, 0.4, 0.0, 4.0, 0.0, 1e-6);
        assert!(lam < 1e-3, "should be near-zero flatness: {lam}");
        assert!(v[2].abs() > 0.999, "normal should be z: {v:?}");
    }
}
