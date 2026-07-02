//! Pose estimation / visual odometry.
//!
//! The target is depth-assisted RGB-D visual odometry: because the metric depth
//! model turns the monocular stream into RGB-D, we can estimate full 6-DoF
//! camera motion by aligning successive frames' metric geometry. This file
//! holds the optimisation **core** — linearised point-to-plane ICP — validated
//! offline (see tests) before it is wired to live frames. Frame-to-keyframe
//! data association, keyframing, and drift handling build on top.
//!
//! [`IdentityPose`] remains for the M0 spine.

use ge_backend_trait::{DepthMap, Frame, Intrinsics, Pose, PoseEstimator};
use glam::{Affine2, Affine3A, Quat, Vec2, Vec3};

pub mod sim;

/// A trivial pose estimator that always reports the identity (camera fixed at
/// the origin). Lets fusion run on a static scene.
pub struct IdentityPose;

impl PoseEstimator for IdentityPose {
    fn track(&mut self, _frame: &Frame, _depth: &DepthMap) -> anyhow::Result<Pose> {
        Ok(Pose::IDENTITY)
    }
}

/// Solve the symmetric positive-definite system `A x = b` (6×6) via Cholesky.
/// Returns `None` if `A` is not positive-definite (degenerate constraints).
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut l = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = [0.0f64; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    // Back solve Lᵀ x = y.
    let mut x = [0.0f64; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Estimate the rigid transform `T` that best maps `src` points onto the
/// `dst` surface (points + per-point normals), minimising the point-to-plane
/// error `((T·src - dst)·n)²` via Gauss-Newton (linearised small-angle steps).
///
/// `src[i]` corresponds to `dst[i]`/`dst_normals[i]`. Returns the camera-style
/// transform such that `T.transform_point3(src[i]) ≈ dst[i]`.
pub fn align_point_to_plane(
    src: &[Vec3],
    dst: &[Vec3],
    dst_normals: &[Vec3],
    max_iters: usize,
) -> Affine3A {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), dst_normals.len());

    let mut t = Affine3A::IDENTITY;
    for _ in 0..max_iters {
        let mut ata = [[0.0f64; 6]; 6];
        let mut atb = [0.0f64; 6];
        for ((s_src, d), n) in src.iter().zip(dst).zip(dst_normals) {
            let s = t.transform_point3(*s_src);
            // residual e = (s - d)·n ; we drive it to zero.
            let e = (s - *d).dot(*n) as f64;
            // Jacobian row J = [ s×n , n ] for x = [ω, t].
            let c = s.cross(*n);
            let j = [
                c.x as f64, c.y as f64, c.z as f64, n.x as f64, n.y as f64, n.z as f64,
            ];
            for r in 0..6 {
                atb[r] -= j[r] * e;
                for col in 0..6 {
                    ata[r][col] += j[r] * j[col];
                }
            }
        }
        // Levenberg-style damping for conditioning.
        for k in 0..6 {
            ata[k][k] += 1e-9;
        }
        let Some(x) = solve6(&ata, &atb) else { break };
        let omega = Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
        let trans = Vec3::new(x[3] as f32, x[4] as f32, x[5] as f32);
        let step = Affine3A::from_rotation_translation(Quat::from_scaled_axis(omega), trans);
        t = step * t;
        if omega.length() < 1e-7 && trans.length() < 1e-7 {
            break;
        }
    }
    t
}

/// An organized (per-pixel) point + normal frame in camera coordinates, used as
/// the alignment target for the next frame.
struct RefFrame {
    width: usize,
    height: usize,
    points: Vec<Vec3>,
    normals: Vec<Vec3>,
    valid: Vec<bool>,
}

fn build_ref_frame(depth: &DepthMap, intr: &Intrinsics, max_depth: f32) -> RefFrame {
    let w = depth.width as usize;
    let h = depth.height as usize;
    let mut points = vec![Vec3::ZERO; w * h];
    let mut has_pt = vec![false; w * h];
    for v in 0..h {
        for u in 0..w {
            let d = depth.depth_m[v * w + u];
            if d.is_finite() && d > 0.1 && d < max_depth {
                points[v * w + u] = intr.unproject(u as f32, v as f32, d);
                has_pt[v * w + u] = true;
            }
        }
    }
    // Normals from neighbour cross products, oriented toward the camera (+z).
    let mut normals = vec![Vec3::Z; w * h];
    let mut valid = vec![false; w * h];
    for v in 1..h.saturating_sub(1) {
        for u in 1..w.saturating_sub(1) {
            let i = v * w + u;
            if !has_pt[i] || !has_pt[i + 1] || !has_pt[i + w] {
                continue;
            }
            let n = (points[i + 1] - points[i]).cross(points[i + w] - points[i]);
            let len = n.length();
            if len > 1e-6 {
                let mut nn = n / len;
                if nn.z > 0.0 {
                    nn = -nn;
                }
                normals[i] = nn;
                valid[i] = true;
            }
        }
    }
    RefFrame {
        width: w,
        height: h,
        points,
        normals,
        valid,
    }
}

/// Frame-to-frame depth-assisted visual odometry via projective point-to-plane
/// ICP, accumulating a metric camera-to-world pose.
///
/// This is dead-reckoning (no keyframes or loop closure yet), so it drifts over
/// long sessions — good for "recent" backtracking; global consistency comes with
/// the pose-graph milestone.
pub struct RgbdVoTracker {
    intr: Intrinsics,
    world_from_cam: Affine3A,
    prev: Option<RefFrame>,
    /// Pixel stride for ICP source sampling (speed vs. accuracy).
    pub stride: usize,
    pub max_iters: usize,
    pub max_depth: f32,
    pub dist_thresh: f32,
}

impl RgbdVoTracker {
    pub fn new(intr: Intrinsics) -> Self {
        Self {
            intr,
            world_from_cam: Affine3A::IDENTITY,
            prev: None,
            stride: 4,
            max_iters: 12,
            max_depth: 5.0,
            dist_thresh: 0.2,
        }
    }

    pub fn world_from_cam(&self) -> Affine3A {
        self.world_from_cam
    }

    /// Track a new metric depth frame; returns the updated camera-to-world pose.
    pub fn track(&mut self, depth: &DepthMap) -> Affine3A {
        let cur = build_ref_frame(depth, &self.intr, self.max_depth);
        if self.prev.is_none() {
            self.prev = Some(cur);
            return self.world_from_cam;
        }
        let rel = self.estimate_relative(&cur);
        // world_from_cur = world_from_prev * prev_from_cur
        self.world_from_cam *= rel;
        self.prev = Some(cur);
        self.world_from_cam
    }

    /// Estimate `prev_from_cur` aligning the current frame onto the previous one
    /// (projective association + point-to-plane Gauss-Newton).
    fn estimate_relative(&self, cur: &RefFrame) -> Affine3A {
        let prev = self.prev.as_ref().unwrap();
        let (fx, fy, cx, cy) = (self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy);
        let (pw, ph) = (prev.width, prev.height);
        let mut t = Affine3A::IDENTITY;
        for _ in 0..self.max_iters {
            let mut ata = [[0.0f64; 6]; 6];
            let mut atb = [0.0f64; 6];
            let mut count = 0usize;
            let mut v = 0;
            while v < cur.height {
                let mut u = 0;
                while u < cur.width {
                    let si = v * cur.width + u;
                    if cur.valid[si] {
                        let s = t.transform_point3(cur.points[si]);
                        if s.z > 0.05 {
                            let pu = (fx * s.x / s.z + cx).round();
                            let pv = (fy * s.y / s.z + cy).round();
                            if pu >= 0.0 && pv >= 0.0 && (pu as usize) < pw && (pv as usize) < ph {
                                let qi = (pv as usize) * pw + (pu as usize);
                                if prev.valid[qi] {
                                    let (q, n) = (prev.points[qi], prev.normals[qi]);
                                    if (s - q).length() < self.dist_thresh {
                                        let e = (s - q).dot(n) as f64;
                                        let c = s.cross(n);
                                        let j = [
                                            c.x as f64, c.y as f64, c.z as f64, n.x as f64,
                                            n.y as f64, n.z as f64,
                                        ];
                                        for r in 0..6 {
                                            atb[r] -= j[r] * e;
                                            for col in 0..6 {
                                                ata[r][col] += j[r] * j[col];
                                            }
                                        }
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                    u += self.stride;
                }
                v += self.stride;
            }
            if count < 50 {
                break;
            }
            for k in 0..6 {
                ata[k][k] += 1e-6;
            }
            let Some(x) = solve6(&ata, &atb) else { break };
            let omega = Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
            let trans = Vec3::new(x[3] as f32, x[4] as f32, x[5] as f32);
            let step = Affine3A::from_rotation_translation(Quat::from_scaled_axis(omega), trans);
            t = step * t;
            if omega.length() < 1e-6 && trans.length() < 1e-6 {
                break;
            }
        }
        t
    }
}

// ---------------------------------------------------------------------------
// 2D image-translation tracking (for panorama / rotation development).
//
// Tracking camera rotation in the image plane (a panorama) isolates the
// motion-estimation problem from depth/3D. A pure pan/tilt of the camera moves
// image content by a 2D translation; pyramidal Lucas-Kanade recovers it.
// ---------------------------------------------------------------------------

/// Bilinear sample of a single-channel image at floating `(x, y)` (clamped).
fn sample_bilinear(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = img[y0 * w + x0] + (img[y0 * w + x1] - img[y0 * w + x0]) * fx;
    let bot = img[y1 * w + x0] + (img[y1 * w + x1] - img[y1 * w + x0]) * fx;
    top + (bot - top) * fy
}

fn downsample2(img: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let (x0, y0) = (2 * x, 2 * y);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            out[y * nw + x] =
                0.25 * (img[y0 * w + x0] + img[y0 * w + x1] + img[y1 * w + x0] + img[y1 * w + x1]);
        }
    }
    (out, nw, nh)
}

fn pyramid(img: &[f32], w: usize, h: usize) -> Vec<(Vec<f32>, usize, usize)> {
    let mut levels = vec![(img.to_vec(), w, h)];
    while levels.len() < 5 {
        let (ref im, cw, ch) = *levels.last().unwrap();
        if cw <= 48 || ch <= 48 {
            break;
        }
        let (d, nw, nh) = downsample2(im, cw, ch);
        levels.push((d, nw, nh));
    }
    levels
}

/// Estimate the 2D translation `(dx, dy)` such that `cur(x, y) ≈ prev(x-dx, y-dy)`
/// via pyramidal Lucas-Kanade. Inputs are single-channel intensities.
pub fn estimate_translation(prev: &[f32], cur: &[f32], width: usize, height: usize) -> (f32, f32) {
    let pp = pyramid(prev, width, height);
    let pc = pyramid(cur, width, height);
    let n = pp.len();
    let (mut dx, mut dy) = (0.0f32, 0.0f32);
    for li in (0..n).rev() {
        if li != n - 1 {
            dx *= 2.0;
            dy *= 2.0;
        }
        let (ref p, pw, ph) = pp[li];
        let c = &pc[li].0;
        for _ in 0..12 {
            let (mut g00, mut g01, mut g11, mut b0, mut b1) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for y in 1..ph.saturating_sub(1) {
                for x in 1..pw.saturating_sub(1) {
                    let (sx, sy) = (x as f32 - dx, y as f32 - dy);
                    if sx < 1.0 || sy < 1.0 || sx > (pw - 2) as f32 || sy > (ph - 2) as f32 {
                        continue;
                    }
                    let pv = sample_bilinear(p, pw, ph, sx, sy);
                    let r = (c[y * pw + x] - pv) as f64;
                    let gx = ((sample_bilinear(p, pw, ph, sx + 1.0, sy)
                        - sample_bilinear(p, pw, ph, sx - 1.0, sy))
                        * 0.5) as f64;
                    let gy = ((sample_bilinear(p, pw, ph, sx, sy + 1.0)
                        - sample_bilinear(p, pw, ph, sx, sy - 1.0))
                        * 0.5) as f64;
                    g00 += gx * gx;
                    g01 += gx * gy;
                    g11 += gy * gy;
                    b0 += gx * r;
                    b1 += gy * r;
                }
            }
            let det = g00 * g11 - g01 * g01;
            if det.abs() < 1e-9 {
                break;
            }
            // δ = -G⁻¹ b
            let ddx = -(g11 * b0 - g01 * b1) / det;
            let ddy = -(-g01 * b0 + g00 * b1) / det;
            dx += ddx as f32;
            dy += ddy as f32;
            if ddx.abs() + ddy.abs() < 1e-3 {
                break;
            }
        }
    }
    (dx, dy)
}

/// Solve the 3×3 system `A x = b` via Cramer's rule. `None` if near-singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let det = det3(&a);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut x = [0.0f64; 3];
    for k in 0..3 {
        let mut m = a;
        for (r, &bv) in b.iter().enumerate() {
            m[r][k] = bv;
        }
        x[k] = det3(&m) / det;
    }
    Some(x)
}

/// Estimate a 2D rigid transform (in-plane rotation + translation) `prev_from_cur`
/// such that `cur(x) ≈ prev(M·x)`, via pyramidal Lucas-Kanade. Handles the
/// camera rolling about its optical axis, which translation-only tracking can't.
pub fn estimate_rigid2d(prev: &[f32], cur: &[f32], width: usize, height: usize) -> Affine2 {
    let pp = pyramid(prev, width, height);
    let pc = pyramid(cur, width, height);
    let n = pp.len();
    let (mut theta, mut tx, mut ty) = (0.0f32, 0.0f32, 0.0f32);
    for li in (0..n).rev() {
        if li != n - 1 {
            tx *= 2.0;
            ty *= 2.0; // angle is scale-invariant
        }
        let (ref p, pw, ph) = pp[li];
        let c = &pc[li].0;
        let (cx, cy) = ((pw as f32 - 1.0) * 0.5, (ph as f32 - 1.0) * 0.5);
        for _ in 0..12 {
            let (s, co) = (theta.sin(), theta.cos());
            let mut h = [[0.0f64; 3]; 3];
            let mut bb = [0.0f64; 3];
            for y in 1..ph.saturating_sub(1) {
                for x in 1..pw.saturating_sub(1) {
                    let (xc, yc) = (x as f32 - cx, y as f32 - cy);
                    // W(x) = R(theta)*(xc, yc) + (cx + tx, cy + ty)
                    let wx = co * xc - s * yc + cx + tx;
                    let wy = s * xc + co * yc + cy + ty;
                    if wx < 1.0 || wy < 1.0 || wx > (pw - 2) as f32 || wy > (ph - 2) as f32 {
                        continue;
                    }
                    let pv = sample_bilinear(p, pw, ph, wx, wy);
                    let r = (c[y * pw + x] - pv) as f64;
                    let gx = ((sample_bilinear(p, pw, ph, wx + 1.0, wy)
                        - sample_bilinear(p, pw, ph, wx - 1.0, wy))
                        * 0.5) as f64;
                    let gy = ((sample_bilinear(p, pw, ph, wx, wy + 1.0)
                        - sample_bilinear(p, pw, ph, wx, wy - 1.0))
                        * 0.5) as f64;
                    // dW/dtheta = R'(theta)*(xc, yc)
                    let dwt_x = (-s * xc - co * yc) as f64;
                    let dwt_y = (co * xc - s * yc) as f64;
                    let sd = [gx * dwt_x + gy * dwt_y, gx, gy];
                    for rr in 0..3 {
                        bb[rr] += sd[rr] * r;
                        for cc in 0..3 {
                            h[rr][cc] += sd[rr] * sd[cc];
                        }
                    }
                }
            }
            for k in 0..3 {
                h[k][k] += 1e-9;
            }
            let Some(d) = solve3(h, bb) else { break };
            theta += d[0] as f32;
            tx += d[1] as f32;
            ty += d[2] as f32;
            if d[0].abs() + d[1].abs() + d[2].abs() < 1e-4 {
                break;
            }
        }
    }
    let c = Vec2::new((width as f32 - 1.0) * 0.5, (height as f32 - 1.0) * 0.5);
    Affine2::from_translation(c + Vec2::new(tx, ty))
        * Affine2::from_angle(theta)
        * Affine2::from_translation(-c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::EulerRot;

    /// Points on three perpendicular planes (a room corner) give full 6-DoF
    /// observability for point-to-plane alignment.
    fn corner_cloud() -> (Vec<Vec3>, Vec<Vec3>) {
        let (mut pts, mut normals) = (Vec::new(), Vec::new());
        for i in 0..10 {
            for j in 0..10 {
                let (a, b) = (i as f32 * 0.1, j as f32 * 0.1);
                pts.push(Vec3::new(0.0, a, b));
                normals.push(Vec3::X);
                pts.push(Vec3::new(a, 0.0, b));
                normals.push(Vec3::Y);
                pts.push(Vec3::new(a, b, 0.0));
                normals.push(Vec3::Z);
            }
        }
        (pts, normals)
    }

    #[test]
    fn icp_recovers_known_motion() {
        let (src, normals) = corner_cloud();
        let known = Affine3A::from_rotation_translation(
            Quat::from_euler(EulerRot::XYZ, 0.05, -0.08, 0.06),
            Vec3::new(0.10, -0.05, 0.08),
        );
        let dst: Vec<Vec3> = src.iter().map(|p| known.transform_point3(*p)).collect();
        let dst_normals: Vec<Vec3> = normals
            .iter()
            .map(|n| known.transform_vector3(*n).normalize())
            .collect();

        let est = align_point_to_plane(&src, &dst, &dst_normals, 40);

        let max_err = src
            .iter()
            .zip(&dst)
            .map(|(p, d)| (est.transform_point3(*p) - *d).length())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max alignment error too large: {max_err}");
    }

    #[test]
    fn icp_identity_on_aligned_clouds() {
        let (src, normals) = corner_cloud();
        let est = align_point_to_plane(&src, &src, &normals, 5);
        let max_err = src
            .iter()
            .map(|p| (est.transform_point3(*p) - *p).length())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "should be identity: {max_err}");
    }

    #[test]
    fn vo_static_scene_has_no_drift() {
        let intr = Intrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 64.0,
            width: 128,
            height: 128,
        };
        // A gently curved surface gives normal variety (full observability).
        let mut depth_m = vec![0.0f32; 128 * 128];
        for v in 0..128 {
            for u in 0..128 {
                let (fu, fv) = (u as f32, v as f32);
                depth_m[v * 128 + u] =
                    2.0 + 0.003 * fu + 0.002 * fv + 0.0004 * (fu - 64.0) * (fu - 64.0) / 64.0;
            }
        }
        let dm = DepthMap {
            width: 128,
            height: 128,
            depth_m,
            confidence: None,
        };
        let mut vo = RgbdVoTracker::new(intr);
        let _ = vo.track(&dm);
        let pose = vo.track(&dm); // identical frame -> no motion
        assert!(
            pose.translation.length() < 0.02,
            "static-scene drift too high: {}",
            pose.translation.length()
        );
    }

    #[test]
    fn lk_recovers_known_shift() {
        let (w, h) = (160usize, 120usize);
        let mut prev = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // 2D texture (gradients in both axes) for observability.
                prev[y * w + x] =
                    (0.5 + 0.5 * (x as f32 * 0.3).sin() * (y as f32 * 0.27).cos()).clamp(0.0, 1.0);
            }
        }
        let (kx, ky) = (3.0f32, -2.0f32);
        let mut cur = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                cur[y * w + x] = sample_bilinear(&prev, w, h, x as f32 - kx, y as f32 - ky);
            }
        }
        let (dx, dy) = estimate_translation(&prev, &cur, w, h);
        assert!(
            (dx - kx).abs() < 0.4 && (dy - ky).abs() < 0.4,
            "recovered ({dx}, {dy}), expected ({kx}, {ky})"
        );
    }

    #[test]
    fn lk_rigid_recovers_rotation_and_shift() {
        let (w, h) = (160usize, 120usize);
        let mut prev = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                prev[y * w + x] =
                    (0.5 + 0.5 * (x as f32 * 0.21).sin() * (y as f32 * 0.19).cos()).clamp(0.0, 1.0);
            }
        }
        let c = Vec2::new((w as f32 - 1.0) * 0.5, (h as f32 - 1.0) * 0.5);
        let (theta_k, tx_k, ty_k) = (0.06f32, 4.0f32, -3.0f32);
        let m = Affine2::from_translation(c + Vec2::new(tx_k, ty_k))
            * Affine2::from_angle(theta_k)
            * Affine2::from_translation(-c);
        let mut cur = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = m.transform_point2(Vec2::new(x as f32, y as f32));
                cur[y * w + x] = sample_bilinear(&prev, w, h, p.x, p.y);
            }
        }
        let est = estimate_rigid2d(&prev, &cur, w, h);
        for &(px, py) in &[(20.0f32, 20.0f32), (140.0, 30.0), (80.0, 100.0)] {
            let a = m.transform_point2(Vec2::new(px, py));
            let b = est.transform_point2(Vec2::new(px, py));
            assert!((a - b).length() < 1.2, "rigid mismatch at ({px},{py})");
        }
    }
}
