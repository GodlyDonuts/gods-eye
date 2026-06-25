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
use glam::{Affine3A, Quat, Vec3};

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
}
