//! Synthetic ground-truth harness for offline VO validation.
//!
//! The live tracker's projective data-association path can only be trusted if we
//! measure it against a **known** trajectory. This module renders metric depth
//! of a known room from a known camera path, runs the tracker, and reports drift
//! as a number — deterministically, in CI, with no external data or COLMAP.
//!
//! It can also inject per-frame depth "breathing" (an affine scale/shift on all
//! depths) to reproduce the learned-depth inconsistency that is the project's
//! central correctness risk (roadmap risk #1), so the cost of that risk — and
//! later, the value of the affine-alignment mitigation — is a measured quantity.

use ge_backend_trait::{DepthMap, Intrinsics};
use glam::{Affine3A, Quat, Vec3};

use crate::RgbdVoTracker;

/// An infinite plane `{ p : normal·p = offset }` with unit `normal`.
#[derive(Clone, Copy, Debug)]
pub struct SimPlane {
    pub normal: Vec3,
    pub offset: f32,
}

impl SimPlane {
    fn new(normal: Vec3, offset: f32) -> Self {
        Self {
            normal: normal.normalize(),
            offset,
        }
    }
}

/// A closed box room: floor, ceiling, front wall, two side walls. The camera
/// starts near the centre looking toward the front wall (+z); every pose in
/// [`loop_trajectory`] keeps several non-parallel planes in view, giving full
/// point-to-plane observability.
pub fn default_room() -> Vec<SimPlane> {
    vec![
        SimPlane::new(Vec3::new(0.0, 1.0, 0.0), 0.8),  // floor  (below, y=+0.8)
        SimPlane::new(Vec3::new(0.0, 1.0, 0.0), -1.2), // ceiling(above, y=-1.2)
        SimPlane::new(Vec3::new(0.0, 0.0, 1.0), 3.0),  // front wall z=3.0
        SimPlane::new(Vec3::new(1.0, 0.0, 0.0), -1.6), // left wall  x=-1.6
        SimPlane::new(Vec3::new(1.0, 0.0, 0.0), 1.6),  // right wall x=+1.6
    ]
}

/// Deterministic [-1, 1] pseudo-noise (no `Math.random`/`Date` in scope).
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 9) as f32 / 8_388_608.0 - 1.0
}

/// Render the metric depth of `planes` seen from `world_from_cam`.
///
/// `noise_m` adds uniform per-pixel depth noise (metres). `scale`/`shift_m`
/// apply an affine transform to every depth (`d' = scale·d + shift_m`) to
/// simulate a frame's learned-depth breathing; pass `scale = 1.0, shift_m = 0.0`
/// for a perfect render.
pub fn render_depth(
    planes: &[SimPlane],
    intr: &Intrinsics,
    world_from_cam: &Affine3A,
    noise_m: f32,
    scale: f32,
    shift_m: f32,
    seed: &mut u32,
) -> DepthMap {
    let (w, h) = (intr.width as usize, intr.height as usize);
    let mut depth = vec![0.0f32; w * h];
    let o = world_from_cam.translation;
    let ow = Vec3::new(o.x, o.y, o.z);
    for v in 0..h {
        for u in 0..w {
            let x = (u as f32 - intr.cx) / intr.fx;
            let y = (v as f32 - intr.cy) / intr.fy;
            // Camera-frame ray dir (x,y,1); param along it IS the camera-z depth.
            let dir = world_from_cam.transform_vector3(Vec3::new(x, y, 1.0));
            let mut best = f32::MAX;
            for pl in planes {
                let denom = pl.normal.dot(dir);
                if denom.abs() < 1e-6 {
                    continue;
                }
                let s = (pl.offset - pl.normal.dot(ow)) / denom;
                if s > 0.05 && s < best {
                    best = s;
                }
            }
            if best < f32::MAX {
                let d = scale * best + shift_m + lcg(seed) * noise_m;
                depth[v * w + u] = d;
            }
        }
    }
    DepthMap {
        width: intr.width,
        height: intr.height,
        depth_m: depth,
        confidence: None,
    }
}

/// A smooth closed-loop trajectory of `n + 1` camera poses (`world_from_cam`):
/// a horizontal circle of radius `r` metres with a gentle yaw sweep, starting
/// **and** ending at the identity. Because the last pose equals the first, the
/// end-to-end drift is simply the final estimated translation magnitude.
pub fn loop_trajectory(n: usize, r: f32) -> Vec<Affine3A> {
    let mut poses = Vec::with_capacity(n + 1);
    let tau = std::f32::consts::TAU;
    for i in 0..=n {
        let phi = tau * i as f32 / n as f32;
        // Circle centred at (0,0,r): starts at origin, returns to it at phi=τ.
        let pos = Vec3::new(r * phi.sin(), 0.0, r * (1.0 - phi.cos()));
        // Yaw sweep that also returns to zero at the loop close.
        let yaw = 0.15 * phi.sin();
        poses.push(Affine3A::from_rotation_translation(
            Quat::from_rotation_y(yaw),
            pos,
        ));
    }
    poses
}

/// Drift metrics from running the tracker over a synthetic sequence.
#[derive(Clone, Copy, Debug)]
pub struct DriftReport {
    pub frames: usize,
    /// Max positional error over the trajectory (m).
    pub max_pos_err: f32,
    /// Positional error at the final frame (m).
    pub final_pos_err: f32,
    /// Max rotational error over the trajectory (deg).
    pub max_rot_err_deg: f32,
    /// Ground-truth path length (m).
    pub path_length: f32,
    /// `final_pos_err / path_length` — drift as a fraction of distance travelled.
    pub drift_ratio: f32,
}

/// Angle (radians) between two rotations.
fn rot_angle(a: &Affine3A, b: &Affine3A) -> f32 {
    let qa = Quat::from_mat3a(&a.matrix3).normalize();
    let qb = Quat::from_mat3a(&b.matrix3).normalize();
    qa.angle_between(qb)
}

/// Run the tracker over depth rendered along ground-truth poses `gt` and compare
/// the estimated trajectory against it. `gt[0]` should be the identity (the
/// tracker starts there). See [`render_depth`] for the noise/breathing params.
pub fn run_drift(
    gt: &[Affine3A],
    planes: &[SimPlane],
    intr: &Intrinsics,
    noise_m: f32,
    breathing: f32,
    mut seed: u32,
) -> DriftReport {
    let mut vo = RgbdVoTracker::new(*intr);
    let mut max_pos_err = 0.0f32;
    let mut max_rot_err = 0.0f32;
    let mut final_pos_err = 0.0f32;

    for (i, pose) in gt.iter().enumerate() {
        // Per-frame affine breathing: scale in [1-breathing, 1+breathing].
        let (scale, shift) = if breathing > 0.0 {
            (1.0 + breathing * lcg(&mut seed), breathing * 0.1 * lcg(&mut seed))
        } else {
            (1.0, 0.0)
        };
        let depth = render_depth(planes, intr, pose, noise_m, scale, shift, &mut seed);
        let est = vo.track(&depth);

        let pos_err = (est.translation - pose.translation).length();
        let rot_err = rot_angle(&est, pose);
        max_pos_err = max_pos_err.max(pos_err);
        max_rot_err = max_rot_err.max(rot_err);
        if i == gt.len() - 1 {
            final_pos_err = pos_err;
        }
    }

    let path_length: f32 = gt
        .windows(2)
        .map(|w| (w[1].translation - w[0].translation).length())
        .sum();
    DriftReport {
        frames: gt.len(),
        max_pos_err,
        final_pos_err,
        max_rot_err_deg: max_rot_err.to_degrees(),
        path_length,
        drift_ratio: if path_length > 1e-6 {
            final_pos_err / path_length
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intr() -> Intrinsics {
        Intrinsics {
            fx: 90.0,
            fy: 90.0,
            cx: 80.0,
            cy: 60.0,
            width: 160,
            height: 120,
        }
    }

    #[test]
    fn perfect_depth_tracks_the_loop() {
        let gt = loop_trajectory(60, 0.3);
        let report = run_drift(&gt, &default_room(), &intr(), 0.0, 0.0, 1);
        eprintln!("perfect: {report:?}");
        // With perfect depth the tracker should follow the loop closely.
        assert!(
            report.drift_ratio < 0.10,
            "drift ratio {:.3} too high (final {:.3} m over {:.3} m)",
            report.drift_ratio,
            report.final_pos_err,
            report.path_length
        );
        assert!(report.max_rot_err_deg < 5.0, "rot err {:.2}°", report.max_rot_err_deg);
    }

    #[test]
    fn noisy_depth_stays_bounded() {
        let gt = loop_trajectory(60, 0.3);
        let report = run_drift(&gt, &default_room(), &intr(), 0.01, 0.0, 7);
        eprintln!("noisy: {report:?}");
        assert!(report.drift_ratio < 0.25, "drift ratio {:.3}", report.drift_ratio);
    }

    #[test]
    fn breathing_is_worse_than_perfect() {
        // Documents risk #1: per-frame depth breathing degrades tracking. The
        // affine-alignment mitigation (L2b) should later shrink this gap.
        let gt = loop_trajectory(60, 0.3);
        let clean = run_drift(&gt, &default_room(), &intr(), 0.0, 0.0, 3);
        let breathing = run_drift(&gt, &default_room(), &intr(), 0.0, 0.05, 3);
        eprintln!("clean {:?}\nbreathing {:?}", clean, breathing);
        assert!(
            breathing.final_pos_err >= clean.final_pos_err,
            "breathing ({:.3}) should not beat clean ({:.3})",
            breathing.final_pos_err,
            clean.final_pos_err
        );
    }
}
