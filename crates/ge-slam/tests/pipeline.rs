//! End-to-end pipeline test: the L2 "walk around" promise.
//!
//! Composes the real live-pipeline stages — synthetic metric depth → 6-DoF
//! tracker → plane detection → world plane registry — over a moving camera
//! trajectory, and asserts the fused world planes converge to the room's true
//! geometry. If pose tracking were wrong (e.g. rotation-only, ignoring the
//! camera's translation), observations from different viewpoints would fuse into
//! smeared or mis-placed planes and this would fail. Correct 6-DoF tracking is
//! what makes the world "stay put" as the camera moves.

use ge_backend_trait::Intrinsics;
use ge_prim::{detect_planes, DetectParams, RegistryParams, WorldPlaneRegistry};
use ge_slam::sim::{default_room, loop_trajectory, render_depth};
use ge_slam::RgbdVoTracker;

#[test]
fn planes_stay_put_while_walking() {
    let intr = Intrinsics {
        fx: 90.0,
        fy: 90.0,
        cx: 80.0,
        cy: 60.0,
        width: 160,
        height: 120,
    };
    let room = default_room();
    // Two laps so planes are observed from many viewpoints and get confirmed.
    let gt = loop_trajectory(120, 0.3);

    let det = DetectParams {
        cell: 10,
        min_cell_points: 40,
        sigma_k: 0.02,
        jump_ratio: 0.5,
        normal_cos: 0.90,
        offset_tol: 0.15,
        min_cells: 3,
        min_depth: 0.2,
        max_depth: 6.0,
    };
    let mut reg = WorldPlaneRegistry::new(RegistryParams::default());
    let mut vo = RgbdVoTracker::new(intr);
    let mut seed = 11u32;

    for pose in &gt {
        let depth = render_depth(&room, &intr, pose, 0.0, 1.0, 0.0, &mut seed);
        let cam_to_world = vo.track(&depth);
        let segs = detect_planes(&depth, &intr, &det);
        reg.observe(&segs, &cam_to_world);
    }

    let confirmed = reg.confirmed_planes();
    assert!(
        confirmed.len() >= 3,
        "expected the room's major planes, got {}",
        confirmed.len()
    );

    // The floor and the front wall must both be recovered near their true world
    // offsets — only possible if the camera's motion was tracked correctly.
    let matches = |truth_n: glam::Vec3, truth_off: f32, n_tol: f32, off_tol: f32| {
        confirmed.iter().any(|(p, _)| {
            p.normal.dot(truth_n).abs() > n_tol
                && (p.offset.abs() - truth_off.abs()).abs() < off_tol
        })
    };
    assert!(
        matches(glam::Vec3::Y, 0.8, 0.97, 0.2),
        "floor not recovered: {:?}",
        confirmed.iter().map(|(p, _)| (p.normal, p.offset)).collect::<Vec<_>>()
    );
    assert!(
        matches(glam::Vec3::Z, 3.0, 0.97, 0.3),
        "front wall not recovered: {:?}",
        confirmed.iter().map(|(p, _)| (p.normal, p.offset)).collect::<Vec<_>>()
    );

    // Tracking stayed near the true loop (both laps close near the origin).
    let end = vo.world_from_cam().translation.length();
    assert!(end < 0.15, "camera drifted {end:.3} m from the loop start");
}
