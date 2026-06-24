//! Pose estimation.
//!
//! The real stage is depth-assisted RGB-D-style direct visual odometry
//! (robust photometric + geometric residual minimized on the SE3 Lie algebra
//! via Gauss-Newton/LM), with a per-keyframe affine depth-alignment step
//! *before* fusion and a keyframe pose-graph with loop closure. That is the
//! biggest greenfield line item (no mature Rust SLAM crate exists) and lands
//! as its own milestone (M1/M2) with an offline accuracy harness.
//!
//! M0 uses [`IdentityPose`]: a trivial estimator so the spine runs.

use ge_backend_trait::{DepthMap, Frame, Pose, PoseEstimator};

/// A trivial pose estimator that always reports the identity (camera fixed at
/// the origin). Lets fusion run on a static scene during M0.
pub struct IdentityPose;

impl PoseEstimator for IdentityPose {
    fn track(&mut self, _frame: &Frame, _depth: &DepthMap) -> anyhow::Result<Pose> {
        Ok(Pose::IDENTITY)
    }
}
