//! Shared traits and core data types that define the seams between Gods Eye
//! pipeline stages.
//!
//! Backends (depth models, pose estimators, capture sources, compute backends)
//! sit behind these traits so platform accelerators (CoreML/CUDA/DirectML) stay
//! optional behind a portable fallback. See `docs/design/ARCHITECTURE.md`.

use glam::{Affine3A, Vec3};

/// Pinhole camera intrinsics, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// Back-project a pixel `(u, v)` with metric `depth_m` (meters) to a
    /// camera-space 3D point. The camera looks down +Z.
    #[inline]
    pub fn unproject(&self, u: f32, v: f32, depth_m: f32) -> Vec3 {
        Vec3::new(
            (u - self.cx) / self.fx * depth_m,
            (v - self.cy) / self.fy * depth_m,
            depth_m,
        )
    }
}

/// An RGB8 camera frame.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Capture timestamp in nanoseconds (monotonic).
    pub timestamp_ns: u64,
    /// Tightly packed RGB8, `len == width * height * 3`.
    pub rgb: Vec<u8>,
}

impl Frame {
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// A metric depth map, in meters. A value `<= 0.0` or `NaN` is invalid.
#[derive(Clone)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    /// Row-major depth in meters, `len == width * height`.
    pub depth_m: Vec<f32>,
    /// Optional row-major confidence weights in `[0, 1]`, same length as
    /// `depth_m`. Missing confidence means every valid depth has weight 1.
    pub confidence: Option<Vec<f32>>,
}

/// A 6-DoF camera pose expressed as a camera-to-world transform.
pub type Pose = Affine3A;

/// A source of frames: webcam, video file, or (later) a drone stream.
pub trait CaptureSource: Send {
    /// Intrinsics for the produced frames, if known/calibrated.
    fn intrinsics(&self) -> Option<Intrinsics>;
    /// Pull the next frame, or `Ok(None)` at end of stream.
    fn next_frame(&mut self) -> anyhow::Result<Option<Frame>>;
}

/// A monocular metric-depth backend (e.g. Depth Anything V2 via `ort`).
pub trait DepthBackend: Send {
    fn name(&self) -> &str;
    fn infer(&mut self, frame: &Frame) -> anyhow::Result<DepthMap>;
}

/// A 6-DoF pose estimator (depth-assisted visual odometry).
pub trait PoseEstimator: Send {
    /// Track a new `frame`+`depth`, returning the camera-to-world pose.
    fn track(&mut self, frame: &Frame, depth: &DepthMap) -> anyhow::Result<Pose>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unproject_principal_point_is_on_axis() {
        let k = Intrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        };
        let p = k.unproject(320.0, 240.0, 2.0);
        assert!(p.x.abs() < 1e-6 && p.y.abs() < 1e-6);
        assert!((p.z - 2.0).abs() < 1e-6);
    }
}
