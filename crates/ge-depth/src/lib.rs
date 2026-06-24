//! Metric-depth backends.
//!
//! The shipping backend is Depth Anything V2 Metric-Small exported to ONNX and
//! run via `ort` (ONNX Runtime) with CoreML/CUDA/DirectML execution providers
//! and a CPU floor, all behind the [`DepthBackend`] trait. That backend (and
//! the week-1 latency spike that measures it on the actual M1) lands next.
//!
//! For now [`ConstantDepth`] lets the fusion/mesh stages be exercised without
//! a model download.

use ge_backend_trait::{DepthBackend, DepthMap, Frame};

/// A placeholder backend that returns a constant metric depth for every pixel.
pub struct ConstantDepth {
    pub depth_m: f32,
}

impl ConstantDepth {
    pub fn new(depth_m: f32) -> Self {
        Self { depth_m }
    }
}

impl DepthBackend for ConstantDepth {
    fn name(&self) -> &str {
        "constant"
    }

    fn infer(&mut self, frame: &Frame) -> anyhow::Result<DepthMap> {
        Ok(DepthMap {
            width: frame.width,
            height: frame.height,
            depth_m: vec![self.depth_m; frame.pixel_count()],
        })
    }
}
