//! Pipeline orchestration for Gods Eye.
//!
//! The full design is one `std::thread` per stage joined by bounded
//! `crossbeam` channels (backpressure + triple-buffering) with object-pooled
//! buffers recycled through return channels. This M0 skeleton holds the config
//! and a synchronous single-thread runner used for smoke tests; the threaded
//! frame-graph lands incrementally. See `docs/design/ARCHITECTURE.md`.

use ge_backend_trait::{CaptureSource, DepthBackend};

/// Pipeline configuration.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Bounded channel capacity between stages (backpressure window).
    pub channel_capacity: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 3,
        }
    }
}

/// Run a minimal synchronous capture -> depth loop, invoking `on_depth` for
/// each frame. Returns the number of frames processed.
///
/// This is the M0 smoke-test spine; the production runner is multi-threaded.
pub fn run_sync<C, D, F>(source: &mut C, depth: &mut D, mut on_depth: F) -> anyhow::Result<usize>
where
    C: CaptureSource,
    D: DepthBackend,
    F: FnMut(&ge_backend_trait::Frame, &ge_backend_trait::DepthMap),
{
    let mut n = 0;
    while let Some(frame) = source.next_frame()? {
        let dm = depth.infer(&frame)?;
        on_depth(&frame, &dm);
        n += 1;
    }
    Ok(n)
}
