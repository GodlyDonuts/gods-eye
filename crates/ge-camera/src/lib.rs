//! Capture sources.
//!
//! M0 ships a synthetic source so the pipeline can run end-to-end with no
//! camera or video-decode dependency. Real sources (nokhwa webcam, video-file
//! decode, drone H.264) land in later milestones behind the same
//! [`CaptureSource`] trait.

use ge_backend_trait::{CaptureSource, Frame, Intrinsics};

/// A synthetic capture source that emits a fixed number of solid-color frames.
/// Useful for pipeline smoke tests before real capture lands.
pub struct SolidColorSource {
    width: u32,
    height: u32,
    remaining: u32,
    next_ts_ns: u64,
    color: [u8; 3],
    intrinsics: Option<Intrinsics>,
}

impl SolidColorSource {
    pub fn new(width: u32, height: u32, frames: u32, color: [u8; 3]) -> Self {
        Self {
            width,
            height,
            remaining: frames,
            next_ts_ns: 0,
            color,
            intrinsics: None,
        }
    }

    /// Attach intrinsics reported to the pipeline.
    pub fn with_intrinsics(mut self, k: Intrinsics) -> Self {
        self.intrinsics = Some(k);
        self
    }
}

impl CaptureSource for SolidColorSource {
    fn intrinsics(&self) -> Option<Intrinsics> {
        self.intrinsics
    }

    fn next_frame(&mut self) -> anyhow::Result<Option<Frame>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for _ in 0..(self.width * self.height) {
            rgb.extend_from_slice(&self.color);
        }
        let ts = self.next_ts_ns;
        self.next_ts_ns += 33_333_333; // ~30 fps spacing
        Ok(Some(Frame {
            width: self.width,
            height: self.height,
            timestamp_ns: ts,
            rgb,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_exactly_n_frames() {
        let mut s = SolidColorSource::new(4, 4, 3, [10, 20, 30]);
        let mut count = 0;
        while let Some(f) = s.next_frame().unwrap() {
            assert_eq!(f.rgb.len(), f.pixel_count() * 3);
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
