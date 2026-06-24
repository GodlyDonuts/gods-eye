//! Capture sources.
//!
//! M0 ships synthetic and image-file sources so the pipeline can run
//! end-to-end with no camera or video-decode dependency. Real sources (nokhwa
//! webcam, video-file decode, drone H.264) land in later milestones behind the
//! same [`CaptureSource`] trait.

use ge_backend_trait::{CaptureSource, Frame, Intrinsics};
use std::path::Path;

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

/// A file-backed capture source that repeats one decoded RGB image.
///
/// This is the primary practical input until video decode/webcam capture lands:
/// it exercises the real image normalization and depth path while keeping tests
/// deterministic and platform-light.
pub struct ImageFileSource {
    width: u32,
    height: u32,
    remaining: u32,
    next_ts_ns: u64,
    rgb: Vec<u8>,
    intrinsics: Option<Intrinsics>,
}

impl ImageFileSource {
    pub fn open(path: impl AsRef<Path>, frames: u32) -> anyhow::Result<Self> {
        let img = image::ImageReader::open(path.as_ref())?
            .with_guessed_format()?
            .decode()?
            .into_rgb8();
        let (width, height) = img.dimensions();
        anyhow::ensure!(width > 0 && height > 0, "image dimensions must be non-zero");
        Ok(Self {
            width,
            height,
            remaining: frames,
            next_ts_ns: 0,
            rgb: img.into_raw(),
            intrinsics: None,
        })
    }

    /// Attach intrinsics reported to the pipeline.
    pub fn with_intrinsics(mut self, k: Intrinsics) -> Self {
        self.intrinsics = Some(k);
        self
    }
}

impl CaptureSource for ImageFileSource {
    fn intrinsics(&self) -> Option<Intrinsics> {
        self.intrinsics
    }

    fn next_frame(&mut self) -> anyhow::Result<Option<Frame>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let ts = self.next_ts_ns;
        self.next_ts_ns += 33_333_333; // ~30 fps spacing
        Ok(Some(Frame {
            width: self.width,
            height: self.height,
            timestamp_ns: ts,
            rgb: self.rgb.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    #[test]
    fn image_file_source_decodes_and_repeats_rgb_frames() {
        let path =
            std::env::temp_dir().join(format!("gods-eye-image-source-{}.png", std::process::id()));

        let mut png = std::fs::File::create(&path).unwrap();
        let img = image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        png.flush().unwrap();

        let mut source = ImageFileSource::open(&path, 2).unwrap();
        let first = source.next_frame().unwrap().unwrap();
        let second = source.next_frame().unwrap().unwrap();
        assert_eq!(first.width, 2);
        assert_eq!(first.height, 1);
        assert_eq!(first.rgb, vec![255, 0, 0, 0, 255, 0]);
        assert_eq!(second.rgb, first.rgb);
        assert!(source.next_frame().unwrap().is_none());

        let _ = std::fs::remove_file(path);
    }
}
