//! Metric-depth backends.
//!
//! The shipping backend is Depth Anything V2 Metric-Small exported to ONNX and
//! run via `ort` (ONNX Runtime) with CoreML/CUDA/DirectML execution providers
//! and a CPU floor, all behind the [`DepthBackend`] trait.
//!
//! [`ConstantDepth`] (always available) lets the fusion/mesh stages be
//! exercised without a model download. [`OrtDepth`] and [`bench`] (behind the
//! `onnx` feature) are the real ONNX backend and the M0 latency spike.

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
            confidence: None,
        })
    }
}

const DAV2_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DAV2_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Convert an RGB8 frame into the normalized NCHW tensor expected by DAv2.
///
/// DAv2 ViT inputs should be square and divisible by the 14-pixel patch size.
/// The implementation is allocation-conscious and deterministic: one output
/// allocation, bilinear RGB sampling, ImageNet normalization, channel-major
/// writes.
pub fn preprocess_rgb_to_chw(frame: &Frame, size: u32) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(size > 0, "input size must be non-zero");
    anyhow::ensure!(
        size.is_multiple_of(14),
        "input size must be divisible by 14"
    );
    anyhow::ensure!(
        frame.rgb.len() == frame.pixel_count() * 3,
        "frame RGB buffer has length {}, expected {}",
        frame.rgb.len(),
        frame.pixel_count() * 3
    );

    let out_w = size as usize;
    let out_h = size as usize;
    let mut chw = vec![0.0f32; 3 * out_w * out_h];

    let src_w = frame.width as usize;
    let src_h = frame.height as usize;
    anyhow::ensure!(src_w > 0 && src_h > 0, "frame dimensions must be non-zero");

    let scale_x = frame.width as f32 / size as f32;
    let scale_y = frame.height as f32 / size as f32;
    let plane = out_w * out_h;

    for y in 0..out_h {
        let src_y = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = src_y - y0 as f32;

        for x in 0..out_w {
            let src_x = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = src_x - x0 as f32;

            let dst = y * out_w + x;
            for c in 0..3 {
                let p00 = frame.rgb[(y0 * src_w + x0) * 3 + c] as f32;
                let p10 = frame.rgb[(y0 * src_w + x1) * 3 + c] as f32;
                let p01 = frame.rgb[(y1 * src_w + x0) * 3 + c] as f32;
                let p11 = frame.rgb[(y1 * src_w + x1) * 3 + c] as f32;
                let top = p00 + (p10 - p00) * wx;
                let bottom = p01 + (p11 - p01) * wx;
                let rgb01 = (top + (bottom - top) * wy) / 255.0;
                chw[c * plane + dst] = (rgb01 - DAV2_MEAN[c]) / DAV2_STD[c];
            }
        }
    }

    Ok(chw)
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn resize_depth_bilinear(
    depth: &[f32],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        src_width > 0 && src_height > 0,
        "source depth dimensions must be non-zero"
    );
    anyhow::ensure!(
        dst_width > 0 && dst_height > 0,
        "target depth dimensions must be non-zero"
    );
    anyhow::ensure!(
        depth.len() == (src_width as usize) * (src_height as usize),
        "depth buffer has length {}, expected {}",
        depth.len(),
        (src_width as usize) * (src_height as usize)
    );

    if src_width == dst_width && src_height == dst_height {
        return Ok(depth.to_vec());
    }

    let src_w = src_width as usize;
    let src_h = src_height as usize;
    let dst_w = dst_width as usize;
    let dst_h = dst_height as usize;
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let mut out = vec![0.0f32; dst_w * dst_h];

    for y in 0..dst_h {
        let src_y = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = src_y - y0 as f32;

        for x in 0..dst_w {
            let src_x = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = src_x - x0 as f32;

            let d00 = depth[y0 * src_w + x0];
            let d10 = depth[y0 * src_w + x1];
            let d01 = depth[y1 * src_w + x0];
            let d11 = depth[y1 * src_w + x1];
            let top = d00 + (d10 - d00) * wx;
            let bottom = d01 + (d11 - d01) * wx;
            out[y * dst_w + x] = top + (bottom - top) * wy;
        }
    }

    Ok(out)
}

/// Estimate per-pixel depth confidence from validity, distance, and local
/// depth discontinuities.
///
/// This is intentionally model-agnostic: invalid values get zero weight, far
/// values are down-weighted, and sharp local depth jumps are treated as likely
/// object boundaries where monocular depth is least stable.
pub fn estimate_confidence(depth: &DepthMap, far_m: f32) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(far_m > 0.0, "far_m must be positive");
    anyhow::ensure!(
        depth.depth_m.len() == (depth.width as usize) * (depth.height as usize),
        "depth buffer has length {}, expected {}",
        depth.depth_m.len(),
        (depth.width as usize) * (depth.height as usize)
    );

    let w = depth.width as usize;
    let h = depth.height as usize;
    let mut confidence = vec![0.0f32; depth.depth_m.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let d = depth.depth_m[i];
            if d <= 0.0 || !d.is_finite() {
                continue;
            }

            let mut max_jump = 0.0f32;
            if x > 0 {
                max_jump = max_jump.max(neighbor_jump(d, depth.depth_m[i - 1]));
            }
            if x + 1 < w {
                max_jump = max_jump.max(neighbor_jump(d, depth.depth_m[i + 1]));
            }
            if y > 0 {
                max_jump = max_jump.max(neighbor_jump(d, depth.depth_m[i - w]));
            }
            if y + 1 < h {
                max_jump = max_jump.max(neighbor_jump(d, depth.depth_m[i + w]));
            }

            let range_w = (1.0 - (d / far_m).powi(2)).clamp(0.0, 1.0);
            let edge_scale = 0.04 * d + 0.02;
            let edge_w = (1.0 - max_jump / edge_scale).clamp(0.0, 1.0);
            confidence[i] = range_w * edge_w;
        }
    }
    Ok(confidence)
}

fn neighbor_jump(center: f32, neighbor: f32) -> f32 {
    if neighbor <= 0.0 || !neighbor.is_finite() {
        f32::INFINITY
    } else {
        (center - neighbor).abs()
    }
}

#[cfg(feature = "onnx")]
mod onnx_backend {
    use anyhow::Result;
    use ort::session::Session;
    use ort::value::Tensor;
    use std::time::Instant;

    /// Convert any ort error (some carry non-`Send` typed recovery values) into
    /// an anyhow error via its `Display` impl, so `?` works against
    /// `anyhow::Result`.
    fn oerr<E: std::fmt::Display>(e: E) -> anyhow::Error {
        anyhow::anyhow!("{e}")
    }

    /// Which execution provider to request.
    #[derive(Clone, Copy, Debug)]
    pub enum Accel {
        /// Portable CPU provider (always available).
        Cpu,
        /// Apple CoreML (ANE/GPU) — macOS only; falls back to CPU otherwise.
        CoreMl,
    }

    impl Accel {
        pub fn label(self) -> &'static str {
            match self {
                Accel::Cpu => "cpu",
                Accel::CoreMl => "coreml",
            }
        }
    }

    /// Depth Anything V2 (ViT-S) ONNX backend driven by ONNX Runtime.
    ///
    /// Note: the public `onnx-community` checkpoint is the *relative* DAv2-Small;
    /// it shares the exact ViT-S/DPT backbone with the metric variant, so its
    /// inference latency is a faithful proxy. A metric-checkpoint ONNX export is
    /// a separate offline-tooling task tracked for correctness.
    pub struct OrtDepth {
        session: Session,
        accel: Accel,
        input_size: u32,
    }

    impl OrtDepth {
        pub fn new(model_path: &str, accel: Accel) -> Result<Self> {
            Self::new_with_size(model_path, accel, 518)
        }

        pub fn new_with_size(model_path: &str, accel: Accel, input_size: u32) -> Result<Self> {
            anyhow::ensure!(input_size > 0, "input size must be non-zero");
            anyhow::ensure!(input_size % 14 == 0, "input size must be divisible by 14");
            let mut builder = Session::builder().map_err(oerr)?;
            if let Accel::CoreMl = accel {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::CoreML::default().build()])
                        .map_err(oerr)?;
                }
            }
            let session = builder.commit_from_file(model_path).map_err(oerr)?;
            Ok(Self {
                session,
                accel,
                input_size,
            })
        }

        pub fn accel(&self) -> Accel {
            self.accel
        }

        pub fn input_size(&self) -> u32 {
            self.input_size
        }

        /// Run inference on a pre-normalized NCHW (`1×3×h×w`) f32 buffer.
        pub fn run_raw_depth(&mut self, h: usize, w: usize, chw: &[f32]) -> Result<Vec<f32>> {
            anyhow::ensure!(chw.len() == 3 * h * w, "input buffer is not 3*h*w");
            let tensor = Tensor::from_array(([1usize, 3, h, w], chw.to_vec().into_boxed_slice()))
                .map_err(oerr)?;
            let outputs = self.session.run(ort::inputs![tensor]).map_err(oerr)?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(oerr)?;
            Ok(data.to_vec())
        }

        /// Run inference on a pre-normalized NCHW (`1×3×h×w`) f32 buffer.
        /// Returns `(output_element_count, output_min)` for a sanity check.
        pub fn run_raw(&mut self, h: usize, w: usize, chw: &[f32]) -> Result<(usize, f32)> {
            let data = self.run_raw_depth(h, w, chw)?;
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            Ok((data.len(), min))
        }
    }

    impl ge_backend_trait::DepthBackend for OrtDepth {
        fn name(&self) -> &str {
            match self.accel {
                Accel::Cpu => "ort-cpu",
                Accel::CoreMl => "ort-coreml",
            }
        }

        fn infer(
            &mut self,
            frame: &ge_backend_trait::Frame,
        ) -> anyhow::Result<ge_backend_trait::DepthMap> {
            let input = crate::preprocess_rgb_to_chw(frame, self.input_size)?;
            let raw =
                self.run_raw_depth(self.input_size as usize, self.input_size as usize, &input)?;
            let input_pixels = (self.input_size as usize) * (self.input_size as usize);
            anyhow::ensure!(
                raw.len() == input_pixels,
                "model output has {} elements, expected {} for a {}x{} depth map",
                raw.len(),
                input_pixels,
                self.input_size,
                self.input_size
            );
            let depth_m = crate::resize_depth_bilinear(
                &raw,
                self.input_size,
                self.input_size,
                frame.width,
                frame.height,
            )?;
            Ok(ge_backend_trait::DepthMap {
                width: frame.width,
                height: frame.height,
                confidence: Some(crate::estimate_confidence(
                    &ge_backend_trait::DepthMap {
                        width: frame.width,
                        height: frame.height,
                        depth_m: depth_m.clone(),
                        confidence: None,
                    },
                    20.0,
                )?),
                depth_m,
            })
        }
    }

    /// Per-resolution latency measurement.
    #[derive(Debug)]
    pub struct BenchResult {
        pub size: u32,
        pub ok: bool,
        pub note: String,
        pub iters: usize,
        pub min_ms: f64,
        pub median_ms: f64,
        pub p95_ms: f64,
        pub out_len: usize,
        pub out_min: f32,
    }

    /// Benchmark depth inference at several square input sizes.
    ///
    /// Input is synthetic (constant-filled NCHW); inference cost is
    /// data-independent so this faithfully measures wall-clock latency. A size
    /// the model rejects (fixed-shape export) is recorded as `ok = false`
    /// rather than aborting the sweep.
    pub fn bench(
        model_path: &str,
        accel: Accel,
        sizes: &[u32],
        iters: usize,
        warmup: usize,
    ) -> Result<Vec<BenchResult>> {
        let mut depth = OrtDepth::new(model_path, accel)?;
        let mut results = Vec::new();
        for &s in sizes {
            let (h, w) = (s as usize, s as usize);
            let input = vec![0.5f32; 3 * h * w];

            // First run doubles as a shape-support probe + warmup.
            let sample = match depth.run_raw(h, w, &input) {
                Ok(v) => v,
                Err(e) => {
                    results.push(BenchResult {
                        size: s,
                        ok: false,
                        note: format!("{e}"),
                        iters: 0,
                        min_ms: 0.0,
                        median_ms: 0.0,
                        p95_ms: 0.0,
                        out_len: 0,
                        out_min: 0.0,
                    });
                    continue;
                }
            };
            for _ in 1..warmup {
                depth.run_raw(h, w, &input)?;
            }

            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                depth.run_raw(h, w, &input)?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95_idx = (((times.len() as f64) * 0.95) as usize).min(times.len() - 1);
            results.push(BenchResult {
                size: s,
                ok: true,
                note: String::new(),
                iters,
                min_ms: times[0],
                median_ms: times[times.len() / 2],
                p95_ms: times[p95_idx],
                out_len: sample.0,
                out_min: sample.1,
            });
        }
        Ok(results)
    }
}

#[cfg(feature = "onnx")]
pub use onnx_backend::{bench, Accel, BenchResult, OrtDepth};

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, rgb: Vec<u8>) -> Frame {
        Frame {
            width,
            height,
            timestamp_ns: 0,
            rgb,
        }
    }

    #[test]
    fn preprocess_rejects_bad_frame_buffer() {
        let f = frame(2, 2, vec![0; 11]);
        let err = preprocess_rgb_to_chw(&f, 14).unwrap_err().to_string();
        assert!(err.contains("RGB buffer"));
    }

    #[test]
    fn preprocess_writes_channel_major_normalized_tensor() {
        let f = frame(1, 1, vec![255, 0, 128]);
        let chw = preprocess_rgb_to_chw(&f, 14).unwrap();
        let plane = 14 * 14;
        assert_eq!(chw.len(), 3 * plane);
        assert!((chw[0] - ((1.0 - DAV2_MEAN[0]) / DAV2_STD[0])).abs() < 1e-6);
        assert!((chw[plane] - ((0.0 - DAV2_MEAN[1]) / DAV2_STD[1])).abs() < 1e-6);
        let b = (128.0 / 255.0 - DAV2_MEAN[2]) / DAV2_STD[2];
        assert!((chw[2 * plane] - b).abs() < 1e-6);
    }

    #[test]
    fn depth_resize_preserves_target_shape() {
        let resized = resize_depth_bilinear(&[1.0, 2.0, 3.0, 4.0], 2, 2, 4, 3).unwrap();
        assert_eq!(resized.len(), 12);
        assert_eq!(resized[0], 1.0);
        assert_eq!(resized[11], 4.0);
    }

    #[test]
    fn confidence_rejects_invalid_and_downweights_edges() {
        let depth = DepthMap {
            width: 3,
            height: 1,
            depth_m: vec![2.0, 2.0, 4.0],
            confidence: None,
        };
        let c = estimate_confidence(&depth, 20.0).unwrap();
        assert!(c[0] > 0.9, "flat valid depth should stay high confidence");
        assert!(c[1] < c[0], "depth discontinuity should lower confidence");
        assert!(c[2] < c[0], "edge neighbor should lower confidence");

        let invalid = DepthMap {
            width: 1,
            height: 1,
            depth_m: vec![f32::NAN],
            confidence: None,
        };
        assert_eq!(estimate_confidence(&invalid, 20.0).unwrap()[0], 0.0);
    }
}
