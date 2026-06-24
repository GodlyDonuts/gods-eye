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
        })
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
    }

    impl OrtDepth {
        pub fn new(model_path: &str, accel: Accel) -> Result<Self> {
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
            Ok(Self { session, accel })
        }

        pub fn accel(&self) -> Accel {
            self.accel
        }

        /// Run inference on a pre-normalized NCHW (`1×3×h×w`) f32 buffer.
        /// Returns `(output_element_count, output_min)` for a sanity check.
        pub fn run_raw(&mut self, h: usize, w: usize, chw: &[f32]) -> Result<(usize, f32)> {
            anyhow::ensure!(chw.len() == 3 * h * w, "input buffer is not 3*h*w");
            let tensor = Tensor::from_array(([1usize, 3, h, w], chw.to_vec().into_boxed_slice()))
                .map_err(oerr)?;
            let outputs = self.session.run(ort::inputs![tensor]).map_err(oerr)?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(oerr)?;
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            Ok((data.len(), min))
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
