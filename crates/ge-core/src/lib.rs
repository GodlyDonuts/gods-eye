//! Pipeline orchestration for Gods Eye.
//!
//! The full design is one `std::thread` per stage joined by bounded
//! `crossbeam` channels (backpressure + triple-buffering) with object-pooled
//! buffers recycled through return channels. This M0 skeleton holds the config
//! and a synchronous single-thread runner used for smoke tests; the threaded
//! frame-graph lands incrementally. See `docs/design/ARCHITECTURE.md`.

use ge_backend_trait::{
    CaptureSource, DepthBackend, DepthMap, Frame, Intrinsics, Pose, PoseEstimator,
};
use ge_mesh::Mesh;

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

/// Dense CPU fusion settings used by the current headless M0 producer.
#[derive(Clone, Debug)]
pub struct FusionConfig {
    /// Dense voxel edge length in metres.
    pub voxel_size_m: f32,
    /// TSDF truncation distance in metres.
    pub truncation_m: f32,
    /// Near bound of the fixed M0 volume in metres.
    pub z_min_m: f32,
    /// Far bound of the fixed M0 volume in metres.
    pub z_max_m: f32,
    /// Frustum side margin in metres.
    pub margin_m: f32,
    /// Emit a mesh every N integrated frames.
    pub mesh_every_n: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.05,
            truncation_m: 0.20,
            z_min_m: 1.2,
            z_max_m: 2.9,
            margin_m: 0.15,
            mesh_every_n: 1,
        }
    }
}

/// A captured frame with stable metadata for downstream consumers.
#[derive(Clone)]
pub struct FramePacket {
    pub frame_index: u64,
    pub timestamp_ns: u64,
    pub intrinsics: Intrinsics,
    pub frame: Frame,
}

/// A depth result aligned to a [`FramePacket`].
#[derive(Clone)]
pub struct DepthPacket {
    pub frame_index: u64,
    pub timestamp_ns: u64,
    pub depth: DepthMap,
}

/// A pose result aligned to a [`FramePacket`].
#[derive(Clone)]
pub struct PosePacket {
    pub frame_index: u64,
    pub timestamp_ns: u64,
    pub cam_to_world: Pose,
}

/// A mesh snapshot emitted after integrating one or more frames.
#[derive(Clone)]
pub struct MeshPacket {
    pub frame_index: u64,
    pub timestamp_ns: u64,
    pub mesh: Mesh,
}

/// Viewer-agnostic callbacks for the producer side of the pipeline.
///
/// UI code can implement this trait to receive the raw camera stream, depth
/// results, poses, and mesh snapshots without depending on the runner internals.
pub trait PipelineSink {
    fn on_frame(&mut self, _packet: &FramePacket) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_depth(&mut self, _frame: &FramePacket, _depth: &DepthPacket) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_pose(&mut self, _frame: &FramePacket, _pose: &PosePacket) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mesh(&mut self, _mesh: &MeshPacket) -> anyhow::Result<()> {
        Ok(())
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

/// Run capture -> depth -> identity pose -> TSDF fusion -> mesh extraction.
///
/// This is still a synchronous M0 runner, but its packets are the stable
/// producer contract for live viewers and future threaded stages.
pub fn run_fusion_sync<C, D, S>(
    source: &mut C,
    depth: &mut D,
    config: FusionConfig,
    sink: &mut S,
) -> anyhow::Result<usize>
where
    C: CaptureSource,
    D: DepthBackend,
    S: PipelineSink,
{
    let mut pose = IdentityPose;
    run_fusion_with_pose_sync(source, depth, &mut pose, config, sink)
}

/// Run capture -> depth -> pose estimation -> TSDF fusion -> mesh extraction.
///
/// This is the viewer-facing producer path. M0 callers can pass
/// [`IdentityPose`]; M1 can replace it with direct visual odometry without
/// changing the packet contract consumed by viewers.
pub fn run_fusion_with_pose_sync<C, D, P, S>(
    source: &mut C,
    depth: &mut D,
    pose: &mut P,
    config: FusionConfig,
    sink: &mut S,
) -> anyhow::Result<usize>
where
    C: CaptureSource,
    D: DepthBackend,
    P: PoseEstimator,
    S: PipelineSink,
{
    anyhow::ensure!(config.voxel_size_m > 0.0, "voxel size must be positive");
    anyhow::ensure!(config.truncation_m > 0.0, "truncation must be positive");
    anyhow::ensure!(
        config.z_max_m > config.z_min_m,
        "z_max must be greater than z_min"
    );
    anyhow::ensure!(config.mesh_every_n > 0, "mesh_every_n must be non-zero");

    let source_intrinsics = source.intrinsics();
    let mut tsdf = None;
    let mut n = 0usize;

    while let Some(frame) = source.next_frame()? {
        let intrinsics = source_intrinsics
            .filter(|k| k.width == frame.width && k.height == frame.height)
            .unwrap_or_else(|| default_intrinsics(frame.width, frame.height));
        let frame_packet = FramePacket {
            frame_index: n as u64,
            timestamp_ns: frame.timestamp_ns,
            intrinsics,
            frame,
        };
        sink.on_frame(&frame_packet)?;

        let depth_map = depth.infer(&frame_packet.frame)?;
        validate_depth_shape(&frame_packet.frame, &depth_map)?;
        let depth_packet = DepthPacket {
            frame_index: frame_packet.frame_index,
            timestamp_ns: frame_packet.timestamp_ns,
            depth: depth_map,
        };
        sink.on_depth(&frame_packet, &depth_packet)?;

        let cam_to_world = pose.track(&frame_packet.frame, &depth_packet.depth)?;
        let pose_packet = PosePacket {
            frame_index: frame_packet.frame_index,
            timestamp_ns: frame_packet.timestamp_ns,
            cam_to_world,
        };
        sink.on_pose(&frame_packet, &pose_packet)?;

        let volume = tsdf.get_or_insert_with(|| make_tsdf(&frame_packet.intrinsics, &config));
        volume.integrate(
            &depth_packet.depth,
            &frame_packet.intrinsics,
            &pose_packet.cam_to_world,
        );

        n += 1;
        if n.is_multiple_of(config.mesh_every_n) {
            let mesh_packet = MeshPacket {
                frame_index: frame_packet.frame_index,
                timestamp_ns: frame_packet.timestamp_ns,
                mesh: volume.extract_mesh(),
            };
            sink.on_mesh(&mesh_packet)?;
        }
    }

    Ok(n)
}

/// A trivial pose estimator that always reports the camera at the world origin.
pub struct IdentityPose;

impl PoseEstimator for IdentityPose {
    fn track(&mut self, _frame: &Frame, _depth: &DepthMap) -> anyhow::Result<Pose> {
        Ok(Pose::IDENTITY)
    }
}

/// A conservative 60-degree pinhole fallback for sources without calibration.
pub fn default_intrinsics(width: u32, height: u32) -> Intrinsics {
    let side = width.min(height).max(1) as f32;
    let f = side / 2.0 / (60.0f32.to_radians() / 2.0).tan();
    Intrinsics {
        fx: f,
        fy: f,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
    }
}

fn validate_depth_shape(frame: &Frame, depth: &DepthMap) -> anyhow::Result<()> {
    anyhow::ensure!(
        depth.width == frame.width && depth.height == frame.height,
        "depth map is {}x{}, expected frame shape {}x{}",
        depth.width,
        depth.height,
        frame.width,
        frame.height
    );
    anyhow::ensure!(
        depth.depth_m.len() == frame.pixel_count(),
        "depth buffer has length {}, expected {}",
        depth.depth_m.len(),
        frame.pixel_count()
    );
    Ok(())
}

fn make_tsdf(intr: &Intrinsics, config: &FusionConfig) -> ge_fusion::Tsdf {
    let half_w = (intr.width as f32 * 0.5) / intr.fx * config.z_max_m + config.margin_m;
    let half_h = (intr.height as f32 * 0.5) / intr.fy * config.z_max_m + config.margin_m;
    let dims = [
        ((2.0 * half_w) / config.voxel_size_m).ceil().max(1.0) as u32,
        ((2.0 * half_h) / config.voxel_size_m).ceil().max(1.0) as u32,
        ((config.z_max_m - config.z_min_m) / config.voxel_size_m)
            .ceil()
            .max(1.0) as u32,
    ];
    ge_fusion::Tsdf::new(
        dims,
        config.voxel_size_m,
        [-half_w, -half_h, config.z_min_m],
        config.truncation_m,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ge_backend_trait::DepthMap;

    struct LastMeshSink {
        frames: usize,
        depths: usize,
        meshes: Vec<MeshPacket>,
    }

    impl PipelineSink for LastMeshSink {
        fn on_frame(&mut self, _packet: &FramePacket) -> anyhow::Result<()> {
            self.frames += 1;
            Ok(())
        }

        fn on_depth(&mut self, _frame: &FramePacket, _depth: &DepthPacket) -> anyhow::Result<()> {
            self.depths += 1;
            Ok(())
        }

        fn on_mesh(&mut self, mesh: &MeshPacket) -> anyhow::Result<()> {
            self.meshes.push(mesh.clone());
            Ok(())
        }
    }

    struct BadDepth;

    impl DepthBackend for BadDepth {
        fn name(&self) -> &str {
            "bad"
        }

        fn infer(&mut self, _frame: &Frame) -> anyhow::Result<DepthMap> {
            Ok(DepthMap {
                width: 1,
                height: 1,
                depth_m: vec![1.0],
            })
        }
    }

    #[test]
    fn fusion_runner_emits_frame_depth_and_mesh_packets() {
        let mut source = ge_camera_like_source(2, 64, 64);
        let mut depth = ConstantDepthForTest(2.5);
        let mut sink = LastMeshSink {
            frames: 0,
            depths: 0,
            meshes: Vec::new(),
        };
        let config = FusionConfig {
            voxel_size_m: 0.10,
            truncation_m: 0.30,
            mesh_every_n: 1,
            ..FusionConfig::default()
        };

        let processed = run_fusion_sync(&mut source, &mut depth, config, &mut sink).unwrap();
        assert_eq!(processed, 2);
        assert_eq!(sink.frames, 2);
        assert_eq!(sink.depths, 2);
        assert_eq!(sink.meshes.len(), 2);
        assert!(sink.meshes.last().unwrap().mesh.vertex_count() > 0);
    }

    #[test]
    fn fusion_runner_rejects_depth_shape_mismatch() {
        let mut source = ge_camera_like_source(1, 8, 8);
        let mut depth = BadDepth;
        let mut sink = LastMeshSink {
            frames: 0,
            depths: 0,
            meshes: Vec::new(),
        };
        let err = run_fusion_sync(&mut source, &mut depth, FusionConfig::default(), &mut sink)
            .unwrap_err()
            .to_string();
        assert!(err.contains("depth map"));
    }

    struct ConstantDepthForTest(f32);

    impl DepthBackend for ConstantDepthForTest {
        fn name(&self) -> &str {
            "constant-test"
        }

        fn infer(&mut self, frame: &Frame) -> anyhow::Result<DepthMap> {
            Ok(DepthMap {
                width: frame.width,
                height: frame.height,
                depth_m: vec![self.0; frame.pixel_count()],
            })
        }
    }

    struct TestSource {
        remaining: usize,
        width: u32,
        height: u32,
    }

    fn ge_camera_like_source(frames: usize, width: u32, height: u32) -> TestSource {
        TestSource {
            remaining: frames,
            width,
            height,
        }
    }

    impl CaptureSource for TestSource {
        fn intrinsics(&self) -> Option<Intrinsics> {
            Some(default_intrinsics(self.width, self.height))
        }

        fn next_frame(&mut self) -> anyhow::Result<Option<Frame>> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            Ok(Some(Frame {
                width: self.width,
                height: self.height,
                timestamp_ns: 0,
                rgb: vec![0; self.width as usize * self.height as usize * 3],
            }))
        }
    }
}
