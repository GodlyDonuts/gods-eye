//! Truncated signed-distance (TSDF) fusion.
//!
//! Design target: a sparse hash of 8³ voxel blocks allocated only near
//! surfaces, depth-confidence-weighted integration, and LRU streaming, with the
//! integrate step running as a `wgpu` compute kernel. This module currently
//! holds a **dense CPU grid** — correct and simple — so the
//! depth → unproject → fuse → mesh chain can be validated end-to-end. The sparse
//! hash + GPU kernel replace the dense grid next, behind the same surface.

use ge_backend_trait::{DepthMap, Intrinsics, Pose};
use ge_mesh::Mesh;
use glam::Vec3;

/// TSDF parameters (block/sparse design constants, retained for the GPU path).
#[derive(Clone, Copy, Debug)]
pub struct TsdfConfig {
    pub voxel_size_m: f32,
    pub truncation_m: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.02,
            truncation_m: 0.08,
        }
    }
}

/// Voxel blocks are 8×8×8 (for the future sparse-hash layout).
pub const BLOCK_DIM: i32 = 8;

/// Map a world-space voxel coordinate to its containing block coordinate.
#[inline]
pub fn block_of(voxel: glam::IVec3) -> glam::IVec3 {
    glam::IVec3::new(
        voxel.x.div_euclid(BLOCK_DIM),
        voxel.y.div_euclid(BLOCK_DIM),
        voxel.z.div_euclid(BLOCK_DIM),
    )
}

/// A dense TSDF volume on a regular grid.
///
/// Voxel `(x,y,z)` covers world point
/// `origin + (xyz + 0.5) * voxel_size`. `tsdf` holds the signed distance to the
/// nearest surface, normalized to `[-1, 1]` by `trunc` (negative = behind the
/// surface / inside, positive = free space toward the camera). `weight` is the
/// running observation count for the weighted average.
pub struct Tsdf {
    pub dims: [u32; 3],
    pub voxel_size: f32,
    pub origin: Vec3,
    pub trunc: f32,
    tsdf: Vec<f32>,
    weight: Vec<f32>,
}

impl Tsdf {
    pub fn new(dims: [u32; 3], voxel_size: f32, origin: [f32; 3], trunc: f32) -> Self {
        let n = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        Self {
            dims,
            voxel_size,
            origin: Vec3::from_array(origin),
            trunc,
            tsdf: vec![1.0; n],
            weight: vec![0.0; n],
        }
    }

    #[inline]
    fn linear(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.dims[0] + z * self.dims[0] * self.dims[1]) as usize
    }

    pub fn voxel_count(&self) -> usize {
        self.tsdf.len()
    }

    /// Number of voxels that received at least one observation.
    pub fn observed_voxels(&self) -> usize {
        self.weight.iter().filter(|&&w| w > 0.0).count()
    }

    /// Read-only TSDF values, exposed for CPU/GPU parity tests and future
    /// backend implementations.
    pub fn tsdf_values(&self) -> &[f32] {
        &self.tsdf
    }

    /// Read-only integration weights.
    pub fn weights(&self) -> &[f32] {
        &self.weight
    }

    /// Integrate one depth frame, given its camera intrinsics and the
    /// camera-to-world pose (projective TSDF integration).
    pub fn integrate(&mut self, depth: &DepthMap, intr: &Intrinsics, cam_to_world: &Pose) {
        let world_to_cam = cam_to_world.inverse();
        let half = 0.5 * self.voxel_size;
        let (dw, dh) = (depth.width as f32, depth.height as f32);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let world = self.origin
                        + Vec3::new(x as f32, y as f32, z as f32) * self.voxel_size
                        + Vec3::splat(half);
                    let c = world_to_cam.transform_point3(world);
                    if c.z <= 1e-4 {
                        continue;
                    }
                    let u = intr.fx * c.x / c.z + intr.cx;
                    let v = intr.fy * c.y / c.z + intr.cy;
                    if u < 0.0 || v < 0.0 || u >= dw || v >= dh {
                        continue;
                    }
                    let d = depth.depth_m[(v as usize) * depth.width as usize + (u as usize)];
                    if d <= 0.0 || d.is_nan() {
                        continue;
                    }
                    let depth_i = (v as usize) * depth.width as usize + (u as usize);
                    let obs_weight = depth
                        .confidence
                        .as_ref()
                        .map(|c| c[depth_i].clamp(0.0, 1.0))
                        .unwrap_or(1.0);
                    if obs_weight <= 0.0 {
                        continue;
                    }
                    // Signed distance along the ray: + in front of the surface.
                    let sdf = d - c.z;
                    if sdf < -self.trunc {
                        continue; // occluded / behind surface beyond truncation
                    }
                    let val = (sdf / self.trunc).clamp(-1.0, 1.0);
                    let i = self.linear(x, y, z);
                    let w = self.weight[i];
                    self.tsdf[i] = (self.tsdf[i] * w + val * obs_weight) / (w + obs_weight);
                    self.weight[i] = w + obs_weight;
                }
            }
        }
    }

    /// Extract a world-space triangle mesh from the current volume.
    pub fn extract_mesh(&self) -> Mesh {
        let mut mesh = ge_mesh::surface_nets_mesh(&self.tsdf, self.dims);
        for p in mesh.positions.iter_mut() {
            p[0] = self.origin.x + p[0] * self.voxel_size;
            p[1] = self.origin.y + p[1] * self.voxel_size;
            p[2] = self.origin.z + p[2] * self.voxel_size;
        }
        mesh
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::Tsdf;
    use bytemuck::{Pod, Zeroable};
    use ge_backend_trait::{DepthMap, Intrinsics, Pose};
    use std::sync::mpsc;
    use wgpu::util::DeviceExt;

    const WORKGROUP_SIZE: u32 = 256;

    /// Dense `wgpu` TSDF integrator.
    ///
    /// This is the first GPU backend step: it mirrors the CPU dense-grid
    /// reference path and copies the updated buffers back into [`Tsdf`] after
    /// each dispatch. Keeping CPU and GPU on the same data model gives us a
    /// correctness oracle before moving to resident sparse GPU blocks.
    pub struct WgpuTsdfIntegrator {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
    }

    impl WgpuTsdfIntegrator {
        pub fn new() -> anyhow::Result<Self> {
            pollster::block_on(Self::new_async())
        }

        pub async fn new_async() -> anyhow::Result<Self> {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ge-fusion device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await?;

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tsdf integrate bind group layout"),
                entries: &[
                    storage_entry(0, false),
                    storage_entry(1, false),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tsdf integrate shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/tsdf_integrate.wgsl").into(),
                ),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tsdf integrate pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tsdf integrate pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                layout,
            })
        }

        /// Allocate a dense TSDF volume whose TSDF and weight buffers stay
        /// resident on the GPU across integration calls.
        pub fn create_volume(
            &self,
            dims: [u32; 3],
            voxel_size: f32,
            origin: [f32; 3],
            trunc: f32,
        ) -> WgpuTsdfVolume {
            let cpu = Tsdf::new(dims, voxel_size, origin, trunc);
            WgpuTsdfVolume {
                dims,
                voxel_size,
                origin: cpu.origin.to_array(),
                trunc,
                voxel_count: cpu.voxel_count(),
                tsdf_buffer: self.storage_buffer("resident tsdf", cpu.tsdf_values()),
                weight_buffer: self.storage_buffer("resident weight", cpu.weights()),
            }
        }

        /// Integrate one depth frame into a resident GPU volume.
        ///
        /// Depth/confidence and params are uploaded per call, but the large
        /// TSDF/weight buffers remain on the GPU. This removes the most obvious
        /// per-frame transfer cost from the initial parity wrapper.
        pub fn integrate_volume(
            &self,
            volume: &WgpuTsdfVolume,
            depth: &DepthMap,
            intr: &Intrinsics,
            cam_to_world: &Pose,
        ) -> anyhow::Result<()> {
            self.validate_depth(depth)?;

            let depth_buffer = self.storage_buffer("depth", &depth.depth_m);
            let confidence_storage;
            let confidence_slice: &[f32] = if let Some(confidence) = &depth.confidence {
                confidence
            } else {
                confidence_storage = vec![1.0f32];
                &confidence_storage
            };
            let confidence_buffer = self.storage_buffer("confidence", confidence_slice);
            let params = GpuParams::new_volume(volume, depth, intr, cam_to_world);
            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tsdf params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident tsdf integrate bind group"),
                layout: &self.layout,
                entries: &[
                    bind_entry(0, &volume.tsdf_buffer),
                    bind_entry(1, &volume.weight_buffer),
                    bind_entry(2, &depth_buffer),
                    bind_entry(3, &confidence_buffer),
                    bind_entry(4, &params_buffer),
                ],
            });

            self.dispatch(volume.voxel_count as u32, &bind_group);
            Ok(())
        }

        /// Download a resident GPU volume into the CPU reference representation.
        pub fn download_volume(&self, volume: &WgpuTsdfVolume) -> anyhow::Result<Tsdf> {
            let mut tsdf = Tsdf::new(volume.dims, volume.voxel_size, volume.origin, volume.trunc);
            tsdf.tsdf = self.read_f32_buffer(&volume.tsdf_buffer, volume.voxel_count)?;
            tsdf.weight = self.read_f32_buffer(&volume.weight_buffer, volume.voxel_count)?;
            Ok(tsdf)
        }

        pub fn integrate(
            &self,
            tsdf: &mut Tsdf,
            depth: &DepthMap,
            intr: &Intrinsics,
            cam_to_world: &Pose,
        ) -> anyhow::Result<()> {
            self.validate_depth(depth)?;

            let total = tsdf.voxel_count() as u32;
            let tsdf_buffer = self.storage_buffer("tsdf", &tsdf.tsdf);
            let weight_buffer = self.storage_buffer("weight", &tsdf.weight);
            let depth_buffer = self.storage_buffer("depth", &depth.depth_m);
            let confidence_storage;
            let confidence_slice: &[f32] = if let Some(confidence) = &depth.confidence {
                confidence
            } else {
                confidence_storage = vec![1.0f32];
                &confidence_storage
            };
            let confidence_buffer = self.storage_buffer("confidence", confidence_slice);
            let params = GpuParams::new(tsdf, depth, intr, cam_to_world);
            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tsdf params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tsdf integrate bind group"),
                layout: &self.layout,
                entries: &[
                    bind_entry(0, &tsdf_buffer),
                    bind_entry(1, &weight_buffer),
                    bind_entry(2, &depth_buffer),
                    bind_entry(3, &confidence_buffer),
                    bind_entry(4, &params_buffer),
                ],
            });

            self.dispatch(total, &bind_group);

            tsdf.tsdf = self.read_f32_buffer(&tsdf_buffer, tsdf.tsdf.len())?;
            tsdf.weight = self.read_f32_buffer(&weight_buffer, tsdf.weight.len())?;
            Ok(())
        }

        fn validate_depth(&self, depth: &DepthMap) -> anyhow::Result<()> {
            anyhow::ensure!(
                depth.depth_m.len() == depth.width as usize * depth.height as usize,
                "depth buffer length does not match dimensions"
            );
            if let Some(confidence) = &depth.confidence {
                anyhow::ensure!(
                    confidence.len() == depth.depth_m.len(),
                    "confidence buffer length does not match depth"
                );
            }
            Ok(())
        }

        fn dispatch(&self, total: u32, bind_group: &wgpu::BindGroup) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tsdf integrate encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tsdf integrate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        fn storage_buffer(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
        }

        fn read_f32_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> anyhow::Result<Vec<f32>> {
            let slice = buffer.slice(..);
            let (tx, rx) = mpsc::sync_channel(1);
            wgpu::util::DownloadBuffer::read_buffer(&self.device, &self.queue, &slice, move |r| {
                let _ = tx.send(r.map(|b| bytemuck::cast_slice::<u8, f32>(&b).to_vec()));
            });
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            let out = rx.recv()??;
            anyhow::ensure!(out.len() == len, "downloaded buffer length mismatch");
            Ok(out)
        }
    }

    /// Dense GPU-resident TSDF volume.
    pub struct WgpuTsdfVolume {
        pub dims: [u32; 3],
        pub voxel_size: f32,
        pub origin: [f32; 3],
        pub trunc: f32,
        voxel_count: usize,
        tsdf_buffer: wgpu::Buffer,
        weight_buffer: wgpu::Buffer,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct GpuParams {
        dims: [u32; 4],
        depth_size: [u32; 4],
        origin: [f32; 4],
        intrinsics: [f32; 4],
        scalars: [f32; 4],
        world_to_cam: [f32; 16],
    }

    impl GpuParams {
        fn new(tsdf: &Tsdf, depth: &DepthMap, intr: &Intrinsics, cam_to_world: &Pose) -> Self {
            let world_to_cam = cam_to_world.inverse();
            Self {
                dims: [
                    tsdf.dims[0],
                    tsdf.dims[1],
                    tsdf.dims[2],
                    tsdf.voxel_count() as u32,
                ],
                depth_size: [
                    depth.width,
                    depth.height,
                    u32::from(depth.confidence.is_some()),
                    0,
                ],
                origin: [tsdf.origin.x, tsdf.origin.y, tsdf.origin.z, 0.0],
                intrinsics: [intr.fx, intr.fy, intr.cx, intr.cy],
                scalars: [tsdf.voxel_size, tsdf.trunc, 0.0, 0.0],
                world_to_cam: glam::Mat4::from(world_to_cam).to_cols_array(),
            }
        }

        fn new_volume(
            volume: &WgpuTsdfVolume,
            depth: &DepthMap,
            intr: &Intrinsics,
            cam_to_world: &Pose,
        ) -> Self {
            let world_to_cam = cam_to_world.inverse();
            Self {
                dims: [
                    volume.dims[0],
                    volume.dims[1],
                    volume.dims[2],
                    volume.voxel_count as u32,
                ],
                depth_size: [
                    depth.width,
                    depth.height,
                    u32::from(depth.confidence.is_some()),
                    0,
                ],
                origin: [volume.origin[0], volume.origin[1], volume.origin[2], 0.0],
                intrinsics: [intr.fx, intr.fy, intr.cx, intr.cy],
                scalars: [volume.voxel_size, volume.trunc, 0.0, 0.0],
                world_to_cam: glam::Mat4::from(world_to_cam).to_cols_array(),
            }
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::scenes;

        #[test]
        fn gpu_integrator_matches_cpu_on_wall_scene() {
            let integrator = match WgpuTsdfIntegrator::new() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("skipping GPU parity test: {e}");
                    return;
                }
            };
            let (depth, intr) = scenes::wall_with_panel(32);
            let dims = [32, 32, 24];
            let voxel = 0.08;
            let origin = [-1.3, -1.3, 1.2];
            let trunc = 4.0 * voxel;
            let mut cpu = Tsdf::new(dims, voxel, origin, trunc);
            cpu.integrate(&depth, &intr, &Pose::IDENTITY);
            let mut gpu = Tsdf::new(dims, voxel, origin, trunc);
            integrator
                .integrate(&mut gpu, &depth, &intr, &Pose::IDENTITY)
                .unwrap();

            assert_eq!(cpu.voxel_count(), gpu.voxel_count());
            for (a, b) in cpu.tsdf_values().iter().zip(gpu.tsdf_values()) {
                assert!((a - b).abs() < 1e-4, "tsdf mismatch: cpu={a} gpu={b}");
            }
            for (a, b) in cpu.weights().iter().zip(gpu.weights()) {
                assert!((a - b).abs() < 1e-4, "weight mismatch: cpu={a} gpu={b}");
            }
        }

        #[test]
        fn resident_gpu_volume_matches_cpu_after_two_integrations() {
            let integrator = match WgpuTsdfIntegrator::new() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("skipping GPU resident parity test: {e}");
                    return;
                }
            };
            let (depth, intr) = scenes::wall_with_panel(32);
            let dims = [32, 32, 24];
            let voxel = 0.08;
            let origin = [-1.3, -1.3, 1.2];
            let trunc = 4.0 * voxel;

            let mut cpu = Tsdf::new(dims, voxel, origin, trunc);
            cpu.integrate(&depth, &intr, &Pose::IDENTITY);
            cpu.integrate(&depth, &intr, &Pose::IDENTITY);

            let gpu_volume = integrator.create_volume(dims, voxel, origin, trunc);
            integrator
                .integrate_volume(&gpu_volume, &depth, &intr, &Pose::IDENTITY)
                .unwrap();
            integrator
                .integrate_volume(&gpu_volume, &depth, &intr, &Pose::IDENTITY)
                .unwrap();
            let gpu = integrator.download_volume(&gpu_volume).unwrap();

            assert_eq!(cpu.voxel_count(), gpu.voxel_count());
            for (a, b) in cpu.tsdf_values().iter().zip(gpu.tsdf_values()) {
                assert!((a - b).abs() < 1e-4, "tsdf mismatch: cpu={a} gpu={b}");
            }
            for (a, b) in cpu.weights().iter().zip(gpu.weights()) {
                assert!((a - b).abs() < 1e-4, "weight mismatch: cpu={a} gpu={b}");
            }
        }
    }
}

/// Synthetic depth scenes for validating the fusion + meshing chain without a
/// camera or model.
pub mod scenes {
    use ge_backend_trait::{DepthMap, Intrinsics};

    /// A frontal wall at `wall_z` metres with a nearer square panel at
    /// `panel_z` in the centre — mimics "a wall with a raised painting/shelf".
    /// Returns the depth map and matching pinhole intrinsics (60° FOV).
    pub fn wall_with_panel(size: u32) -> (DepthMap, Intrinsics) {
        let f = (size as f32) / 2.0 / (60.0f32.to_radians() / 2.0).tan();
        let intr = Intrinsics {
            fx: f,
            fy: f,
            cx: size as f32 / 2.0,
            cy: size as f32 / 2.0,
            width: size,
            height: size,
        };
        let (wall_z, panel_z) = (2.5f32, 1.8f32);
        let (lo, hi) = (size / 3, 2 * size / 3);
        let mut depth_m = vec![0.0f32; (size * size) as usize];
        for v in 0..size {
            for u in 0..size {
                let in_panel = u >= lo && u < hi && v >= lo && v < hi;
                depth_m[(v * size + u) as usize] = if in_panel { panel_z } else { wall_z };
            }
        }
        (
            DepthMap {
                width: size,
                height: size,
                depth_m,
                confidence: None,
            },
            intr,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn block_of_handles_negative_coords() {
        assert_eq!(block_of(IVec3::new(0, 0, 0)), IVec3::new(0, 0, 0));
        assert_eq!(block_of(IVec3::new(7, 7, 7)), IVec3::new(0, 0, 0));
        assert_eq!(block_of(IVec3::new(8, -1, -8)), IVec3::new(1, -1, -1));
    }

    #[test]
    fn integrate_wall_yields_surface_near_expected_depth() {
        let (depth, intr) = scenes::wall_with_panel(128);
        // Volume spanning the frustum; z brackets both wall (2.5) and panel (1.8).
        let voxel = 0.03;
        let origin = [-1.8, -1.8, 1.2];
        let dims = [120, 120, 60];
        let mut tsdf = Tsdf::new(dims, voxel, origin, 4.0 * voxel);
        tsdf.integrate(&depth, &intr, &Pose::IDENTITY);
        assert!(tsdf.observed_voxels() > 0, "some voxels observed");
        let mesh = tsdf.extract_mesh();
        assert!(mesh.triangle_count() > 100, "wall produces a real surface");
        // Mean vertex depth should sit between the panel and wall planes.
        let mean_z = mesh.positions.iter().map(|p| p[2]).sum::<f32>() / mesh.positions.len() as f32;
        assert!(
            (1.6..2.7).contains(&mean_z),
            "surface depth {mean_z} should be near the wall/panel"
        );
    }

    #[test]
    fn low_confidence_conflicting_depth_moves_tsdf_less() {
        let intr = Intrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
            width: 1,
            height: 1,
        };
        let dims = [1, 1, 1];
        let voxel = 1.0;
        let origin = [-0.5, -0.5, 1.0];
        let trunc = 1.0;

        let initial = DepthMap {
            width: 1,
            height: 1,
            depth_m: vec![1.25],
            confidence: Some(vec![1.0]),
        };
        let high_conflict = DepthMap {
            width: 1,
            height: 1,
            depth_m: vec![1.75],
            confidence: Some(vec![1.0]),
        };
        let low_conflict = DepthMap {
            width: 1,
            height: 1,
            depth_m: vec![1.75],
            confidence: Some(vec![0.25]),
        };

        let mut high_tsdf = Tsdf::new(dims, voxel, origin, trunc);
        high_tsdf.integrate(&initial, &intr, &Pose::IDENTITY);
        high_tsdf.integrate(&high_conflict, &intr, &Pose::IDENTITY);
        let mut low_tsdf = Tsdf::new(dims, voxel, origin, trunc);
        low_tsdf.integrate(&initial, &intr, &Pose::IDENTITY);
        low_tsdf.integrate(&low_conflict, &intr, &Pose::IDENTITY);

        assert_eq!(high_tsdf.observed_voxels(), 1);
        assert_eq!(low_tsdf.observed_voxels(), 1);
        assert!(
            high_tsdf.weight[0] > low_tsdf.weight[0],
            "high-confidence observation should accumulate more weight"
        );
        assert!(
            high_tsdf.tsdf[0] > low_tsdf.tsdf[0],
            "high-confidence conflicting depth should move the TSDF farther"
        );
    }
}
