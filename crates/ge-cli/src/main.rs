//! Gods Eye CLI.
//!
//! `run` drives the M0 synthetic capture -> depth smoke spine. `bench-depth`
//! (built with `--features coreml` or `--features onnx`) is the M0 latency
//! spike: it measures Depth Anything V2 (ViT-S) inference latency on this
//! machine at several input resolutions — the number that unblocks the fps
//! budget.

use clap::{Parser, Subcommand};
use ge_core::run_sync;

#[derive(Parser, Debug)]
#[command(name = "gods-eye", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the synthetic capture -> depth smoke spine (M0).
    Run(RunArgs),
    /// Benchmark monocular depth inference latency (M0 spike).
    BenchDepth(BenchArgs),
    /// Extract a triangle mesh from a synthetic SDF and write it to PLY.
    DemoMesh(DemoMeshArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Number of synthetic frames to process.
    #[arg(long, default_value_t = 30)]
    frames: u32,
    /// Synthetic frame width.
    #[arg(long, default_value_t = 640)]
    width: u32,
    /// Synthetic frame height.
    #[arg(long, default_value_t = 480)]
    height: u32,
}

#[derive(clap::Args, Debug)]
struct BenchArgs {
    /// Path to the DAv2-Small ONNX model.
    #[arg(long)]
    model: String,
    /// Use the Apple CoreML execution provider (else CPU).
    #[arg(long)]
    coreml: bool,
    /// Square input sizes to sweep (multiples of 14).
    #[arg(long, value_delimiter = ',', default_value = "252,392,518")]
    sizes: Vec<u32>,
    /// Timed iterations per size.
    #[arg(long, default_value_t = 30)]
    iters: usize,
    /// Warmup iterations per size (untimed).
    #[arg(long, default_value_t = 5)]
    warmup: usize,
}

#[derive(clap::Args, Debug)]
struct DemoMeshArgs {
    /// Output PLY path.
    #[arg(long, default_value = "out/sphere.ply")]
    out: String,
    /// Sphere radius in normalized grid units (~[0, 1]).
    #[arg(long, default_value_t = 0.7)]
    radius: f32,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(args),
        Command::BenchDepth(args) => cmd_bench_depth(args),
        Command::DemoMesh(args) => cmd_demo_mesh(args),
    }
}

fn cmd_demo_mesh(args: DemoMeshArgs) -> anyhow::Result<()> {
    let mesh = ge_mesh::demo::sphere_mesh(args.radius);
    let path = std::path::Path::new(&args.out);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    mesh.write_ply(path)?;
    println!(
        "wrote {} — {} vertices, {} triangles (CPU surface-nets, sphere r={})",
        args.out,
        mesh.vertex_count(),
        mesh.triangle_count(),
        args.radius
    );
    Ok(())
}

fn cmd_run(args: RunArgs) -> anyhow::Result<()> {
    let mut source =
        ge_camera::SolidColorSource::new(args.width, args.height, args.frames, [60, 120, 200]);
    let mut depth = ge_depth::ConstantDepth::new(2.5);

    let mut depth_acc = 0.0f64;
    let processed = run_sync(&mut source, &mut depth, |_frame, dm| {
        depth_acc += dm.depth_m.first().copied().unwrap_or(0.0) as f64;
    })?;

    println!(
        "gods-eye M0 spine: processed {processed} frame(s) at {}x{} (depth backend: constant)",
        args.width, args.height
    );
    let _ = depth_acc;
    Ok(())
}

#[cfg(feature = "onnx")]
fn cmd_bench_depth(args: BenchArgs) -> anyhow::Result<()> {
    let accel = if args.coreml {
        ge_depth::Accel::CoreMl
    } else {
        ge_depth::Accel::Cpu
    };
    println!(
        "Benchmarking depth: model={} accel={} iters={} warmup={}",
        args.model,
        accel.label(),
        args.iters,
        args.warmup
    );
    let results = ge_depth::bench(&args.model, accel, &args.sizes, args.iters, args.warmup)?;

    println!("\n  size   |  min ms  | median ms |  p95 ms  |  ~fps  | note");
    println!("  -------+----------+-----------+----------+--------+------");
    for r in &results {
        if r.ok {
            let fps = if r.median_ms > 0.0 {
                1000.0 / r.median_ms
            } else {
                0.0
            };
            println!(
                "  {:>4}²  | {:>7.1}  | {:>8.1}  | {:>7.1}  | {:>5.1}  | out_len={} min={:.3}",
                r.size, r.min_ms, r.median_ms, r.p95_ms, fps, r.out_len, r.out_min
            );
        } else {
            println!(
                "  {:>4}²  |    -     |     -     |    -     |    -   | SKIPPED: {}",
                r.size, r.note
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn cmd_bench_depth(_args: BenchArgs) -> anyhow::Result<()> {
    anyhow::bail!(
        "bench-depth needs ONNX Runtime. Rebuild with `--features coreml` (macOS) or `--features onnx` (CPU)."
    )
}
