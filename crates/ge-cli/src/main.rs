//! Gods Eye CLI.
//!
//! M0: wires a minimal capture -> depth spine and runs it on a synthetic
//! source, reporting throughput. Real sources, fusion, meshing, and the viewer
//! are wired in as their stages land.

use clap::Parser;
use ge_core::run_sync;

/// Real-time monocular triangle-mesh reconstruction.
#[derive(Parser, Debug)]
#[command(name = "gods-eye", version, about)]
struct Args {
    /// Number of synthetic frames to process (M0 smoke run).
    #[arg(long, default_value_t = 30)]
    frames: u32,

    /// Synthetic frame width.
    #[arg(long, default_value_t = 640)]
    width: u32,

    /// Synthetic frame height.
    #[arg(long, default_value_t = 480)]
    height: u32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut source =
        ge_camera::SolidColorSource::new(args.width, args.height, args.frames, [60, 120, 200]);
    let mut depth = ge_depth::ConstantDepth::new(2.5);

    let mut last_depth_sum = 0.0f64;
    let processed = run_sync(&mut source, &mut depth, |_frame, dm| {
        // Touch the depth so the work isn't optimized away; real M0 forwards
        // this to fusion -> meshing -> viewer.
        last_depth_sum += dm.depth_m.first().copied().unwrap_or(0.0) as f64;
    })?;

    println!(
        "gods-eye M0 spine: processed {processed} frame(s) at {}x{} (depth backend: constant)",
        args.width, args.height
    );
    let _ = last_depth_sum;
    Ok(())
}
