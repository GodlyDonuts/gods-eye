//! Show the live webcam feed in a window.
//!
//! Usage:
//!   cargo run -p ge-viewer --example camera --features "window camera"
//!   cargo run -p ge-viewer --example camera --features "window camera" -- 1   # camera index 1
//!
//! On macOS, grant camera permission when prompted (or in System Settings →
//! Privacy & Security → Camera).

fn main() -> anyhow::Result<()> {
    let index = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    println!("opening camera {index} — grant camera permission if prompted…");
    let mut camera = ge_viewer::WebcamSource::open(index)?;
    ge_viewer::view_frames(move || camera.next_frame(), "Gods Eye — camera feed")
}
