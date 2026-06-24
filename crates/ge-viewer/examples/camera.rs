//! Show a live camera feed in a window, with device selection.
//!
//! Usage:
//!   cargo run --release -p ge-viewer --example camera --features "window camera"            # default cam
//!   cargo run --release -p ge-viewer --example camera --features "window camera" -- list    # list devices
//!   cargo run --release -p ge-viewer --example camera --features "window camera" -- iPhone   # by name
//!   cargo run --release -p ge-viewer --example camera --features "window camera" -- 1        # by index
//!
//! On macOS an iPhone/iPad on Continuity Camera shows up as its own device —
//! list to see it, then select it by name (e.g. `iPhone`). Grant camera
//! permission when prompted.

fn main() -> anyhow::Result<()> {
    let arg = std::env::args().nth(1);

    // Always show what's available — makes picking the iPhone easy.
    match ge_viewer::list_cameras() {
        Ok(devices) if !devices.is_empty() => {
            println!("available cameras:");
            for d in &devices {
                println!("  [{}] {}", d.index, d.name);
            }
        }
        Ok(_) => eprintln!("no cameras found"),
        Err(e) => eprintln!("(could not enumerate cameras: {e})"),
    }

    if matches!(arg.as_deref(), Some("list") | Some("--list")) {
        return Ok(());
    }

    let source = match arg.as_deref() {
        None => {
            println!("opening default camera (index 0)…");
            ge_viewer::WebcamSource::open(0)?
        }
        Some(s) => match s.parse::<u32>() {
            Ok(index) => {
                println!("opening camera index {index}…");
                ge_viewer::WebcamSource::open(index)?
            }
            Err(_) => {
                println!("opening camera matching {s:?}…");
                ge_viewer::WebcamSource::open_named(s)?
            }
        },
    };

    let mut camera = source;
    ge_viewer::view_frames(move || camera.next_frame(), "Gods Eye — camera feed")
}
