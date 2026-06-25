//! Live 2D panorama — develops camera-rotation tracking in isolation.
//!
//! Each frame's 2D motion (a pan/tilt of the camera) is estimated with
//! Lucas-Kanade and accumulated; the frame is pasted onto a large canvas at the
//! tracked offset. Pan around and the canvas fills out; pan back and what you
//! already captured is still there. No depth, no 3D — just the tracking, so we
//! can make the "backtracking" solid before re-introducing reconstruction.
//!
//! Run in release:
//!   cargo run --release -p ge-viewer --example panorama --features panorama -- 1

use ge_backend_trait::Frame;

/// Downsample a frame to `target_w` and return (w, h, RGB8, grayscale[0,1]).
fn to_working(frame: &Frame, target_w: u32) -> (usize, usize, Vec<u8>, Vec<f32>) {
    let (sw, sh) = (frame.width as usize, frame.height as usize);
    let tw = (target_w as usize).max(1);
    let th = (sh * tw / sw).max(1);
    let mut rgb = vec![0u8; tw * th * 3];
    let mut gray = vec![0f32; tw * th];
    for ty in 0..th {
        let sy = ty * sh / th;
        for tx in 0..tw {
            let sx = tx * sw / tw;
            let si = (sy * sw + sx) * 3;
            let di = ty * tw + tx;
            let (r, g, b) = (frame.rgb[si], frame.rgb[si + 1], frame.rgb[si + 2]);
            rgb[di * 3] = r;
            rgb[di * 3 + 1] = g;
            rgb[di * 3 + 2] = b;
            gray[di] = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0;
        }
    }
    (tw, th, rgb, gray)
}

fn main() -> anyhow::Result<()> {
    let cam = std::env::args().nth(1);

    if let Ok(devices) = ge_viewer::list_cameras() {
        println!("available cameras:");
        for d in &devices {
            println!("  [{}] {}", d.index, d.name);
        }
    }
    let mut camera = match cam.as_deref() {
        None => ge_viewer::WebcamSource::open(0)?,
        Some(s) => match s.parse::<u32>() {
            Ok(index) => ge_viewer::WebcamSource::open(index)?,
            Err(_) => ge_viewer::WebcamSource::open_named(s)?,
        },
    };

    let (cw, ch) = (2400usize, 1000usize);
    let mut canvas = vec![16u8; cw * ch * 3]; // near-black background
    let mut prev_gray: Option<(Vec<f32>, usize, usize)> = None;
    let (mut ox, mut oy) = (cw as f32 * 0.5, ch as f32 * 0.5);

    println!("panorama — pan slowly to build it, pan back to revisit…");
    ge_viewer::view_frames(
        move || {
            let frame = match camera.next_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let (tw, th, rgb, gray) = to_working(&frame, 240);

            if let Some((pg, pw, ph)) = prev_gray.as_ref() {
                if *pw == tw && *ph == th {
                    let (dx, dy) = ge_slam::estimate_translation(pg, &gray, tw, th);
                    // Reject implausible jumps (tracking loss on fast motion).
                    let max_step = tw as f32 * 0.4;
                    if dx.abs() < max_step && dy.abs() < max_step {
                        // Camera pan is opposite to image-content motion.
                        ox = (ox - dx).clamp(tw as f32 * 0.5, cw as f32 - tw as f32 * 0.5);
                        oy = (oy - dy).clamp(th as f32 * 0.5, ch as f32 - th as f32 * 0.5);
                    }
                }
            }

            // Paste the current frame onto the canvas at the tracked offset.
            let left = (ox - tw as f32 * 0.5).round() as i32;
            let top = (oy - th as f32 * 0.5).round() as i32;
            for y in 0..th {
                let cy = top + y as i32;
                if cy < 0 || cy >= ch as i32 {
                    continue;
                }
                for x in 0..tw {
                    let cx = left + x as i32;
                    if cx < 0 || cx >= cw as i32 {
                        continue;
                    }
                    let di = ((cy as usize) * cw + cx as usize) * 3;
                    let si = (y * tw + x) * 3;
                    canvas[di] = rgb[si];
                    canvas[di + 1] = rgb[si + 1];
                    canvas[di + 2] = rgb[si + 2];
                }
            }

            prev_gray = Some((gray, tw, th));
            Ok(Some(Frame {
                width: cw as u32,
                height: ch as u32,
                timestamp_ns: 0,
                rgb: canvas.clone(),
            }))
        },
        "Gods Eye — panorama",
    )
}
