//! Live 2D panorama — develops camera-rotation tracking in isolation.
//!
//! Each frame's 2D motion is estimated with Lucas-Kanade as a *rigid* transform
//! (in-plane rotation + translation), so camera roll is detected and corrected,
//! and accumulated into a canvas transform. Frames are warp-pasted onto a large
//! canvas; pan/roll around and it fills out, pan back and what you captured is
//! still there — aligned. No depth/3D, just the tracking.
//!
//! Run in release:
//!   cargo run --release -p ge-viewer --example panorama --features panorama -- 1

use ge_backend_trait::Frame;
use glam::{Affine2, Vec2};

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
    // Maps current-frame pixel coords -> canvas pixel coords (accumulated).
    let mut canvas_from_cur: Option<Affine2> = None;

    println!("panorama — pan/roll slowly to build it, return to revisit…");
    ge_viewer::view_frames(
        move || {
            let frame = match camera.next_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let (tw, th, rgb, gray) = to_working(&frame, 240);
            let frame_center = Vec2::new(tw as f32 * 0.5, th as f32 * 0.5);

            // Initialize the canvas transform: place the first frame centered.
            let mut m = *canvas_from_cur.get_or_insert_with(|| {
                Affine2::from_translation(
                    Vec2::new(cw as f32 * 0.5, ch as f32 * 0.5) - frame_center,
                )
            });

            if let Some((pg, pw, ph)) = prev_gray.as_ref() {
                if *pw == tw && *ph == th {
                    let rel = ge_slam::estimate_rigid2d(pg, &gray, tw, th); // prev_from_cur
                                                                            // Reject implausible motion (tracking loss on fast moves).
                    let moved = (rel.transform_point2(frame_center) - frame_center).length();
                    let angle = rel.matrix2.x_axis.y.atan2(rel.matrix2.x_axis.x).abs();
                    if moved < tw as f32 * 0.4 && angle < 0.4 {
                        m *= rel;
                        canvas_from_cur = Some(m);
                    }
                }
            }

            // Warp-paste the current frame onto the canvas (inverse warp so a
            // rolled frame fills without holes).
            let cur_from_canvas = m.inverse();
            let corners = [
                m.transform_point2(Vec2::new(0.0, 0.0)),
                m.transform_point2(Vec2::new(tw as f32, 0.0)),
                m.transform_point2(Vec2::new(0.0, th as f32)),
                m.transform_point2(Vec2::new(tw as f32, th as f32)),
            ];
            let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for c in corners {
                minx = minx.min(c.x);
                miny = miny.min(c.y);
                maxx = maxx.max(c.x);
                maxy = maxy.max(c.y);
            }
            let x0 = (minx.floor() as i32).clamp(0, cw as i32);
            let x1 = (maxx.ceil() as i32).clamp(0, cw as i32);
            let y0 = (miny.floor() as i32).clamp(0, ch as i32);
            let y1 = (maxy.ceil() as i32).clamp(0, ch as i32);
            for cy in y0..y1 {
                for cx in x0..x1 {
                    let src = cur_from_canvas.transform_point2(Vec2::new(cx as f32, cy as f32));
                    if src.x >= 0.0 && src.y >= 0.0 && src.x < tw as f32 && src.y < th as f32 {
                        let si = ((src.y as usize) * tw + src.x as usize) * 3;
                        let di = ((cy as usize) * cw + cx as usize) * 3;
                        canvas[di] = rgb[si];
                        canvas[di + 1] = rgb[si + 1];
                        canvas[di + 2] = rgb[si + 2];
                    }
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
