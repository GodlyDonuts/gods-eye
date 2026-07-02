//! Footprint polygonization — a plane's true outline, not a bounding box.
//!
//! A confirmed plane owns a cloud of world-space footprint points (its cell
//! centroids). Projected into the plane's own `u`/`v` frame they become 2D
//! points; this module rasterizes them into an occupancy grid, fills gaps and
//! holes, extracts the outline of the largest filled region, and simplifies it
//! to a handful of vertices. The result is the plane's real shape — an L-shaped
//! floor stays L-shaped — instead of an oriented bounding rectangle.
//!
//! The pipeline is deliberately robust to sparse, noisy footprints: a
//! morphological close bridges the gaps between per-cell centroids, hole-filling
//! makes the region solid, and Douglas–Peucker smooths the rasterization
//! staircase back into straight walls. Output is triangulated by ear clipping so
//! concave outlines (the whole point) render correctly.

/// Tuning for footprint → polygon extraction. All distances are in metres
/// (plane-local, i.e. world units).
#[derive(Clone, Copy, Debug)]
pub struct PolyParams {
    /// Occupancy cell size.
    pub res: f32,
    /// Morphological-close radius in cells — bridges gaps between the sparse
    /// per-detection-cell centroids so one surface is one region.
    pub bridge: i32,
    /// Douglas–Peucker tolerance — collapses the rasterization staircase and
    /// collinear runs into straight edges while preserving real corners.
    pub simplify_eps: f32,
    /// Safety clamp on grid dimension; `res` is coarsened if the footprint would
    /// exceed this many cells on a side.
    pub max_grid: usize,
    /// Reject a region smaller than this many occupied cells (noise).
    pub min_area_cells: usize,
}

impl Default for PolyParams {
    fn default() -> Self {
        Self {
            res: 0.08,
            bridge: 2,
            simplify_eps: 0.12,
            max_grid: 512,
            min_area_cells: 6,
        }
    }
}

/// Extract a simplified outline polygon (plane-local 2D) from footprint points
/// already projected into the plane's `u`/`v` frame. Returns the polygon as an
/// ordered ring of `[a, b]` vertices (CCW, no repeated closing vertex), or
/// `None` if the footprint is too sparse or degenerate to outline — in which
/// case the caller should fall back to a bounding rectangle.
pub fn footprint_polygon(pts: &[[f32; 2]], p: &PolyParams) -> Option<Vec<[f32; 2]>> {
    if pts.len() < 3 {
        return None;
    }

    // Bounding box of the footprint in plane-local coords.
    let (mut amin, mut amax, mut bmin, mut bmax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &[a, b] in pts {
        amin = amin.min(a);
        amax = amax.max(a);
        bmin = bmin.min(b);
        bmax = bmax.max(b);
    }
    let (ext_a, ext_b) = (amax - amin, bmax - bmin);
    if !ext_a.is_finite() || !ext_b.is_finite() || ext_a.max(ext_b) < 3.0 * p.res {
        return None; // degenerate (a point or a thin line)
    }

    // Grid resolution, coarsened if the footprint is huge, plus a border wide
    // enough that dilation never touches the edge (boundary tracing needs it).
    let border = (p.bridge + 1).max(1);
    let mut res = p.res;
    let span = ext_a.max(ext_b);
    let max_inner = (p.max_grid as i32 - 2 * border).max(4) as f32;
    if span / res > max_inner {
        res = span / max_inner;
    }
    let w = ((ext_a / res).ceil() as i32 + 2 * border).max(border * 2 + 1) as usize;
    let h = ((ext_b / res).ceil() as i32 + 2 * border).max(border * 2 + 1) as usize;
    let (a0, b0) = (amin, bmin); // plane-local coord of inner-grid origin

    // Rasterize occupancy.
    let cell = |a: f32, b: f32| -> (i32, i32) {
        (
            ((a - a0) / res).floor() as i32 + border,
            ((b - b0) / res).floor() as i32 + border,
        )
    };
    let mut occ = vec![false; w * h];
    for &[a, b] in pts {
        let (gx, gy) = cell(a, b);
        if gx >= 0 && gy >= 0 && (gx as usize) < w && (gy as usize) < h {
            occ[gy as usize * w + gx as usize] = true;
        }
    }

    // Close (dilate then erode) to bridge centroid gaps, then fill interior
    // holes so the region is solid and yields one clean outer boundary.
    for _ in 0..p.bridge {
        occ = dilate(&occ, w, h);
    }
    for _ in 0..p.bridge {
        occ = erode(&occ, w, h);
    }
    fill_holes(&mut occ, w, h);

    // Keep the largest connected region only.
    let region = largest_component(&occ, w, h)?;
    if region.iter().filter(|&&b| b).count() < p.min_area_cells {
        return None;
    }

    // Trace its boundary, map to plane-local coords, simplify.
    let loop_cells = trace_boundary(&region, w, h)?;
    let ring: Vec<[f32; 2]> = loop_cells
        .iter()
        .map(|&(vx, vy)| {
            [
                a0 + (vx - border as f32) * res,
                b0 + (vy - border as f32) * res,
            ]
        })
        .collect();
    let simplified = simplify_closed(&ring, p.simplify_eps);
    if simplified.len() < 3 {
        return None;
    }
    Some(ensure_ccw(simplified))
}

/// Triangulate a simple polygon (CCW or CW, no holes) into triangle index
/// triples referencing the input vertices. Ear clipping — O(n²), fine for the
/// tens-of-vertices polygons here — so concave outlines triangulate correctly.
pub fn triangulate(poly: &[[f32; 2]]) -> Vec<[u32; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    // Work on a CCW copy so the convex-vertex test has a consistent sign; if we
    // flipped it, map indices back to the caller's ordering at the end.
    let ccw = signed_area(poly) >= 0.0;
    let idx_map: Vec<u32> = if ccw {
        (0..n as u32).collect()
    } else {
        (0..n as u32).rev().collect()
    };
    let verts: Vec<[f32; 2]> = idx_map.iter().map(|&i| poly[i as usize]).collect();

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0;
    while remaining.len() > 3 {
        let m = remaining.len();
        let mut clipped = false;
        for i in 0..m {
            let (ip, ic, in_) = (
                remaining[(i + m - 1) % m],
                remaining[i],
                remaining[(i + 1) % m],
            );
            if is_ear(&verts, &remaining, ip, ic, in_) {
                tris.push([idx_map[ip], idx_map[ic], idx_map[in_]]);
                remaining.remove(i);
                clipped = true;
                break;
            }
        }
        guard += 1;
        if !clipped || guard > n + 4 {
            break; // degenerate / self-touching; stop rather than loop forever
        }
    }
    if remaining.len() == 3 {
        tris.push([
            idx_map[remaining[0]],
            idx_map[remaining[1]],
            idx_map[remaining[2]],
        ]);
    }
    tris
}

/// Snap a plane-local polygon to a plane–plane intersection line so two planes
/// meet in a crisp edge. `line = (α, β, γ)` defines `f(a,b) = α·a + β·b − γ`;
/// the polygon is clipped to the half-plane containing its own centroid (the
/// bulk of the surface), removing the sliver that pokes across the line.
///
/// Guarded: the clip is applied **only** if it removes at most `max_frac` of the
/// area — a thin overshoot — so a real partition that genuinely divides the
/// surface (a large fraction on the far side) is left intact. Returns the
/// clipped polygon, or the original when the guard trips or the geometry is
/// degenerate.
pub fn snap_to_line(poly: &[[f32; 2]], line: (f32, f32, f32), max_frac: f32) -> Vec<[f32; 2]> {
    let (alpha, beta, gamma) = line;
    let scale = (alpha * alpha + beta * beta).sqrt();
    if poly.len() < 3 || scale < 1e-6 {
        return poly.to_vec();
    }
    let f = |p: [f32; 2]| (alpha * p[0] + beta * p[1] - gamma) / scale;

    // Centroid side decides which half to keep; ambiguous if centroid is on the
    // line (surface centered on it) — then don't cut.
    let c = centroid(poly);
    let cs = f(c);
    if cs.abs() < 1e-4 {
        return poly.to_vec();
    }
    let keep_positive = cs > 0.0;
    let inside = |p: [f32; 2]| (f(p) > 0.0) == keep_positive;

    // Sutherland–Hodgman against the single half-plane.
    let n = poly.len();
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(n + 4);
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let (ci, pi) = (inside(cur), inside(prev));
        if ci {
            if !pi {
                out.push(intersect(prev, cur, &f));
            }
            out.push(cur);
        } else if pi {
            out.push(intersect(prev, cur, &f));
        }
    }
    if out.len() < 3 {
        return poly.to_vec();
    }

    let (a_full, a_clip) = (signed_area(poly).abs(), signed_area(&out).abs());
    if a_full <= 0.0 || (a_full - a_clip) / a_full > max_frac {
        return poly.to_vec(); // too much removed → a real partition, not a sliver
    }
    out
}

fn centroid(poly: &[[f32; 2]]) -> [f32; 2] {
    let n = poly.len() as f32;
    let (mut cx, mut cy) = (0.0, 0.0);
    for p in poly {
        cx += p[0];
        cy += p[1];
    }
    [cx / n, cy / n]
}

fn intersect(p: [f32; 2], q: [f32; 2], f: &impl Fn([f32; 2]) -> f32) -> [f32; 2] {
    let (fp, fq) = (f(p), f(q));
    let t = fp / (fp - fq); // fp, fq have opposite signs here
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

// ---- occupancy-grid helpers -------------------------------------------------

fn dilate(occ: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = occ.to_vec();
    for y in 0..h {
        for x in 0..w {
            if occ[y * w + x] {
                continue;
            }
            let mut any = false;
            'n: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && occ[ny as usize * w + nx as usize] {
                        any = true;
                        break 'n;
                    }
                }
            }
            out[y * w + x] = any;
        }
    }
    out
}

fn erode(occ: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = occ.to_vec();
    for y in 0..h {
        for x in 0..w {
            if !occ[y * w + x] {
                continue;
            }
            let mut all = true;
            'n: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h || !occ[ny as usize * w + nx as usize] {
                        all = false;
                        break 'n;
                    }
                }
            }
            out[y * w + x] = all;
        }
    }
    out
}

/// Fill background cells not reachable from the border → interior holes solid.
fn fill_holes(occ: &mut [bool], w: usize, h: usize) {
    let mut reach = vec![false; w * h];
    let mut stack = Vec::new();
    for x in 0..w {
        stack.push((x, 0));
        stack.push((x, h - 1));
    }
    for y in 0..h {
        stack.push((0, y));
        stack.push((w - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        let i = y * w + x;
        if occ[i] || reach[i] {
            continue;
        }
        reach[i] = true;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    for i in 0..w * h {
        if !occ[i] && !reach[i] {
            occ[i] = true; // enclosed background = hole
        }
    }
}

/// Largest 4-connected component of occupied cells, as a bool mask.
fn largest_component(occ: &[bool], w: usize, h: usize) -> Option<Vec<bool>> {
    let mut label = vec![0u32; w * h];
    let mut best = (0usize, Vec::new());
    let mut next = 1u32;
    for start in 0..w * h {
        if !occ[start] || label[start] != 0 {
            continue;
        }
        let mut stack = vec![start];
        let mut cells = Vec::new();
        label[start] = next;
        while let Some(i) = stack.pop() {
            cells.push(i);
            let (x, y) = (i % w, i / w);
            let push = |nx: usize, ny: usize, stack: &mut Vec<usize>, label: &mut Vec<u32>| {
                let ni = ny * w + nx;
                if occ[ni] && label[ni] == 0 {
                    label[ni] = next;
                    stack.push(ni);
                }
            };
            if x > 0 {
                push(x - 1, y, &mut stack, &mut label);
            }
            if x + 1 < w {
                push(x + 1, y, &mut stack, &mut label);
            }
            if y > 0 {
                push(x, y - 1, &mut stack, &mut label);
            }
            if y + 1 < h {
                push(x, y + 1, &mut stack, &mut label);
            }
        }
        if cells.len() > best.0 {
            best = (cells.len(), cells);
        }
        next += 1;
    }
    if best.0 == 0 {
        return None;
    }
    let mut mask = vec![false; w * h];
    for i in best.1 {
        mask[i] = true;
    }
    Some(mask)
}

/// Trace the outer boundary of a filled region as an ordered ring of integer
/// grid-vertex coordinates. Emits, for every occupied/empty cell face, a
/// directed edge (occupied cell kept on a consistent side) and chains them into
/// a single closed loop — robust for a solid, simply-connected region.
fn trace_boundary(region: &[bool], w: usize, h: usize) -> Option<Vec<(f32, f32)>> {
    use std::collections::HashMap;
    let occ = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && region[y as usize * w + x as usize]
    };
    // Directed boundary edges: interior on the left. Vertex (x,y) is a grid
    // corner; cell (x,y) spans corners (x,y)..(x+1,y+1).
    let mut succ: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    let mut start = None;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !occ(x, y) {
                continue;
            }
            // (tail -> head) for each side whose neighbor is empty.
            if !occ(x, y - 1) {
                succ.insert((x, y), (x + 1, y)); // top: +x
            }
            if !occ(x + 1, y) {
                succ.insert((x + 1, y), (x + 1, y + 1)); // right: +y
            }
            if !occ(x, y + 1) {
                succ.insert((x + 1, y + 1), (x, y + 1)); // bottom: -x
            }
            if !occ(x - 1, y) {
                succ.insert((x, y + 1), (x, y)); // left: -y
            }
            if start.is_none() {
                start = Some((x, y));
            }
        }
    }
    let start = start?;
    let mut ring = Vec::new();
    let mut cur = start;
    for _ in 0..succ.len() + 1 {
        ring.push((cur.0 as f32, cur.1 as f32));
        match succ.get(&cur) {
            Some(&nxt) => {
                cur = nxt;
                if cur == start {
                    break;
                }
            }
            None => break,
        }
    }
    if ring.len() < 3 {
        None
    } else {
        Some(ring)
    }
}

// ---- polygon helpers --------------------------------------------------------

fn signed_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s
}

fn ensure_ccw(mut poly: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if signed_area(&poly) < 0.0 {
        poly.reverse();
    }
    poly
}

/// Douglas–Peucker on a closed ring: split at the vertex farthest from vertex 0,
/// simplify each half as a polyline, and stitch back into a ring.
fn simplify_closed(ring: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    let n = ring.len();
    if n < 4 {
        return ring.to_vec();
    }
    let mut far = 1;
    let mut far_d = -1.0;
    for i in 1..n {
        let d = dist2(ring[0], ring[i]);
        if d > far_d {
            far_d = d;
            far = i;
        }
    }
    let mut out = dp(&ring[0..=far], eps);
    out.pop(); // shared endpoint
    let mut second: Vec<[f32; 2]> = ring[far..].to_vec();
    second.push(ring[0]);
    let mut tail = dp(&second, eps);
    tail.pop();
    out.append(&mut tail);
    out
}

fn dp(pts: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let (a, b) = (pts[0], pts[n - 1]);
    let (mut idx, mut dmax) = (0, 0.0);
    for (i, &pt) in pts.iter().enumerate().take(n - 1).skip(1) {
        let d = perp_dist(pt, a, b);
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }
    if dmax > eps {
        let mut left = dp(&pts[..=idx], eps);
        let right = dp(&pts[idx..], eps);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![a, b]
    }
}

fn perp_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        return dist2(p, a).sqrt();
    }
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    dx * dx + dy * dy
}

fn is_ear(verts: &[[f32; 2]], remaining: &[usize], ip: usize, ic: usize, in_: usize) -> bool {
    let (a, b, c) = (verts[ip], verts[ic], verts[in_]);
    // Convex (CCW) vertex?
    if cross(a, b, c) <= 0.0 {
        return false;
    }
    // No other remaining vertex inside triangle abc.
    for &r in remaining {
        if r == ip || r == ic || r == in_ {
            continue;
        }
        if point_in_tri(verts[r], a, b, c) {
            return false;
        }
    }
    true
}

fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn point_in_tri(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = cross(p, a, b);
    let d2 = cross(p, b, c);
    let d3 = cross(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense fill of a rectangle → 4-ish vertex polygon covering ~full area.
    #[test]
    fn rectangle_footprint_outlines_a_quad() {
        let mut pts = Vec::new();
        let mut a = 0.0;
        while a <= 2.0 {
            let mut b = 0.0;
            while b <= 1.0 {
                pts.push([a, b]);
                b += 0.04;
            }
            a += 0.04;
        }
        let poly = footprint_polygon(&pts, &PolyParams::default()).expect("polygon");
        assert!(poly.len() >= 4 && poly.len() <= 8, "verts: {}", poly.len());
        let area = signed_area(&poly).abs();
        assert!((area - 2.0).abs() < 0.5, "area {area} far from 2.0");
    }

    /// An L-shaped fill must stay concave: its area is well under the bounding
    /// box (a rectangle fit would report the full 2×2 = 4).
    #[test]
    fn l_shape_stays_concave() {
        let mut pts = Vec::new();
        let mut a = 0.0;
        while a <= 2.0 {
            let mut b = 0.0;
            while b <= 2.0 {
                // Remove the top-right quadrant → an L.
                if !(a > 1.0 && b > 1.0) {
                    pts.push([a, b]);
                }
                b += 0.04;
            }
            a += 0.04;
        }
        let poly = footprint_polygon(&pts, &PolyParams::default()).expect("polygon");
        let area = signed_area(&poly).abs();
        assert!(area < 3.5, "area {area} — outline is not concave (L collapsed to box)");
        assert!(area > 2.5, "area {area} — outline lost too much of the L");

        // Triangulation covers the same area (fan would fail on the concavity).
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
        let tri_area: f32 = tris
            .iter()
            .map(|t| {
                0.5 * cross(
                    poly[t[0] as usize],
                    poly[t[1] as usize],
                    poly[t[2] as usize],
                )
                .abs()
            })
            .sum();
        assert!((tri_area - area).abs() < 0.1, "tri area {tri_area} vs poly {area}");
    }

    #[test]
    fn snap_trims_a_sliver_to_the_line() {
        // Unit square; line a = 0.9. Centroid at a=0.5 → keep a<0.9, drop the
        // 10% strip. Result terminates on the line.
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let out = snap_to_line(&sq, (1.0, 0.0, 0.9), 0.35);
        let max_a = out.iter().fold(f32::MIN, |m, p| m.max(p[0]));
        assert!(max_a <= 0.9 + 1e-4, "not clipped to line: max_a={max_a}");
        assert!((signed_area(&out).abs() - 0.9).abs() < 0.02, "area off");
    }

    #[test]
    fn snap_preserves_a_real_partition() {
        // Line through the middle (a = 0.5) would remove ~half → a genuine
        // partition, not a sliver. Guard must leave the polygon untouched.
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let out = snap_to_line(&sq, (1.0, 0.0, 0.5), 0.35);
        assert_eq!(out.len(), sq.len());
        assert!((signed_area(&out).abs() - 1.0).abs() < 1e-4, "should be unchanged");
    }

    #[test]
    fn snap_ignores_a_far_line() {
        // Line well outside the polygon removes nothing → returned as-is.
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let out = snap_to_line(&sq, (1.0, 0.0, 5.0), 0.35);
        assert!((signed_area(&out).abs() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sparse_or_degenerate_returns_none() {
        assert!(footprint_polygon(&[[0.0, 0.0], [0.1, 0.1]], &PolyParams::default()).is_none());
        // Colinear thin strip → degenerate.
        let strip: Vec<[f32; 2]> = (0..50).map(|i| [i as f32 * 0.05, 0.0]).collect();
        assert!(footprint_polygon(&strip, &PolyParams::default()).is_none());
    }
}
