//! Per-frame plane detection from an organized metric depth image.
//!
//! CAPE-inspired: tile the depth into cells, fit a plane to each cell by PCA
//! (which averages out per-pixel noise), reject non-planar / depth-discontinuous
//! cells, then cluster cells by plane parameters into a few large planar
//! segments. Each segment carries its raw [`Moments`] (so it can be lifted to
//! world space and fused) plus its cell centroids (a cheap footprint for
//! bounding the rendered rectangle).

use ge_backend_trait::{DepthMap, Intrinsics};
use glam::Vec3;

use crate::{Moments, Plane};

/// A planar segment in the camera frame.
#[derive(Clone)]
pub struct Segment {
    pub moments: Moments,
    pub plane: Plane,
    pub flatness: f32,
    pub cell_count: usize,
    /// One centroid per contributing cell (camera frame) — the footprint.
    pub cell_centroids: Vec<Vec3>,
}

/// Detection tuning. Defaults are loosened for noisy monocular depth.
#[derive(Clone, Copy, Debug)]
pub struct DetectParams {
    pub cell: usize,
    pub min_cell_points: usize,
    /// Cell is planar if its flatness (variance) < (sigma_k · mean_depth)².
    pub sigma_k: f32,
    /// Reject a cell whose intra-cell depth spread exceeds this fraction of its
    /// mean depth (flying pixels / edges).
    pub jump_ratio: f32,
    /// Cells join a cluster if normals agree above this cosine …
    pub normal_cos: f32,
    /// … and the cell centroid is within this distance (m) of the cluster plane.
    pub offset_tol: f32,
    pub min_cells: usize,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            cell: 12,
            min_cell_points: 24,
            sigma_k: 0.02,
            // Loose: grazing planes (floors) legitimately span a depth range
            // within a cell; the 3D flatness test catches real edges/steps.
            jump_ratio: 0.5,
            normal_cos: 0.965, // ~15°
            offset_tol: 0.06,
            min_cells: 4,
            min_depth: 0.2,
            max_depth: 6.0,
        }
    }
}

struct Cell {
    moments: Moments,
    plane: Plane,
    centroid: Vec3,
    points: usize,
}

/// Detect planar segments in a metric depth frame (camera coordinates).
pub fn detect_planes(depth: &DepthMap, intr: &Intrinsics, p: &DetectParams) -> Vec<Segment> {
    let w = depth.width as usize;
    let h = depth.height as usize;
    let cells_x = w / p.cell;
    let cells_y = h / p.cell;

    let mut cells: Vec<Cell> = Vec::new();
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let mut m = Moments::new();
            let mut points = 0usize;
            let (mut dmin, mut dmax) = (f32::MAX, f32::MIN);
            for dy in 0..p.cell {
                for dx in 0..p.cell {
                    let (u, v) = (cx * p.cell + dx, cy * p.cell + dy);
                    let d = depth.depth_m[v * w + u];
                    if d.is_finite() && d > p.min_depth && d < p.max_depth {
                        let pt = intr.unproject(u as f32, v as f32, d);
                        m.add_point(1.0 / (d * d) as f64, pt);
                        points += 1;
                        dmin = dmin.min(d);
                        dmax = dmax.max(d);
                    }
                }
            }
            if points < p.min_cell_points {
                continue;
            }
            let mean_d = 0.5 * (dmin + dmax);
            if dmax - dmin > p.jump_ratio * mean_d {
                continue; // depth discontinuity / flying pixels
            }
            let Some((plane, flat)) = m.fit() else {
                continue;
            };
            let sigma = p.sigma_k * mean_d;
            if flat > sigma * sigma {
                continue; // not planar enough
            }
            let Some(centroid) = m.centroid() else {
                continue;
            };
            cells.push(Cell {
                moments: m,
                plane,
                centroid,
                points,
            });
        }
    }

    // Greedily cluster cells by plane parameters; biggest cells seed first.
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&a, &b| cells[b].points.cmp(&cells[a].points));

    struct Cluster {
        moments: Moments,
        plane: Plane,
        cells: usize,
        centroids: Vec<Vec3>,
    }
    let mut clusters: Vec<Cluster> = Vec::new();
    for i in order {
        let c = &cells[i];
        let mut joined = None;
        for (k, cl) in clusters.iter().enumerate() {
            if c.plane.normal.dot(cl.plane.normal).abs() > p.normal_cos
                && cl.plane.signed_distance(c.centroid).abs() < p.offset_tol
            {
                joined = Some(k);
                break;
            }
        }
        match joined {
            Some(k) => {
                clusters[k].moments.merge(&c.moments);
                clusters[k].cells += 1;
                clusters[k].centroids.push(c.centroid);
                if let Some((pl, _)) = clusters[k].moments.fit() {
                    clusters[k].plane = pl;
                }
            }
            None => clusters.push(Cluster {
                moments: c.moments.clone(),
                plane: c.plane,
                cells: 1,
                centroids: vec![c.centroid],
            }),
        }
    }

    clusters
        .into_iter()
        .filter(|c| c.cells >= p.min_cells)
        .filter_map(|c| {
            let (plane, flatness) = c.moments.fit()?;
            Some(Segment {
                moments: c.moments,
                plane,
                flatness,
                cell_count: c.cells,
                cell_centroids: c.centroids,
            })
        })
        .collect()
}
