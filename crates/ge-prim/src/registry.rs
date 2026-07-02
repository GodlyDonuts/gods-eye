//! Persistent world-plane map.
//!
//! Per-frame [`Segment`]s (camera frame) are lifted to world space by
//! transforming their [`Moments`] through the camera-to-world pose, then
//! associated with existing world planes by normal (tight) + offset (loose) and
//! fused by summing moments — giving √N cross-frame noise reduction with no
//! point storage. Each confirmed plane renders as a single oriented rectangle
//! (2 triangles): the low-poly output.

use ge_mesh::Mesh;
use glam::{Affine3A, Vec3};

use crate::{Moments, Plane, Segment};

struct WorldPlane {
    moments: Moments,
    plane: Plane,
    /// World-space cell centroids (capped) bounding the rendered rectangle.
    footprint: Vec<Vec3>,
    observations: u32,
    confirmed: bool,
}

/// Tuning for cross-frame plane association + confirmation.
#[derive(Clone, Copy, Debug)]
pub struct RegistryParams {
    pub normal_cos: f32,
    pub offset_tol: f32,
    pub confirm_after: u32,
    pub max_footprint: usize,
}

impl Default for RegistryParams {
    fn default() -> Self {
        Self {
            normal_cos: 0.95, // ~18°, looser than per-frame (poses are noisier)
            offset_tol: 0.12, // loose: monocular offset/scale is fragile
            confirm_after: 3,
            max_footprint: 2000,
        }
    }
}

/// The persistent low-poly plane map.
#[derive(Default)]
pub struct WorldPlaneRegistry {
    planes: Vec<WorldPlane>,
    params: RegistryParams,
}

impl WorldPlaneRegistry {
    pub fn new(params: RegistryParams) -> Self {
        Self {
            planes: Vec::new(),
            params,
        }
    }

    /// Number of confirmed (rendered) planes.
    pub fn confirmed_count(&self) -> usize {
        self.planes.iter().filter(|p| p.confirmed).count()
    }

    /// Confirmed world planes as `(plane, observation_count)`.
    pub fn confirmed_planes(&self) -> Vec<(Plane, u32)> {
        self.planes
            .iter()
            .filter(|p| p.confirmed)
            .map(|p| (p.plane, p.observations))
            .collect()
    }

    /// Fuse a frame's detected segments into the world map.
    pub fn observe(&mut self, segments: &[Segment], cam_to_world: &Affine3A) {
        for seg in segments {
            let world_moments = seg.moments.transform(cam_to_world);
            let Some((world_plane, _)) = world_moments.fit() else {
                continue;
            };
            let Some(centroid) = world_moments.centroid() else {
                continue;
            };

            let mut matched = None;
            for (k, wp) in self.planes.iter().enumerate() {
                if world_plane.normal.dot(wp.plane.normal).abs() > self.params.normal_cos
                    && wp.plane.signed_distance(centroid).abs() < self.params.offset_tol
                {
                    matched = Some(k);
                    break;
                }
            }

            let world_centroids: Vec<Vec3> = seg
                .cell_centroids
                .iter()
                .map(|c| cam_to_world.transform_point3(*c))
                .collect();

            match matched {
                Some(k) => {
                    let wp = &mut self.planes[k];
                    wp.moments.merge(&world_moments);
                    if let Some((pl, _)) = wp.moments.fit() {
                        wp.plane = pl;
                    }
                    wp.observations += 1;
                    if wp.observations >= self.params.confirm_after {
                        wp.confirmed = true;
                    }
                    wp.footprint.extend(world_centroids);
                    if wp.footprint.len() > self.params.max_footprint {
                        let drop = wp.footprint.len() - self.params.max_footprint;
                        wp.footprint.drain(0..drop);
                    }
                }
                None => self.planes.push(WorldPlane {
                    moments: world_moments,
                    plane: world_plane,
                    footprint: world_centroids,
                    observations: 1,
                    confirmed: 1 >= self.params.confirm_after,
                }),
            }
        }
    }

    /// Low-poly mesh: each confirmed plane as its true outline polygon (its real
    /// shape — an L-shaped floor stays L-shaped), snapped so adjacent planes meet
    /// in crisp edges, then triangulated. Falls back to an oriented bounding
    /// rectangle when a footprint is too sparse to outline.
    pub fn to_mesh(&self) -> Mesh {
        let poly_params = crate::PolyParams::default();

        // Phase 1 — build each confirmed plane's outline polygon (plane-local).
        struct Poly {
            plane: Plane,
            u: Vec3,
            v: Vec3,
            pts: Vec<[f32; 2]>,
        }
        let mut polys: Vec<Poly> = Vec::new();
        for wp in self.planes.iter().filter(|p| p.confirmed) {
            if wp.footprint.len() < 3 {
                continue;
            }
            let (u, v) = wp.plane.basis();
            let pts2d: Vec<[f32; 2]> = wp.footprint.iter().map(|p| [p.dot(u), p.dot(v)]).collect();
            let poly = crate::footprint_polygon(&pts2d, &poly_params)
                .unwrap_or_else(|| oriented_rect(&pts2d).to_vec());
            polys.push(Poly {
                plane: wp.plane,
                u,
                v,
                pts: poly,
            });
        }

        // Phase 2 — snap each polygon to its neighbours' intersection lines so
        // adjacent planes meet in a crisp edge. Only near, non-parallel planes
        // participate; snap_to_line itself removes only thin slivers.
        const PARALLEL_COS: f32 = 0.98; // planes within ~11° are "parallel" → no edge
        const SNAP_DIST: f32 = 0.15; // line must pass within 15 cm of the polygon
        const MAX_FRAC: f32 = 0.35; // never cut more than this fraction (partition guard)
        let bases: Vec<(Plane, Vec3, Vec3)> = polys.iter().map(|p| (p.plane, p.u, p.v)).collect();
        for i in 0..polys.len() {
            let (pi, ui, vi) = bases[i];
            for &(pj, _, _) in bases.iter() {
                if pi.normal.dot(pj.normal).abs() > PARALLEL_COS {
                    continue;
                }
                // Intersection line of plane j, expressed in plane i's (u,v):
                //   (n_j·u_i) a + (n_j·v_i) b = d_j − (n_j·n_i) d_i
                let alpha = pj.normal.dot(ui);
                let beta = pj.normal.dot(vi);
                let gamma = pj.offset - pj.normal.dot(pi.normal) * pi.offset;
                let scale = (alpha * alpha + beta * beta).sqrt();
                if scale < 1e-6 {
                    continue;
                }
                let near = polys[i]
                    .pts
                    .iter()
                    .any(|&p| ((alpha * p[0] + beta * p[1] - gamma) / scale).abs() < SNAP_DIST);
                if near {
                    polys[i].pts = crate::snap_to_line(&polys[i].pts, (alpha, beta, gamma), MAX_FRAC);
                }
            }
        }

        // Phase 3 — triangulate and emit with per-vertex plane normals.
        let mut mesh = Mesh::default();
        for p in &polys {
            let tris = crate::triangulate(&p.pts);
            if tris.is_empty() {
                continue;
            }
            let base = mesh.positions.len() as u32;
            let n = [p.plane.normal.x, p.plane.normal.y, p.plane.normal.z];
            for &[a, b] in &p.pts {
                let pw = p.plane.point_from_uv(a, b, p.u, p.v);
                mesh.positions.push([pw.x, pw.y, pw.z]);
                mesh.normals.push(n);
            }
            for t in tris {
                mesh.indices
                    .extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
            }
        }
        mesh
    }
}

/// Minimum-ish-area oriented rectangle of 2D points via principal axes (PCA):
/// the AABB in the data's own principal frame, returned as 4 corners (CCW).
fn oriented_rect(pts: &[[f32; 2]]) -> [[f32; 2]; 4] {
    let n = pts.len() as f32;
    let (mut cx, mut cy) = (0.0f32, 0.0f32);
    for p in pts {
        cx += p[0];
        cy += p[1];
    }
    cx /= n;
    cy /= n;
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for p in pts {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (c, s) = (theta.cos(), theta.sin());
    let (mut amin, mut amax, mut bmin, mut bmax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in pts {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        let ra = dx * c + dy * s;
        let rb = -dx * s + dy * c;
        amin = amin.min(ra);
        amax = amax.max(ra);
        bmin = bmin.min(rb);
        bmax = bmax.max(rb);
    }
    let back = |ra: f32, rb: f32| [cx + ra * c - rb * s, cy + ra * s + rb * c];
    [
        back(amin, bmin),
        back(amax, bmin),
        back(amax, bmax),
        back(amin, bmax),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{detect_planes, DetectParams};
    use ge_backend_trait::{DepthMap, Intrinsics};
    use glam::Quat;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 9) as f32 / 8_388_608.0 - 1.0
    }

    /// Render a noisy depth image of a set of world planes from a camera pose.
    fn render_depth(
        planes: &[Plane],
        intr: &Intrinsics,
        cam_to_world: &Affine3A,
        seed: &mut u32,
    ) -> DepthMap {
        let (w, h) = (intr.width as usize, intr.height as usize);
        let mut depth = vec![0.0f32; w * h];
        let o = cam_to_world.translation;
        let ow = Vec3::new(o.x, o.y, o.z);
        for v in 0..h {
            for u in 0..w {
                let x = (u as f32 - intr.cx) / intr.fx;
                let y = (v as f32 - intr.cy) / intr.fy;
                let dir = cam_to_world.transform_vector3(Vec3::new(x, y, 1.0));
                let mut best = f32::MAX;
                for pl in planes {
                    let denom = pl.normal.dot(dir);
                    if denom.abs() < 1e-6 {
                        continue;
                    }
                    let s = (pl.offset - pl.normal.dot(ow)) / denom;
                    if s > 0.05 && s < best {
                        best = s;
                    }
                }
                if best < f32::MAX {
                    depth[v * w + u] = best + lcg(seed) * 0.01; // ~1 cm noise
                }
            }
        }
        DepthMap {
            width: intr.width,
            height: intr.height,
            depth_m: depth,
            confidence: None,
        }
    }

    #[test]
    fn synthetic_room_converges_to_three_planes() {
        // World: floor y=0.7, wall ahead z=2.5, left wall x=-1.0. Wide FOV so
        // all three are amply visible from a forward-looking camera.
        let truth = [
            Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                offset: 0.7,
            },
            Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                offset: 2.5,
            },
            Plane {
                normal: Vec3::new(1.0, 0.0, 0.0),
                offset: -1.0,
            },
        ];
        let intr = Intrinsics {
            fx: 70.0,
            fy: 70.0,
            cx: 80.0,
            cy: 60.0,
            width: 160,
            height: 120,
        };
        let params = DetectParams {
            cell: 10,
            min_cell_points: 40,
            sigma_k: 0.02,
            jump_ratio: 0.5,
            normal_cos: 0.95,
            offset_tol: 0.08,
            min_cells: 3,
            min_depth: 0.2,
            max_depth: 6.0,
        };

        let mut reg = WorldPlaneRegistry::new(RegistryParams::default());
        let mut seed = 7u32;
        // Slow yaw sweep, repeated so planes get confirmed.
        let yaws = [-0.08f32, -0.04, 0.0, 0.04, 0.08, 0.04, 0.0, -0.04];
        for &yaw in &yaws {
            let pose = Affine3A::from_quat(Quat::from_rotation_y(yaw));
            let depth = render_depth(&truth, &intr, &pose, &mut seed);
            let segs = detect_planes(&depth, &intr, &params);
            reg.observe(&segs, &pose);
        }

        let confirmed = reg.confirmed_planes();
        assert!(
            (3..=6).contains(&confirmed.len()),
            "expected ~3 planes, got {}",
            confirmed.len()
        );
        // Every true plane has a matching confirmed world plane.
        for t in &truth {
            let matched = confirmed.iter().any(|(p, _)| {
                p.normal.dot(t.normal).abs() > 0.97 && (p.offset.abs() - t.offset.abs()).abs() < 0.1
            });
            assert!(matched, "no plane matched truth {:?}", t.normal);
        }
        // The map stays tiny.
        assert!(confirmed.len() <= 6);

        // to_mesh() produces a valid, non-empty low-poly mesh: normals present
        // per vertex, indices in range, triangles non-degenerate.
        let mesh = reg.to_mesh();
        assert!(mesh.triangle_count() > 0, "empty mesh");
        assert_eq!(mesh.normals.len(), mesh.positions.len(), "missing normals");
        let nv = mesh.positions.len() as u32;
        assert!(mesh.indices.iter().all(|&i| i < nv), "index out of range");
        // Each plane contributes at least one triangle (>= 3 planes worth).
        assert!(mesh.triangle_count() >= 3, "too few triangles");
    }
}
