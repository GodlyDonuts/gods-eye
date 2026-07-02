//! Persistent world-plane map.
//!
//! Per-frame [`Segment`]s (camera frame) are lifted to world space by
//! transforming their [`Moments`] through the camera-to-world pose, then
//! associated with existing world planes by normal (tight) + offset (loose) and
//! fused by summing moments — giving √N cross-frame noise reduction with no
//! point storage. Each confirmed plane renders as a single oriented rectangle
//! (2 triangles): the low-poly output.

use ge_mesh::Mesh;
use glam::{Affine3A, Quat, Vec3};

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
    /// [`refine_pose`](WorldPlaneRegistry::refine_pose) only aligns to map planes
    /// with at least this many observations — a mature, stable reference, so a
    /// freshly-seen plane's rough geometry can't yank the pose.
    pub refine_min_obs: u32,
    /// Fraction of the computed map correction actually applied per frame. A
    /// gentle exponential pull toward the map: ≈0 when VO already agrees (no
    /// harm), accumulating only against genuine, persistent drift.
    pub refine_blend: f32,
}

impl Default for RegistryParams {
    fn default() -> Self {
        Self {
            normal_cos: 0.95, // ~18°, looser than per-frame (poses are noisier)
            offset_tol: 0.12, // loose: monocular offset/scale is fragile
            confirm_after: 3,
            max_footprint: 2000,
            refine_min_obs: 8,
            refine_blend: 0.3,
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

    /// Refine a predicted camera pose by aligning the current frame's detected
    /// planes to the confirmed world planes — frame-to-map plane registration.
    ///
    /// This is the drift brake: the persistent map is globally consistent, so
    /// snapping each frame's planes onto it re-anchors the pose every frame and,
    /// on returning to a mapped area, pulls the camera back onto the existing
    /// walls instead of laying down doubled ones. Run it *before* [`observe`] and
    /// feed the result back to the tracker so the map stays self-consistent.
    ///
    /// Returns the corrected `cam_to_world`, or `None` when too few or too
    /// parallel planes match to constrain a correction (VO stays in charge).
    pub fn refine_pose(&self, predicted: &Affine3A, segments: &[Segment]) -> Option<Affine3A> {
        // Associate each detected segment (posed by `predicted`) with a confirmed
        // world plane, by normal (tight) + offset (loose) — as in `observe`.
        struct Pair {
            n_p: Vec3, // predicted-plane normal (world)
            c_p: Vec3, // predicted-plane centroid (world)
            n_r: Vec3, // matched registry-plane normal (oriented to n_p)
            d_r: f32,  // matched registry-plane offset (oriented)
        }
        let mut pairs: Vec<Pair> = Vec::new();
        for seg in segments {
            let wm = seg.moments.transform(predicted);
            let (Some((wp, _)), Some(c)) = (wm.fit(), wm.centroid()) else {
                continue;
            };
            for rp in self
                .planes
                .iter()
                .filter(|p| p.confirmed && p.observations >= self.params.refine_min_obs)
            {
                let dot = wp.normal.dot(rp.plane.normal);
                if dot.abs() > self.params.normal_cos
                    && rp.plane.signed_distance(c).abs() < self.params.offset_tol
                {
                    // Orient the registry plane into the same hemisphere as wp.
                    let sign = dot.signum();
                    pairs.push(Pair {
                        n_p: wp.normal,
                        c_p: c,
                        n_r: rp.plane.normal * sign,
                        d_r: rp.plane.offset * sign,
                    });
                    break;
                }
            }
        }
        if pairs.len() < 2 {
            return None;
        }
        // Observability: the matched normals must span at least two directions,
        // else translation is under-constrained (one wall can't fix all axes).
        let mut s = [0.0f64; 6]; // Σ nᵣnᵣᵀ upper triangle [xx,xy,xz,yy,yz,zz]
        for p in &pairs {
            let n = p.n_r;
            s[0] += (n.x * n.x) as f64;
            s[1] += (n.x * n.y) as f64;
            s[2] += (n.x * n.z) as f64;
            s[3] += (n.y * n.y) as f64;
            s[4] += (n.y * n.z) as f64;
            s[5] += (n.z * n.z) as f64;
        }
        let (lam_min, _) = crate::eigen::smallest_eigen(s[0], s[1], s[2], s[3], s[4], s[5]);
        if lam_min < 0.05 {
            return None; // normals ~parallel: cannot fix in-plane translation
        }

        // Gauss-Newton for the world-frame correction `delta` (corrected = delta ·
        // predicted) minimising, per pair, normal misalignment + point-to-plane
        // offset. Damped for the directions the geometry leaves weak.
        const W_N: f64 = 1.0; // normal-alignment weight
        const W_D: f64 = 1.0; // offset weight
        let mut delta = Affine3A::IDENTITY;
        for _ in 0..6 {
            let mut ata = [[0.0f64; 6]; 6];
            let mut atb = [0.0f64; 6];
            let mut acc = |j: &[f64; 6], e: f64, w: f64| {
                for r in 0..6 {
                    atb[r] += w * j[r] * e;
                    for c in 0..6 {
                        ata[r][c] += w * j[r] * j[c];
                    }
                }
            };
            for p in &pairs {
                let n_p = (delta.matrix3 * p.n_p).normalize();
                let c_p = delta.transform_point3(p.c_p);
                // (a) Normal alignment: want ω×n_p = (n_r − n_p). J_ω = −skew(n_p).
                let r_n = p.n_r - n_p;
                let sk = [
                    [0.0, -n_p.z, n_p.y],
                    [n_p.z, 0.0, -n_p.x],
                    [-n_p.y, n_p.x, 0.0],
                ];
                for (row, rn) in [r_n.x, r_n.y, r_n.z].into_iter().enumerate() {
                    let j = [
                        -sk[row][0] as f64,
                        -sk[row][1] as f64,
                        -sk[row][2] as f64,
                        0.0,
                        0.0,
                        0.0,
                    ];
                    acc(&j, rn as f64, W_N);
                }
                // (b) Offset: want n_r·(ω×c_p + τ) = d_r − n_r·c_p.
                let r_d = (p.d_r - p.n_r.dot(c_p)) as f64;
                let jw = c_p.cross(p.n_r);
                let j = [
                    jw.x as f64,
                    jw.y as f64,
                    jw.z as f64,
                    p.n_r.x as f64,
                    p.n_r.y as f64,
                    p.n_r.z as f64,
                ];
                acc(&j, r_d, W_D);
            }
            for (k, row) in ata.iter_mut().enumerate() {
                row[k] += 1e-3;
            }
            let Some(x) = solve6(&ata, &atb) else { break };
            let omega = Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
            let tau = Vec3::new(x[3] as f32, x[4] as f32, x[5] as f32);
            let step = Affine3A::from_rotation_translation(Quat::from_scaled_axis(omega), tau);
            delta = step * delta;
            if omega.length() < 1e-6 && tau.length() < 1e-6 {
                break;
            }
        }

        // Blend: apply only a fraction of the correction so per-frame plane noise
        // can't jerk the pose, and the map acts as a gentle exponential drift
        // brake rather than a hard snap.
        let axis_angle = Quat::from_mat3a(&delta.matrix3).normalize().to_scaled_axis();
        let blend = self.params.refine_blend;
        let b_omega = axis_angle * blend;
        let b_tau = delta.translation * blend;

        // Deadband: VO already agrees with the map → leave it untouched.
        if b_omega.length() < 1e-4 && b_tau.length() < 1e-4 {
            return None;
        }
        // Reject an implausibly large correction (a bad association) — VO is
        // trusted over a wild map snap.
        if Vec3::from(b_tau).length() > 0.2 {
            return None;
        }
        let b_delta =
            Affine3A::from_rotation_translation(Quat::from_scaled_axis(b_omega), Vec3::from(b_tau));
        Some(b_delta * *predicted)
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

/// Solve the symmetric positive-definite system `A x = b` (6×6) via Cholesky.
/// Returns `None` if `A` is not positive-definite (degenerate constraints).
#[allow(clippy::needless_range_loop)] // triangular index math is clearest as-is
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut l = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let mut y = [0.0f64; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0f64; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
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
