//! Shared OpenSubdiv-oracle machinery for the limit and patch oracle
//! gates (`tests/limit_osd_oracle.rs`, `tests/patch_osd_oracle.rs`),
//! plus the in-crate brute-force closest-point oracle of
//! `tests/closest_point.rs`.
//!
//! Each integration-test binary compiles this module independently, so
//! not every helper is used by every binary.
#![allow(dead_code)]

use opensubdiv_petite::far;
use subdiv_kernels::Mesh;

/// One comparison case, shared verbatim by both refiners.
pub struct Case {
    pub positions: Vec<[f32; 3]>,
    /// All-quad faces, flat (4 corners per face).
    pub face_vertices: Vec<u32>,
    /// Creased edges as vertex pairs (infinitely sharp on both sides:
    /// `f32::INFINITY` for our refiner, OpenSubdiv's `10.0` sentinel
    /// for petite).
    pub crease_pairs: Vec<[u32; 2]>,
    /// Infinitely sharp (pinned) vertices, same sharpness encoding.
    pub corner_vertices: Vec<u32>,
    /// Petite boundary mode matching our default `EdgesOnly` where the
    /// mesh is open; `None` (petite's default) for closed meshes.
    pub petite_boundary: Option<far::BoundaryInterpolation>,
    /// Our scheme options (the petite side is always OpenSubdiv, so
    /// cases with corner-rule vertices must select `OpenSubdivDeRose`).
    pub options: subdiv_kernels::SchemeOptions,
}

impl Case {
    /// Number of (all-quad) cage faces; each is one ptex face.
    pub fn face_count(&self) -> usize {
        self.face_vertices.len() / 4
    }

    /// The case as this crate's refiner input.
    pub fn topology(&self) -> Mesh {
        let edge_vertices = self.crease_pairs.clone();
        let edge_creases = vec![f32::INFINITY; edge_vertices.len()];
        let mut vertex_corners = vec![0.0; self.positions.len()];
        for &vi in &self.corner_vertices {
            vertex_corners[vi as usize] = f32::INFINITY;
        }
        Mesh {
            vertex_count: self.positions.len() as u32,
            face_vertex_counts: vec![4; self.face_count()],
            face_vertex_indices: self.face_vertices.clone(),
            edge_vertices,
            edge_creases,
            vertex_corners,
        }
    }
}

pub fn cube_case(creased: bool) -> Case {
    Case {
        positions: vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        face_vertices: vec![
            0, 3, 2, 1, // z = -1, seen from below.
            4, 5, 6, 7, // z = +1.
            0, 1, 5, 4, // y = -1.
            1, 2, 6, 5, // x = +1.
            2, 3, 7, 6, // y = +1.
            3, 0, 4, 7, // x = -1.
        ],
        crease_pairs: if creased {
            vec![[4, 5], [5, 6], [6, 7], [7, 4]]
        } else {
            Vec::new()
        },
        corner_vertices: Vec::new(),
        petite_boundary: None,
        // The petite/oracle side does OpenSubdiv (DeRose) creasing, which
        // auto-corners vertices with 3+ sharp edges — the same rule the
        // refiner uses. Consistent with `spoked_cube_case`.
        options: subdiv_kernels::SchemeOptions {
            corner_rule: subdiv_kernels::CornerRule::OpenSubdivDeRose,
            ..Default::default()
        },
    }
}

/// Three quads fanning around a boundary center vertex: the center has
/// four boundary-fan edges and three faces, so its cross tangent is the
/// Biermann irregular case; the outer corners are single-face crease
/// corners and the spoke rim is the regular B-spline crease. Warped in
/// z so tangent weights actually matter.
pub fn fan_case() -> Case {
    let ring = |angle_deg: f64, radius: f64| {
        let a = angle_deg.to_radians();
        let (x, y) = (radius * a.cos(), radius * a.sin());
        let z = 0.3 * x * x - 0.25 * y + 0.1 * x;
        [x as f32, y as f32, z as f32]
    };
    Case {
        // 0 = center; 1..=4 = spokes a0..a3; 5..=7 = outer corners.
        positions: vec![
            [0.0, 0.0, 0.0],
            ring(0.0, 1.0),
            ring(60.0, 1.0),
            ring(120.0, 1.0),
            ring(180.0, 1.0),
            ring(30.0, 1.5),
            ring(90.0, 1.5),
            ring(150.0, 1.5),
        ],
        face_vertices: vec![0, 1, 5, 2, 0, 2, 6, 3, 0, 3, 7, 4],
        crease_pairs: Vec::new(),
        corner_vertices: Vec::new(),
        petite_boundary: Some(far::BoundaryInterpolation::EdgeOnly),
        // OSD-compared case: select the DeRose path the petite side uses
        // (see `cube_case`).
        options: subdiv_kernels::SchemeOptions {
            corner_rule: subdiv_kernels::CornerRule::OpenSubdivDeRose,
            ..Default::default()
        },
    }
}

/// Once-refined cube (24 quads, closed) with three of the four edges at
/// the top face-center infinitely sharp: that vertex takes the corner
/// rule (pinned under `OpenSubdivDeRose`, matching OpenSubdiv) and its
/// ring splits into two single-face sectors and one two-face sector;
/// the spoke interiors are crease vertices. The spoke far ends are
/// pinned with infinite *vertex* sharpness: left as darts, OpenSubdiv's
/// `infintely_sharp_patch` end caps shift their limit position by
/// ~1.3e-3 (the crease would terminate mid-surface, which the
/// inf-sharp patch approximation does not reproduce), so the oracle
/// could not match positions there. Returns the case and the pinned
/// center's cage vertex index.
pub fn spoked_cube_case() -> (Case, u32) {
    use std::num::NonZeroU8;
    use subdiv_kernels::{
        CornerRule, Mesh, Refiner, Scheme, SchemeOptions, UniformRefine, VertexOrigin,
    };

    let base = cube_case(false);
    let topo = Mesh {
        vertex_count: base.positions.len() as u32,
        face_vertex_counts: vec![4; base.face_vertices.len() / 4],
        face_vertex_indices: base.face_vertices.clone(),
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; base.positions.len()],
    };
    let refiner =
        Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(NonZeroU8::new(1).expect("non-zero")))
        .expect("cage refinement");
    let positions = result.interpolate(&base.positions);

    let center = result
        .lineage
        .vertex_origin
        .iter()
        .position(|origin| *origin == VertexOrigin::Face(1))
        .expect("face point of base face 1") as u32;
    let estart = result.adjacency.vert_edge_offsets[center as usize] as usize;
    let crease_pairs: Vec<[u32; 2]> = result.adjacency.vert_edges[estart..estart + 3]
        .iter()
        .map(|&ei| {
            let [a, b] = result.topology.edge_vertices[ei as usize];
            [center, if a == center { b } else { a }]
        })
        .collect();
    let corner_vertices: Vec<u32> = crease_pairs.iter().map(|&[_, end]| end).collect();

    let case = Case {
        positions,
        face_vertices: result.topology.face_vertex_indices.clone(),
        crease_pairs,
        corner_vertices,
        petite_boundary: None,
        options: SchemeOptions {
            corner_rule: CornerRule::OpenSubdivDeRose,
            ..Default::default()
        },
    };
    (case, center)
}

/// An open `n` x `n` quad grid in the y = height(i, j) sheet over the
/// xz plane (the `tests/limit_surface.rs` layout).
pub fn grid_case(n: u32, height: impl Fn(u32, u32) -> f32) -> Case {
    let stride = n + 1;
    let height = &height;
    let positions: Vec<[f32; 3]> = (0..stride)
        .flat_map(|i| (0..stride).map(move |j| [i as f32, height(i, j), j as f32]))
        .collect();
    let vid = |i: u32, j: u32| i * stride + j;
    let face_vertices: Vec<u32> = (0..n)
        .flat_map(|i| {
            (0..n).flat_map(move |j| [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
        })
        .collect();
    Case {
        positions,
        face_vertices,
        crease_pairs: Vec::new(),
        corner_vertices: Vec::new(),
        petite_boundary: Some(far::BoundaryInterpolation::EdgeOnly),
        options: subdiv_kernels::SchemeOptions::default(),
    }
}

// -- Brute-force closest-point oracle (limit-SDF s3) -----------------------

/// Dense limit-surface point cloud: the limit positions of a deep
/// uniform refinement's vertices. Every sample lies *exactly* on the
/// limit surface (the analytic limit masks), so the brute argmin
/// distance is always an upper bound on the true closest distance --
/// the in-crate oracle of the s3 closest-point gates, no OSD involved.
pub struct BruteOracle {
    pub points: Vec<[f32; 3]>,
    /// Longest limit-position edge of the deep refinement: any limit
    /// point has a sample (its containing quad's corners) within about
    /// one `max_edge`, so the brute argmin overshoots the true
    /// distance by at most that much.
    pub max_edge: f64,
}

pub fn brute_oracle(case: &Case, level: u8) -> BruteOracle {
    use std::num::NonZeroU8;
    use subdiv_kernels::{Refiner, Scheme, UniformRefine};

    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("deep refinement");
    let refined = result.interpolate(&case.positions);
    let limit = result.limit_stencils().expect("limit stencils");
    let points = limit.position.interpolate(&refined);
    let max_edge = result
        .topology
        .edge_vertices
        .iter()
        .map(|&[a, b]| length(sub(v3(points[a as usize]), v3(points[b as usize]))))
        .fold(0.0, f64::max);
    BruteOracle { points, max_edge }
}

impl BruteOracle {
    /// Argmin sample: `(position, distance)`.
    pub fn closest(&self, q: [f64; 3]) -> ([f64; 3], f64) {
        let (pos, dist2) = self
            .points
            .iter()
            .map(|&p| {
                let p = v3(p);
                let d = sub(p, q);
                (p, dot(d, d))
            })
            .fold(([0.0; 3], f64::INFINITY), |best, cand| {
                if cand.1 < best.1 { cand } else { best }
            });
        (pos, dist2.sqrt())
    }

    /// Whether the argmin is ambiguous: some sample within `window` of
    /// the best distance sits farther than `separation` from the
    /// argmin position -- two competing closest regions, i.e. the query
    /// is near the medial axis at sampling resolution.
    pub fn ambiguous(
        &self,
        q: [f64; 3],
        best: ([f64; 3], f64),
        window: f64,
        separation: f64,
    ) -> bool {
        let limit = (best.1 + window) * (best.1 + window);
        let sep2 = separation * separation;
        self.points.iter().any(|&p| {
            let p = v3(p);
            let d = sub(p, q);
            let off = sub(p, best.0);
            dot(d, d) <= limit && dot(off, off) > sep2
        })
    }

    /// Smallest sample norm -- a (sampling-resolution) upper bound on
    /// the inscribed radius of an origin-centered cage, used by the
    /// sign gates' inside margin.
    pub fn min_radius(&self) -> f64 {
        self.points
            .iter()
            .map(|&p| length(v3(p)))
            .fold(f64::INFINITY, f64::min)
    }
}

// -- f64 vector helpers ------------------------------------------------

pub fn v3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = length(a);
    assert!(len > 1e-12, "degenerate vector cannot be normalized: {a:?}");
    [a[0] / len, a[1] / len, a[2] / len]
}

pub fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(normalize(a), normalize(b)).clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI
}

// -- Ptex frame recovery ---------------------------------------------------

/// Position agreement required to match a refined corner to an oracle
/// lattice sample (and the oracle gates' shared position tolerance).
pub const POSITION_TOLERANCE: f64 = 1e-4;

/// Per-quad ptex frame: the root face and the affine in-quad `(u, v)`
/// -> ptex `(s, t)` map as origin (corner 0's `(s, t)`) plus the two
/// columns.
pub struct PtexFrame {
    pub root: usize,
    pub origin: [f64; 2],
    pub e_u: [f64; 2],
    pub e_v: [f64; 2],
}

impl PtexFrame {
    pub fn st(&self, uv: [f32; 2]) -> [f64; 2] {
        let (u, v) = (uv[0] as f64, uv[1] as f64);
        [
            self.origin[0] + u * self.e_u[0] + v * self.e_v[0],
            self.origin[1] + u * self.e_u[1] + v * self.e_v[1],
        ]
    }
}

/// Recover the ptex frame of each listed refined quad by matching its
/// corners' limit positions to the root face's level-`level` lattice
/// samples, asserting each frame is an axis-aligned square cell of
/// side `1/2^level` (possibly rotated/mirrored) -- the convention-free
/// correspondence of the s1/s2 oracle gates.
///
/// `lattice` must be `oracle_samples(case, &lattice_locations(faces,
/// level))`; `limit_positions` is per refined vertex.
pub fn recover_ptex_frames(
    lattice: &[OracleSample],
    level: u8,
    faces: &[u32],
    face_root: &[u32],
    face_vertex_indices: &[u32],
    limit_positions: &[[f32; 3]],
) -> Vec<PtexFrame> {
    let side = (1usize << level) + 1;
    let per_face = side * side;
    let cell = 1.0 / (1u32 << level) as f64;

    faces
        .iter()
        .map(|&face| {
            let root = face_root[face as usize] as usize;
            let corners = &face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
            let st: Vec<[f64; 2]> = corners
                .iter()
                .map(|&vertex| {
                    let position = v3(limit_positions[vertex as usize]);
                    let near: Vec<usize> = (0..per_face)
                        .filter(|&k| {
                            length(sub(lattice[root * per_face + k].position, position))
                                <= POSITION_TOLERANCE
                        })
                        .collect();
                    if near.len() != 1 {
                        let (bk, bd) = (0..lattice.len())
                            .map(|k| (k, length(sub(lattice[k].position, position))))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        eprintln!(
                            "DBG face {face} vtx {vertex} root={root} pos={position:?} GLOBAL-nearest face={} dist={bd:.6}",
                            bk / per_face
                        );
                    }
                    assert_eq!(
                        near.len(),
                        1,
                        "face {face} vertex {vertex}: expected exactly one lattice sample of \
                         ptex face {root} at limit position {position:?}, found {}",
                        near.len(),
                    );
                    let (i, j) = (near[0] / side, near[0] % side);
                    [i as f64 / (side - 1) as f64, j as f64 / (side - 1) as f64]
                })
                .collect();

            let e_u = [st[1][0] - st[0][0], st[1][1] - st[0][1]];
            let e_v = [st[3][0] - st[0][0], st[3][1] - st[0][1]];
            // The four corners must span an axis-aligned level-`level`
            // lattice cell: opposite edges equal, sides of length
            // `cell`, perpendicular.
            for c in 0..2 {
                assert!(
                    (st[2][c] - st[1][c] - e_v[c]).abs() < 1e-9,
                    "face {face}: not affine"
                );
                assert!(
                    (st[2][c] - st[3][c] - e_u[c]).abs() < 1e-9,
                    "face {face}: not affine"
                );
            }
            let norm = |e: [f64; 2]| (e[0] * e[0] + e[1] * e[1]).sqrt();
            assert!(
                (norm(e_u) - cell).abs() < 1e-9,
                "face {face}: u edge is not one cell"
            );
            assert!(
                (norm(e_v) - cell).abs() < 1e-9,
                "face {face}: v edge is not one cell"
            );
            assert!(
                (e_u[0] * e_v[0] + e_u[1] * e_v[1]).abs() < 1e-9,
                "face {face}: cell is not a rectangle",
            );

            PtexFrame {
                root,
                origin: st[0],
                e_u,
                e_v,
            }
        })
        .collect()
}

/// Oracle samples at every frame's mapped `samples`, plus the global
/// oracle row of each `(frame, sample)` pair.
pub fn oracle_at_frame_samples(
    case: &Case,
    frames: &[PtexFrame],
    samples: &[[f32; 2]],
) -> (Vec<OracleSample>, Vec<Vec<usize>>) {
    let mut locations: Vec<PtexLocations> = (0..case.face_count())
        .map(|face| PtexLocations {
            ptex_index: face,
            s: Vec::new(),
            t: Vec::new(),
        })
        .collect();
    let local_rows: Vec<Vec<usize>> = frames
        .iter()
        .map(|frame| {
            samples
                .iter()
                .map(|&uv| {
                    let [s, t] = frame.st(uv);
                    let slot = &mut locations[frame.root];
                    slot.s.push(s as f32);
                    slot.t.push(t as f32);
                    slot.s.len() - 1
                })
                .collect()
        })
        .collect();

    // Petite location arrays must be non-empty; drop unsampled faces
    // and accumulate the survivors' row offsets.
    let mut offsets = vec![usize::MAX; locations.len()];
    let mut total = 0;
    let kept: Vec<PtexLocations> = locations
        .into_iter()
        .filter(|loc| !loc.s.is_empty())
        .inspect(|loc| {
            offsets[loc.ptex_index] = total;
            total += loc.s.len();
        })
        .collect();

    let rows = frames
        .iter()
        .zip(&local_rows)
        .map(|(frame, locals)| {
            locals
                .iter()
                .map(|local| offsets[frame.root] + local)
                .collect()
        })
        .collect();
    (oracle_samples(case, &kept), rows)
}

// -- Oracle evaluation ---------------------------------------------------

/// One oracle limit sample: position and first derivatives at a ptex
/// `(s, t)` location, evaluated from the cage.
pub struct OracleSample {
    pub position: [f64; 3],
    pub du: [f64; 3],
    pub dv: [f64; 3],
}

impl OracleSample {
    /// Unit normal; `None` where the patch derivatives degenerate
    /// (observed exactly at crease-crease patch corners).
    pub fn normal(&self) -> Option<[f64; 3]> {
        let n = cross(self.du, self.dv);
        (length(n) > 1e-9).then(|| normalize(n))
    }
}

/// Evaluation locations on one ptex face.
pub struct PtexLocations {
    pub ptex_index: usize,
    pub s: Vec<f32>,
    pub t: Vec<f32>,
}

/// The `(2^level + 1)^2` lattice of every ptex face, `t` fastest:
/// sample `face * side^2 + i * side + j` is at
/// `(s, t) = (i, j) / 2^level` with `side = 2^level + 1`.
pub fn lattice_locations(face_count: usize, level: u8) -> Vec<PtexLocations> {
    let side = (1usize << level) + 1;
    let lattice: Vec<f32> = (0..side).map(|i| i as f32 / (side - 1) as f32).collect();
    let (s, t): (Vec<f32>, Vec<f32>) = lattice
        .iter()
        .flat_map(|&si| lattice.iter().map(move |&ti| (si, ti)))
        .unzip();
    (0..face_count)
        .map(|face| PtexLocations {
            ptex_index: face,
            s: s.clone(),
            t: t.clone(),
        })
        .collect()
}

/// Petite's limit stencils evaluated at the given per-ptex-face
/// locations, returned concatenated in array order.
pub fn oracle_samples(case: &Case, locations: &[PtexLocations]) -> Vec<OracleSample> {
    let face_count = case.face_count();
    let vertices_per_face = vec![4u32; face_count];
    let mut descriptor = far::TopologyDescriptor::new(
        case.positions.len(),
        &vertices_per_face,
        &case.face_vertices,
    )
    .expect("petite descriptor");
    let crease_flat: Vec<u32> = case.crease_pairs.iter().flatten().copied().collect();
    // 10.0 is OpenSubdiv's infinite-sharpness sentinel.
    let crease_sharpness = vec![10.0f32; case.crease_pairs.len()];
    if !crease_flat.is_empty() {
        // `creases`/`corners` are *consuming* builders (`mut self ->
        // Result<Self>`) and `TopologyDescriptor` is `Copy`, so a bare
        // `descriptor.creases(..)` mutates a discarded copy and silently
        // drops the creases -- rebind the returned descriptor.
        descriptor = descriptor
            .creases(&crease_flat, &crease_sharpness)
            .expect("petite creases");
    }
    let corner_sharpness = vec![10.0f32; case.corner_vertices.len()];
    if !case.corner_vertices.is_empty() {
        descriptor = descriptor
            .corners(&case.corner_vertices, &corner_sharpness)
            .expect("petite corners");
    }
    let options = far::TopologyRefinerOptions {
        boundary_interpolation: case.petite_boundary,
        ..Default::default()
    };
    let mut refiner = far::TopologyRefiner::new(descriptor, options).expect("petite refiner");
    // Infinitely sharp creases must be evaluated as true crease patches
    // -- the default end-cap approximation around them is coarser than
    // the position tolerance.
    let adaptive = far::AdaptiveRefinementOptions {
        infintely_sharp_patch: true,
        isolation_level: 8,
        ..Default::default()
    };
    refiner.refine_adaptive(adaptive, None);

    let location_arrays: Vec<far::LocationArray> = locations
        .iter()
        .map(|loc| far::LocationArray {
            ptex_index: loc.ptex_index,
            s: &loc.s,
            t: &loc.t,
        })
        .collect();

    let table = far::LimitStencilTable::new(
        &refiner,
        &location_arrays,
        None,
        None,
        far::LimitStencilTableOptions::default(),
    )
    .expect("petite limit stencil table");

    let offsets = table.offsets();
    let sizes = table.sizes();
    let indices = table.control_indices();
    let weights = table.weights();
    let du_weights = table.du_weights();
    let dv_weights = table.dv_weights();

    (0..table.len())
        .map(|k| {
            let start: usize = offsets[k].into();
            let len = sizes[k] as usize;
            let mut position = [0.0f64; 3];
            let mut du = [0.0f64; 3];
            let mut dv = [0.0f64; 3];
            for r in start..start + len {
                let cv: usize = indices[r].into();
                let p = v3(case.positions[cv]);
                for c in 0..3 {
                    position[c] += weights[r] as f64 * p[c];
                    du[c] += du_weights[r] as f64 * p[c];
                    dv[c] += dv_weights[r] as f64 * p[c];
                }
            }
            OracleSample { position, du, dv }
        })
        .collect()
}
