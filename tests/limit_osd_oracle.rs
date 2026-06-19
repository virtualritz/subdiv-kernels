//! Limit stencils vs the OpenSubdiv oracle (GPU-subdivision design §5).
//!
//! `opensubdiv-petite`'s `far::LimitStencilTable` evaluates limit
//! position + du/dv from the cage at lattice locations
//! `(i / 2^L, j / 2^L)` per ptex face -- the parametric images of our
//! uniformly refined level-L vertices. Two gates run against it:
//!
//! # Per-vertex (`LimitStencils`), nearest-position matching
//!
//! Our limit points are matched to the oracle's by nearest position (no
//! index-correspondence assumptions), then positions and normalized
//! `cross` normals are compared, allowing one global orientation sign.
//!
//! Normals at *interior* crease vertices are excluded: the limit
//! surface has one normal per side there, OpenSubdiv's limit masks pick
//! a side by a ring-ordering incidental, and its patch samples for the
//! *other* sides inherit that choice's tangents (observed: the flat
//! creased-cube top reports a non-planar corner normal). Positions are
//! still asserted at those vertices, and the crease tangent masks --
//! regular, Biermann irregular, and single-face corner -- are oracle
//! checked one-sidedly via the boundary fan case, where the evaluated
//! span is unambiguous.
//!
//! # Per-corner (`SectoredLimitStencils`), lattice matching
//!
//! The sectored API removes the side ambiguity: a lattice corner of
//! ptex face `F` evaluates *in* `F`'s sector. Matching is per
//! `(ptex face, lattice corner) -> (refined face, corner)`: the refined
//! face's base ancestor is `face_root` (== ptex face on an all-quad
//! cage), and the lattice point within it is found by limit *position*
//! (positions are unambiguous; only normals were not). Every corner's
//! normal is then compared against the oracle sample of its own face,
//! so both sides of a sharp crease are asserted -- exactly the case the
//! per-vertex API failed.
//!
//! One narrow exclusion remains, at *pinned* (corner-rule) vertices
//! whose sector spans more than one face: the limit surface is a true
//! cone point there and OpenSubdiv's own patch evaluation does not
//! converge -- probed on the spoked cube below, its two in-sector faces
//! report corner normals ~22 degrees apart, both drifting monotonically
//! with the adaptive isolation level (45.7 -> 37.2 -> 32.8 degrees off
//! the bounding-edge plane at isolation 5/8/10) while every other
//! sample is isolation-stable. Positions are still asserted there, and
//! the *single*-face pinned sectors match the oracle exactly.
//!
//! Cases: a cube (smooth + extraordinary vertices), a cube with one
//! infinitely sharp edge ring (interior crease rule), a warped 3-quad
//! boundary fan (boundary crease rules; its center has a 3-face span --
//! the Biermann irregular cross-tangent), and a once-refined cube with
//! three infinitely sharp spokes at a face-center (corner rule: pinned
//! vertex with single- and multi-face sectors).

mod common;

use common::{
    Case, OracleSample, cross, cube_case, dot, fan_case, lattice_locations, length, normalize,
    oracle_samples, spoked_cube_case, sub, v3,
};
use std::num::NonZeroU8;
use subdiv_kernels::{
    LimitStencils, RefinementResult, Refiner, Scheme, SectoredLimitStencils, UniformRefine,
};

const LEVEL: u8 = 2;
const POSITION_TOLERANCE: f64 = 1e-4;
const NORMAL_TOLERANCE_DEG: f64 = 0.5;

/// One of our analytic samples: limit position + unit normal (always
/// present; the oracle side is [`OracleSample`], whose normal can
/// degenerate at crease-crease patch corners).
struct Sample {
    position: [f64; 3],
    normal: Option<[f64; 3]>,
}

/// Petite's limit stencils evaluated at the level-`LEVEL` lattice of
/// every ptex (= cage quad) face.
fn oracle_lattice(case: &Case) -> Vec<OracleSample> {
    oracle_samples(case, &lattice_locations(case.face_count(), LEVEL))
}

/// Our analytic limit from the cage via the composed tables, plus a
/// per-vertex "interior crease" flag (true where the limit normal is
/// side-ambiguous and excluded from the normal comparison).
fn our_samples(case: &Case) -> (Vec<Sample>, Vec<bool>) {
    let (result, limit) = our_refinement(case);
    let positions = limit.position.interpolate(&case.positions);
    let tan1 = limit.tangent1.interpolate(&case.positions);
    let tan2 = limit.tangent2.interpolate(&case.positions);
    let samples = positions
        .iter()
        .zip(tan1.iter().zip(&tan2))
        .map(|(&p, (&t1, &t2))| Sample {
            position: v3(p),
            normal: Some(normalize(cross(v3(t1), v3(t2)))),
        })
        .collect();

    // Interior crease flag: the vertex touches a stored (non-boundary)
    // crease at the refined level.
    let ambiguous = (0..result.topology.vertex_count as usize)
        .map(|vi| {
            let start = result.adjacency.vertex_edge_offsets[vi] as usize;
            let end = result.adjacency.vertex_edge_offsets[vi + 1] as usize;
            result.adjacency.vertex_edges[start..end]
                .iter()
                .any(|&ei| result.topology.edge_creases[ei as usize] > 0.0)
        })
        .collect();

    (samples, ambiguous)
}

/// Refine a case's cage to `LEVEL` and compose its per-vertex limit
/// tables onto the cage.
fn our_refinement(case: &Case) -> (RefinementResult, LimitStencils) {
    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(NonZeroU8::new(LEVEL).expect("non-zero")))
        .expect("refinement");
    let limit = result
        .compose_limit_stencils(case.positions.len())
        .expect("composed limit stencils");
    (result, limit)
}

/// Geometric matching: every vertex of ours must coincide with at least
/// one oracle sample; unambiguous vertices must also match a coincident
/// sample's normal within tolerance (after detecting one global
/// orientation sign, asserted consistent everywhere).
fn assert_limit_matches_oracle(case: &Case) {
    let (ours, ambiguous) = our_samples(case);
    let oracle = oracle_lattice(case);

    let candidates: Vec<Vec<usize>> = ours
        .iter()
        .enumerate()
        .map(|(vi, sample)| {
            let near: Vec<usize> = oracle
                .iter()
                .enumerate()
                .filter(|(_, o)| length(sub(o.position, sample.position)) <= POSITION_TOLERANCE)
                .map(|(k, _)| k)
                .collect();
            assert!(
                !near.is_empty(),
                "vertex {vi}: no oracle sample within {POSITION_TOLERANCE} of limit position {:?} \
                 (nearest at {:?})",
                sample.position,
                oracle
                    .iter()
                    .map(|o| o.position)
                    .min_by(|a, b| {
                        let (da, db) = (length(sub(*a, sample.position)),
                                        length(sub(*b, sample.position)));
                        // SAFETY-free total order: distances are finite.
                        da.partial_cmp(&db).expect("finite distances")
                    }),
            );
            near
        })
        .collect();

    // Detect one global sign: oracle and our windings agree here, but
    // the contract only requires consistency, not a fixed sign.
    let votes: f64 = ours
        .iter()
        .zip(&candidates)
        .zip(&ambiguous)
        .filter(|&(_, &skip)| !skip)
        .map(|((sample, near), _)| {
            let normal = sample.normal.expect("our samples always carry normals");
            near.iter()
                .filter_map(|&k| oracle[k].normal().map(|n| dot(normal, n)))
                .fold(0.0f64, |best, d| if d.abs() > best.abs() { d } else { best })
        })
        .sum();
    let sign = if votes >= 0.0 { 1.0 } else { -1.0 };

    let min_alignment = (NORMAL_TOLERANCE_DEG * std::f64::consts::PI / 180.0).cos();
    for (vi, ((sample, near), &skip)) in ours.iter().zip(&candidates).zip(&ambiguous).enumerate() {
        if skip {
            continue;
        }
        let normal = sample.normal.expect("our samples always carry normals");
        let best = near
            .iter()
            .filter_map(|&k| oracle[k].normal().map(|n| sign * dot(normal, n)))
            .fold(f64::MIN, f64::max);
        assert!(
            best >= min_alignment,
            "vertex {vi} at {:?}: analytic normal {normal:?} matches no coincident oracle normal \
             (best alignment {best}, need {min_alignment}; candidates {:?})",
            sample.position,
            near.iter().map(|&k| oracle[k].normal()).collect::<Vec<_>>(),
        );
    }
}

/// Our sectored limit data evaluated from the cage.
struct SectoredSamples {
    result: RefinementResult,
    sectored: SectoredLimitStencils,
    positions: Vec<[f32; 3]>,
    tan1: Vec<[f32; 3]>,
    tan2: Vec<[f32; 3]>,
}

fn sectored_samples(case: &Case) -> SectoredSamples {
    let (result, _) = our_refinement(case);
    let sectored = result
        .compose_sectored_limit_stencils(case.positions.len())
        .expect("composed sectored limit stencils");
    let positions = sectored.position.interpolate(&case.positions);
    let tan1 = sectored.tangent1.interpolate(&case.positions);
    let tan2 = sectored.tangent2.interpolate(&case.positions);
    SectoredSamples {
        result,
        sectored,
        positions,
        tan1,
        tan2,
    }
}

/// Per-corner matching: refined corner `(f, c)` maps to the lattice
/// point of ptex face `face_root[f]` whose oracle position coincides
/// with the corner vertex's limit position, then the corner's sector
/// normal must match that sample's -- so the two sides of a sharp
/// crease are asserted independently. Skipped normals (still
/// position-matched): oracle samples without one (degenerate patch
/// derivatives), and corners of multi-face sectors at pinned vertices
/// (cone points; see the module docs). Returns our samples plus the
/// deduped `(vertex, sector row)` pairs whose normals were compared.
fn assert_sectored_limit_matches_oracle(case: &Case) -> (SectoredSamples, Vec<(u32, u32)>) {
    let ours = sectored_samples(case);
    let oracle = oracle_lattice(case);
    let topo = &ours.result.topology;
    let side = (1usize << LEVEL) + 1;
    let per_face = side * side;

    // Sector face counts (a sector with k fan faces owns k corners) and
    // pinned-vertex flags for the cone-point exclusion.
    let row_corners = ours.sectored.corner_sector.iter().fold(
        vec![0u32; ours.tan1.len()],
        |mut counts, &row| {
            counts[row as usize] += 1;
            counts
        },
    );
    let pinned: Vec<bool> = (0..topo.vertex_count as usize)
        .map(|vi| {
            let start = ours.result.adjacency.vertex_edge_offsets[vi] as usize;
            let end = ours.result.adjacency.vertex_edge_offsets[vi + 1] as usize;
            let sharp = ours.result.adjacency.vertex_edges[start..end]
                .iter()
                .filter(|&&ei| topo.edge_creases[ei as usize] > 0.0)
                .count();
            sharp >= 3 || topo.vertex_corners[vi] > 0.0
        })
        .collect();

    // Match every refined corner to its own face's lattice sample.
    let matches: Vec<(usize, u32, u32, usize)> = topo
        .face_vertex_indices
        .iter()
        .enumerate()
        .map(|(corner, &vertex)| {
            let face = corner / 4;
            let base = ours.result.face_root[face] as usize * per_face;
            let position = v3(ours.positions[vertex as usize]);
            let near: Vec<usize> = (base..base + per_face)
                .filter(|&k| length(sub(oracle[k].position, position)) <= POSITION_TOLERANCE)
                .collect();
            assert_eq!(
                near.len(),
                1,
                "corner {corner} (face {face}, vertex {vertex}): expected exactly one lattice \
                 sample of ptex face {} at limit position {position:?}, found {}",
                ours.result.face_root[face],
                near.len(),
            );
            let row = ours.sectored.corner_sector[corner];
            (corner, vertex, row, near[0])
        })
        .collect();

    let compared = |&(_, vertex, row, sample): &(usize, u32, u32, usize)| -> Option<[f64; 3]> {
        let multi_face_pinned = pinned[vertex as usize] && row_corners[row as usize] > 1;
        if multi_face_pinned {
            None
        } else {
            oracle[sample].normal()
        }
    };

    // Detect one global orientation sign, as in the per-vertex gate.
    let votes: f64 = matches
        .iter()
        .filter_map(|m| {
            compared(m).map(|oracle_normal| {
                let normal = normalize(cross(v3(ours.tan1[m.2 as usize]), v3(ours.tan2[m.2 as usize])));
                dot(normal, oracle_normal)
            })
        })
        .sum();
    let sign = if votes >= 0.0 { 1.0 } else { -1.0 };

    let min_alignment = (NORMAL_TOLERANCE_DEG * std::f64::consts::PI / 180.0).cos();
    let mut rows_compared: Vec<(u32, u32)> = Vec::new();
    for m in &matches {
        let (corner, vertex, row, sample) = *m;
        let Some(oracle_normal) = compared(m) else {
            continue;
        };
        let normal = normalize(cross(v3(ours.tan1[row as usize]), v3(ours.tan2[row as usize])));
        let alignment = sign * dot(normal, oracle_normal);
        assert!(
            alignment >= min_alignment,
            "corner {corner} (vertex {vertex}, sector row {row}): sector normal {normal:?} \
             disagrees with its face's oracle normal {oracle_normal:?} at sample {sample} \
             (alignment {alignment}, need {min_alignment})",
        );
        rows_compared.push((vertex, row));
    }
    assert!(
        !rows_compared.is_empty(),
        "no corner normals were compared at all",
    );
    rows_compared.sort_unstable();
    rows_compared.dedup();
    (ours, rows_compared)
}

/// Refined vertices with exactly two sharp incident edges (the crease
/// rule -- two sectors each).
fn crease_vertices(ours: &SectoredSamples) -> Vec<u32> {
    (0..ours.result.topology.vertex_count)
        .filter(|&vi| {
            let start = ours.result.adjacency.vertex_edge_offsets[vi as usize] as usize;
            let end = ours.result.adjacency.vertex_edge_offsets[vi as usize + 1] as usize;
            ours.result.adjacency.vertex_edges[start..end]
                .iter()
                .filter(|&&ei| ours.result.topology.edge_creases[ei as usize] > 0.0)
                .count()
                == 2
        })
        .collect()
}

#[test]
fn cube_limit_matches_opensubdiv_oracle() {
    assert_limit_matches_oracle(&cube_case(false));
}

#[test]
fn creased_cube_limit_matches_opensubdiv_oracle() {
    assert_limit_matches_oracle(&cube_case(true));
}

#[test]
fn boundary_fan_limit_matches_opensubdiv_oracle() {
    assert_limit_matches_oracle(&fan_case());
}

#[test]
fn cube_sectored_limit_matches_opensubdiv_oracle_per_corner() {
    let (ours, _) = assert_sectored_limit_matches_oracle(&cube_case(false));
    // Smooth mesh: one sector per vertex.
    assert_eq!(ours.tan1.len(), ours.result.topology.vertex_count as usize);
}

#[test]
fn creased_cube_sectored_limit_matches_opensubdiv_oracle_per_corner() {
    let (ours, compared) = assert_sectored_limit_matches_oracle(&cube_case(true));

    // Both sides of the sharp crease matched: every rim vertex has two
    // sector rows, and both rows passed the per-corner normal gate --
    // except possibly at the 4 base corners, whose single top-face
    // corner is a crease-crease patch corner where the oracle has no
    // normal (positions are still asserted there).
    let rim = crease_vertices(&ours);
    assert_eq!(rim.len(), 16, "unexpected rim vertex count");
    let both_sides = rim
        .iter()
        .filter(|&&vi| compared.iter().filter(|&&(v, _)| v == vi).count() >= 2)
        .count();
    assert!(
        both_sides >= rim.len() - 4,
        "only {both_sides} of {} rim vertices had both sector normals oracle-matched",
        rim.len(),
    );
}

#[test]
fn boundary_fan_sectored_limit_matches_opensubdiv_oracle_per_corner() {
    // Boundary crease vertices have a single sector, so the sectored
    // tables stay vertex-sized and every corner normal is compared.
    let (ours, _) = assert_sectored_limit_matches_oracle(&fan_case());
    assert_eq!(ours.tan1.len(), ours.result.topology.vertex_count as usize);
}

#[test]
fn spoked_cube_sectored_limit_matches_opensubdiv_oracle_per_corner() {
    let (case, center) = spoked_cube_case();
    let center_position = v3(case.positions[center as usize]);
    let (ours, compared) = assert_sectored_limit_matches_oracle(&case);

    // The pinned vertex's descendant sits exactly at the cage position
    // (the corner rule interpolates it).
    let pinned = (0..ours.result.topology.vertex_count as usize)
        .find(|&vi| length(sub(v3(ours.positions[vi]), center_position)) < 1e-6)
        .expect("pinned vertex descendant") as u32;

    // Its two single-face sector normals were oracle-matched (they are
    // exact there); the two-face sector is the documented cone-point
    // exclusion, so exactly two of its three rows were compared.
    let pinned_rows: Vec<u32> = compared
        .iter()
        .filter(|&&(v, _)| v == pinned)
        .map(|&(_, row)| row)
        .collect();
    assert_eq!(
        pinned_rows.len(),
        2,
        "expected the two single-face sectors of the pinned vertex to be compared: {pinned_rows:?}",
    );

    // The spoke interiors are crease vertices; both their sector rows
    // must have matched, like the creased cube's rim.
    let spokes = crease_vertices(&ours);
    assert!(!spokes.is_empty(), "expected crease vertices along the spokes");
    let both_sides = spokes
        .iter()
        .filter(|&&vi| compared.iter().filter(|&&(v, _)| v == vi).count() >= 2)
        .count();
    assert!(
        both_sides >= spokes.len() - 1,
        "only {both_sides} of {} spoke vertices had both sector normals oracle-matched",
        spokes.len(),
    );
}
