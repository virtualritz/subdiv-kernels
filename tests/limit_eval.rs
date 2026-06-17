//! Feature-quad limit-evaluation gates that need no external oracle
//! (limit-surface SDF design s2; the OSD comparisons live in
//! `tests/limit_eval_osd_oracle.rs`).
//!
//! Five properties pin the recursive isolation:
//!
//! - **Seam continuity**: a regular/feature quad pair sharing an edge
//!   must evaluate the same limit points along it (1e-5) -- the
//!   isolation submesh's patches are the same surface the s1 table
//!   evaluates on the regular side.
//! - **Corner consistency**: at feature-quad corners the evaluator and
//!   the sectored limit stencils are two routes to the same limit
//!   point (1e-5; at persistent-feature corners they are the *same*
//!   masks), and where `du x dv` does not degenerate it must align
//!   with the corner's sector normal -- including at pinned cone
//!   points (spoked cube), where both routes share the deterministic
//!   sector tangent plane of `src/limit.rs`. The cross *does*
//!   degenerate at the creased cube's four crease-crease top corners
//!   (both parametric derivatives run along the bent rim curve;
//!   OpenSubdiv's patches degenerate there identically).
//! - **Deep recursion**: a query 1e-4 off an extraordinary corner
//!   isolates ~14 levels down and must land next to the EV's limit
//!   (the surface contracts toward the EV: `2 lambda < 1` for
//!   valence 3); a query 1e-9 off it exhausts the depth cap and must
//!   fall back panic-free to the nearest deepest vertex.
//! - **Semi-sharp decay**: finite sharpness decays across isolation
//!   levels (design §10-3 v1), so a crease of sharpness 1.5 is C1 at
//!   the limit: normals from both sides of the seam agree -- while the
//!   infinitely sharp rim keeps one normal per side (~90 degrees
//!   apart, snapped to per-sector masks).
//! - **Determinism**: repeated queries (the cached-isolation path) are
//!   bit-identical to the first.

use std::num::NonZeroU8;
use subdiv_kernels::{
    QuadClass, RefinementResult, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine,
};

mod common;

use common::{Case, angle_deg, cross, cube_case, grid_case, length, spoked_cube_case, sub, v3};

/// Two-routes-to-the-same-limit agreement (seams, corners).
const CONSISTENCY_TOLERANCE: f64 = 1e-5;
/// Normal agreement where both routes have one.
const NORMAL_TOLERANCE_DEG: f64 = 0.5;

const CORNER_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

struct EvaluatedCase {
    result: RefinementResult,
    refined: Vec<[f32; 3]>,
}

fn evaluated_case(case: &Case, level: u8) -> EvaluatedCase {
    refine(case.topology(), case.options, &case.positions, level)
}

fn refine(
    topo: Mesh,
    options: SchemeOptions,
    positions: &[[f32; 3]],
    level: u8,
) -> EvaluatedCase {
    let refiner = Refiner::new(topo, Scheme::CatmullClark, options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement");
    let refined = result.interpolate(positions);
    EvaluatedCase { result, refined }
}

// -- Seam continuity -----------------------------------------------------

/// Every regular/feature seam edge, sampled in both quads' own
/// parameterizations (the shared edge runs corner `s -> s + 1` in each
/// face, in opposite directions under consistent winding).
fn assert_seams_continuous(case: &Case, level: u8) {
    let ours = evaluated_case(case, level);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let corners =
        |face: u32| &ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];

    let mut seams = 0;
    for (ei, &[fa, fb]) in ours.result.adjacency.edge_faces.iter().enumerate() {
        if fa == u32::MAX || fb == u32::MAX {
            continue;
        }
        let (regular, feature) = match (evaluator.quad_class(fa), evaluator.quad_class(fb)) {
            (QuadClass::Regular, QuadClass::Feature) => (fa, fb),
            (QuadClass::Feature, QuadClass::Regular) => (fb, fa),
            _ => continue,
        };
        seams += 1;
        let slot = |face: u32| {
            ours.result.adjacency.face_edges[face as usize * 4..face as usize * 4 + 4]
                .iter()
                .position(|&e| e == ei as u32)
                .expect("edge is in its incident face")
        };
        let (sa, sb) = (slot(regular), slot(feature));
        // Consistent winding: the shared edge is traversed oppositely.
        assert_eq!(corners(regular)[sa], corners(feature)[(sb + 1) % 4]);
        assert_eq!(corners(regular)[(sa + 1) % 4], corners(feature)[sb]);

        let edge_uv = |s: usize, t: f32| {
            let (a, b) = (CORNER_UV[s], CORNER_UV[(s + 1) % 4]);
            [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
        };
        for &t in &[0.0f32, 0.25, 0.4, 0.5, 0.77, 1.0] {
            let pa = evaluator
                .eval(regular, edge_uv(sa, t))
                .expect("regular-side eval");
            let pb = evaluator
                .eval(feature, edge_uv(sb, 1.0 - t))
                .expect("feature-side eval");
            let distance = length(sub(v3(pa), v3(pb)));
            assert!(
                distance <= CONSISTENCY_TOLERANCE,
                "edge {ei} (regular {regular} / feature {feature}) at t = {t}: \
                 {pa:?} vs {pb:?} ({distance} apart)",
            );
        }
    }
    assert!(seams > 0, "case has no regular/feature seam to check");
}

#[test]
fn warped_grid_regular_feature_seams_are_continuous() {
    let case = grid_case(6, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    });
    assert_seams_continuous(&case, 1);
}

#[test]
fn cube_regular_feature_seams_are_continuous() {
    assert_seams_continuous(&cube_case(false), 2);
}

#[test]
fn creased_cube_regular_feature_seams_are_continuous() {
    assert_seams_continuous(&cube_case(true), 2);
}

// -- Corner consistency ----------------------------------------------------

/// Feature-quad corner eval vs the sectored limit stencils, position
/// and (where non-degenerate) the in-sector normal. Returns how many
/// normals were compared.
fn assert_feature_corners_match_sectored_stencils(case: &Case, level: u8) -> usize {
    let ours = evaluated_case(case, level);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let sectored = ours
        .result
        .sectored_limit_stencils()
        .expect("sectored limit stencils");
    let limit_positions = sectored.position.interpolate(&ours.refined);
    let tan1 = sectored.tangent1.interpolate(&ours.refined);
    let tan2 = sectored.tangent2.interpolate(&ours.refined);

    let mut compared = 0;
    let face_count = ours.result.topology.face_vertex_counts.len() as u32;
    for face in (0..face_count).filter(|&f| evaluator.quad_class(f) == QuadClass::Feature) {
        let corners =
            &ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
        for (k, &uv) in CORNER_UV.iter().enumerate() {
            let (p, du, dv) = evaluator
                .eval_with_derivatives(face, uv)
                .expect("corner eval");
            let stencil = v3(limit_positions[corners[k] as usize]);
            let distance = length(sub(v3(p), stencil));
            assert!(
                distance <= CONSISTENCY_TOLERANCE,
                "face {face} corner {k}: eval {p:?} vs limit stencil {stencil:?} \
                 ({distance} apart)",
            );

            let row = sectored.corner_sector[face as usize * 4 + k] as usize;
            let sector_normal = cross(v3(tan1[row]), v3(tan2[row]));
            let our_normal = cross(v3(du), v3(dv));
            if length(sector_normal) > 1e-9 && length(our_normal) > 1e-9 {
                let angle = angle_deg(our_normal, sector_normal);
                assert!(
                    angle <= NORMAL_TOLERANCE_DEG,
                    "face {face} corner {k}: normal {our_normal:?} is {angle} degrees off \
                     sector row {row}'s {sector_normal:?}",
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "no corner normals were compared at all");
    compared
}

#[test]
fn cube_feature_corners_match_sectored_stencils() {
    // All 24 once-refined quads are feature; every corner normal is
    // well defined.
    assert_eq!(
        assert_feature_corners_match_sectored_stencils(&cube_case(false), 1),
        96,
    );
}

#[test]
fn creased_cube_feature_corners_match_sectored_stencils() {
    // 40 feature quads x 4 corners, minus the 4 crease-crease top
    // corners where both one-sided derivatives run along the bent rim
    // curve and the cross degenerates (see the module docs).
    assert_eq!(
        assert_feature_corners_match_sectored_stencils(&cube_case(true), 2),
        156,
    );
}

#[test]
fn spoked_cube_feature_corners_match_sectored_stencils() {
    // Pinned cone point + crease spokes + darts under OpenSubdivDeRose;
    // the count just pins the no-degenerate-cross observation.
    let (case, _) = spoked_cube_case();
    assert_feature_corners_match_sectored_stencils(&case, 2);
}

#[test]
fn warped_grid_feature_corners_match_sectored_stencils() {
    // Real-boundary snaps: the open grid's feature quads put corner
    // queries on boundary crease vertices (and the single-face grid
    // corners) under the default EdgesOnly rule.
    let case = grid_case(6, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    });
    assert_feature_corners_match_sectored_stencils(&case, 1);
}

// -- Deep recursion and the depth cap ----------------------------------------

/// A feature quad of the once-refined cube and the CSR corner of its
/// extraordinary (valence-3) vertex, plus the EV's limit position.
fn cube_ev_corner() -> (EvaluatedCase, u32, usize, [f64; 3]) {
    let case = cube_case(false);
    let ours = evaluated_case(&case, 1);
    let limit = ours.result.limit_stencils().expect("limit stencils");
    let limit_positions = limit.position.interpolate(&ours.refined);

    let valence = |vi: u32| {
        ours.result.adjacency.vert_edge_offsets[vi as usize + 1]
            - ours.result.adjacency.vert_edge_offsets[vi as usize]
    };
    let (face, k) = (0..ours.result.topology.face_vertex_counts.len() as u32)
        .find_map(|face| {
            ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4]
                .iter()
                .position(|&c| valence(c) == 3)
                .map(|k| (face, k))
        })
        .expect("a quad touching an extraordinary vertex");
    let ev = ours.result.topology.face_vertex_indices[face as usize * 4 + k];
    let ev_limit = v3(limit_positions[ev as usize]);
    (ours, face, k, ev_limit)
}

/// `(u, v)` at parametric distance `inset` inside the quad from corner
/// `k`.
fn off_corner(k: usize, inset: f32) -> [f32; 2] {
    [
        if CORNER_UV[k][0] == 0.0 { inset } else { 1.0 - inset },
        if CORNER_UV[k][1] == 0.0 { inset } else { 1.0 - inset },
    ]
}

#[test]
fn deep_recursion_isolates_near_extraordinary_corner() {
    let (ours, face, k, ev_limit) = cube_ev_corner();
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");

    // 1e-4 off the EV corner isolates ~14 levels down to a regular
    // patch; the surface contracts toward the EV ((2 lambda)^depth,
    // lambda ~ 0.41 at valence 3), so the point lands within ~1e-4 of
    // the EV's limit -- and stays distinct from it.
    let (p, du, dv) = evaluator
        .eval_with_derivatives(face, off_corner(k, 1e-4))
        .expect("deep eval");
    let distance = length(sub(v3(p), ev_limit));
    assert!(
        distance <= 1e-4,
        "deep query landed {distance} from the EV limit {ev_limit:?}: {p:?}",
    );
    assert!(
        [du, dv].iter().flatten().all(|c| c.is_finite()),
        "deep-query derivatives are not finite: {du:?}, {dv:?}",
    );
}

#[test]
fn depth_cap_fallback_is_panic_free() {
    let (ours, face, k, ev_limit) = cube_ev_corner();
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");

    // 1e-9 off the EV corner cannot isolate within the cap (the
    // containing child still touches the EV until depth ~30); the
    // fallback snaps to the nearest deepest vertex -- the EV's
    // descendant, converged onto its limit.
    let (p, du, dv) = evaluator
        .eval_with_derivatives(face, off_corner(k, 1e-9))
        .expect("depth-cap eval");
    let distance = length(sub(v3(p), ev_limit));
    assert!(
        distance <= 1e-4,
        "depth-cap fallback landed {distance} from the EV limit {ev_limit:?}: {p:?}",
    );
    assert!(
        [du, dv].iter().flatten().all(|c| c.is_finite()),
        "depth-cap derivatives are not finite: {du:?}, {dv:?}",
    );

    // A query exactly on a crease line whose dyadic bits outlast the
    // cap: both sides' fallbacks snap to nearest deepest rim vertices
    // within 2^-20 of the same crease point.
    let creased = evaluated_case(&cube_case(true), 1);
    let evaluator = creased
        .result
        .limit_evaluator(&creased.refined)
        .expect("limit evaluator");
    let (rim_edge, &[fa, fb]) = creased
        .result
        .adjacency
        .edge_faces
        .iter()
        .enumerate()
        .find(|&(ei, _)| creased.result.topology.edge_creases[ei] > 0.0)
        .expect("a rim edge");
    let edge_uv = |face: u32, t: f32| {
        let s = creased.result.adjacency.face_edges[face as usize * 4..face as usize * 4 + 4]
            .iter()
            .position(|&e| e == rim_edge as u32)
            .expect("edge is in its incident face");
        let (a, b) = (CORNER_UV[s], CORNER_UV[(s + 1) % 4]);
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
    };
    let pa = evaluator.eval(fa, edge_uv(fa, 0.4)).expect("side a eval");
    let pb = evaluator
        .eval(fb, edge_uv(fb, 1.0 - 0.4))
        .expect("side b eval");
    let distance = length(sub(v3(pa), v3(pb)));
    assert!(
        distance <= 1e-4,
        "depth-cap crease-line evals disagree across the seam: {pa:?} vs {pb:?} ({distance})",
    );
}

// -- Semi-sharp decay --------------------------------------------------------

#[test]
fn semi_sharp_crease_decays_to_a_smooth_seam() {
    // The creased-cube rim at finite sharpness 1.5: isolation decays it
    // to zero within two levels, so the limit is C1 across the rim --
    // unlike the infinite crease, which keeps one normal per side.
    let normals_across_rim = |sharpness: f32| -> f64 {
        let case = cube_case(true);
        let mut topo = case.topology();
        topo.edge_creases = vec![sharpness; topo.edge_vertices.len()];
        let ours = refine(topo, case.options, &case.positions, 1);
        let evaluator = ours
            .result
            .limit_evaluator(&ours.refined)
            .expect("limit evaluator");
        let (rim_edge, &[fa, fb]) = ours
            .result
            .adjacency
            .edge_faces
            .iter()
            .enumerate()
            .find(|&(ei, _)| ours.result.topology.edge_creases[ei] > 0.0)
            .expect("a rim edge");
        let normal = |face: u32, t: f32| {
            let s = ours.result.adjacency.face_edges[face as usize * 4..face as usize * 4 + 4]
                .iter()
                .position(|&e| e == rim_edge as u32)
                .expect("edge is in its incident face");
            let (a, b) = (CORNER_UV[s], CORNER_UV[(s + 1) % 4]);
            let uv = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            let (_, du, dv) = evaluator
                .eval_with_derivatives(face, uv)
                .expect("rim eval");
            cross(v3(du), v3(dv))
        };
        // 0.5 is dyadic: the infinite-crease side snaps at depth 1
        // instead of running into the depth cap.
        angle_deg(normal(fa, 0.5), normal(fb, 0.5))
    };

    let smooth = normals_across_rim(1.5);
    assert!(
        smooth <= NORMAL_TOLERANCE_DEG,
        "semi-sharp rim normals disagree by {smooth} degrees, expected a smooth seam",
    );
    let sharp = normals_across_rim(f32::INFINITY);
    assert!(
        (60.0..=120.0).contains(&sharp),
        "infinite rim normals disagree by {sharp} degrees, expected ~90",
    );
}

// -- Determinism ---------------------------------------------------------------

#[test]
fn repeated_queries_are_bit_identical() {
    let ours = evaluated_case(&cube_case(true), 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let face_count = ours.result.topology.face_vertex_counts.len() as u32;
    let samples: Vec<(u32, [f32; 2])> = (0..face_count)
        .filter(|&f| evaluator.quad_class(f) == QuadClass::Feature)
        .flat_map(|f| {
            [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.3, 0.7], [0.85, 0.15], [1e-4, 1e-4]]
                .into_iter()
                .map(move |uv| (f, uv))
        })
        .collect();
    let first: Vec<_> = samples
        .iter()
        .map(|&(f, uv)| evaluator.eval_with_derivatives(f, uv).expect("first pass"))
        .collect();
    let second: Vec<_> = samples
        .iter()
        .map(|&(f, uv)| evaluator.eval_with_derivatives(f, uv).expect("second pass"))
        .collect();
    assert_eq!(first, second, "cached isolation changed a repeated query");
}
