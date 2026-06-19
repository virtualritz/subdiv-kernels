//! W-row identity gates for `LimitEvaluator::weights_at` (amendment
//! limit-oracle design s5-p1; its §8 "W-row identity").
//!
//! - **Identity**: the row dotted with the refined positions equals
//!   `eval(face, uv)` at corner/edge/interior samples on *every* quad
//!   of the fixtures -- regular and feature alike (EV, crease, corner
//!   rule, boundary).
//! - **Partition of unity + well-formedness**: weights sum to 1;
//!   indices are unique, in range, and the row is never empty.
//! - **Cage fold**: folding the refined row through
//!   `compose_stencils` yields a cage row whose dot with the cage
//!   positions is the same limit point -- the host-side fold the
//!   amendment oracle performs (design §3, §10-1).
//! - **Deep isolation + depth cap**: queries just off / dyadically
//!   beyond an extraordinary corner keep identity + unity.
//! - **Clamping**: out-of-range `(u, v)` rows equal the clamped rows
//!   bit for bit, like `eval`.
//! - **Determinism**: repeated rows are bit-identical (the cached
//!   isolation path).

mod common;

use std::collections::BTreeMap;
use std::num::NonZeroU8;

use common::{Case, cube_case, grid_case, length, spoked_cube_case, sub, v3};
use subdiv_kernels::{
    LimitEvaluator, QuadClass, RefinementResult, Refiner, Scheme, UniformRefine,
};

/// Row-vs-eval agreement (the f32 rounding budget of the two routes).
const IDENTITY_TOLERANCE: f64 = 1e-5;
/// Partition-of-unity budget (f32 weight rounding only).
const UNITY_TOLERANCE: f64 = 1e-6;

/// In-quad samples: corners, an edge midpoint, the center, asymmetric
/// interior points (the OSD-oracle sample set).
const UV_SAMPLES: [[f32; 2]; 8] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [0.5, 0.5],
    [0.3, 0.7],
    [0.85, 0.15],
];

const CORNER_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

struct EvaluatedCase {
    result: RefinementResult,
    refined: Vec<[f32; 3]>,
}

fn evaluated_case(case: &Case, level: u8) -> EvaluatedCase {
    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement");
    let refined = result.interpolate(&case.positions);
    EvaluatedCase { result, refined }
}

/// Dot a sparse row with a positions buffer, accumulating in f64.
fn apply(row: &[(u32, f32)], positions: &[[f32; 3]]) -> [f64; 3] {
    row.iter().fold([0.0f64; 3], |acc, &(i, w)| {
        let p = positions[i as usize];
        [
            acc[0] + w as f64 * p[0] as f64,
            acc[1] + w as f64 * p[1] as f64,
            acc[2] + w as f64 * p[2] as f64,
        ]
    })
}

/// The §8 gates for one row: well-formed, partition of unity, and the
/// identity against `eval(face, uv)`. Returns the row.
fn assert_row_gates(
    evaluator: &LimitEvaluator,
    refined: &[[f32; 3]],
    face: u32,
    uv: [f32; 2],
) -> Vec<(u32, f32)> {
    let row = evaluator.weights_at(face, uv).expect("weights_at");
    assert!(!row.is_empty(), "face {face} at {uv:?}: empty weight row");
    let unique: BTreeMap<u32, f32> = row.iter().copied().collect();
    assert_eq!(
        unique.len(),
        row.len(),
        "face {face} at {uv:?}: duplicate indices in {row:?}",
    );
    assert!(
        row.iter().all(|&(i, _)| (i as usize) < refined.len()),
        "face {face} at {uv:?}: index out of range in {row:?}",
    );
    let total: f64 = row.iter().map(|&(_, w)| w as f64).sum();
    assert!(
        (total - 1.0).abs() <= UNITY_TOLERANCE,
        "face {face} at {uv:?}: weights sum to {total}, not 1",
    );
    let evaluated = v3(evaluator.eval(face, uv).expect("eval"));
    let folded = apply(&row, refined);
    let distance = length(sub(folded, evaluated));
    assert!(
        distance <= IDENTITY_TOLERANCE,
        "face {face} at {uv:?}: row gives {folded:?}, eval gives {evaluated:?} \
         ({distance} apart)",
    );
    row
}

/// Sweep every quad of a fixture at every sample; return how many
/// (regular, feature) quads were exercised.
fn sweep_case(case: &Case, level: u8) -> (usize, usize) {
    let ours = evaluated_case(case, level);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let face_count = ours.result.topology.face_vertex_counts.len() as u32;
    let (mut regular, mut feature) = (0, 0);
    for face in 0..face_count {
        match evaluator.quad_class(face) {
            QuadClass::Regular => regular += 1,
            QuadClass::Feature => feature += 1,
        }
        for &uv in &UV_SAMPLES {
            assert_row_gates(&evaluator, &ours.refined, face, uv);
        }
    }
    (regular, feature)
}

// -- Identity + unity sweeps -------------------------------------------------

#[test]
fn cube_rows_match_eval() {
    // Level 1: all 24 quads are feature (EV isolation).
    let (regular, feature) = sweep_case(&cube_case(false), 1);
    assert_eq!((regular, feature), (0, 24));
}

#[test]
fn creased_cube_rows_match_eval() {
    // Level 2: infinite-crease rim quads plus regular interior quads.
    let (regular, feature) = sweep_case(&cube_case(true), 2);
    assert!(regular > 0 && feature > 0);
}

#[test]
fn warped_grid_rows_match_eval() {
    // Open grid: boundary snaps under the default EdgesOnly rule.
    let case = grid_case(6, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    });
    let (regular, feature) = sweep_case(&case, 1);
    assert!(regular > 0 && feature > 0);
}

#[test]
fn spoked_cube_rows_match_eval() {
    // Pinned cone point + crease spokes + darts under OpenSubdivDeRose.
    let (case, _) = spoked_cube_case();
    let (_, feature) = sweep_case(&case, 2);
    assert!(feature > 0);
}

// -- Cage fold (design §3: refined row + host fold) ---------------------------

#[test]
fn cage_fold_reproduces_eval_from_cage_positions() {
    let case = cube_case(true);
    let ours = evaluated_case(&case, 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let cage_to_refined = ours.result.compose_stencils(case.positions.len());

    let face_count = ours.result.topology.face_vertex_counts.len() as u32;
    for face in 0..face_count {
        for &uv in &[[0.5f32, 0.5], [0.3, 0.7], [0.0, 0.0]] {
            let row = evaluator.weights_at(face, uv).expect("weights_at");
            let cage_row: BTreeMap<u32, f64> =
                row.iter().fold(BTreeMap::new(), |mut acc, &(i, w)| {
                    let start = cage_to_refined.offsets[i as usize] as usize;
                    let end = cage_to_refined.offsets[i as usize + 1] as usize;
                    cage_to_refined.indices[start..end]
                        .iter()
                        .zip(&cage_to_refined.weights[start..end])
                        .for_each(|(&ci, &cw)| {
                            *acc.entry(ci).or_insert(0.0) += w as f64 * cw as f64;
                        });
                    acc
                });
            let total: f64 = cage_row.values().sum();
            assert!(
                (total - 1.0).abs() <= UNITY_TOLERANCE,
                "face {face} at {uv:?}: cage fold sums to {total}, not 1",
            );
            let folded = cage_row.iter().fold([0.0f64; 3], |acc, (&ci, &w)| {
                let p = case.positions[ci as usize];
                [
                    acc[0] + w * p[0] as f64,
                    acc[1] + w * p[1] as f64,
                    acc[2] + w * p[2] as f64,
                ]
            });
            let evaluated = v3(evaluator.eval(face, uv).expect("eval"));
            let distance = length(sub(folded, evaluated));
            assert!(
                distance <= IDENTITY_TOLERANCE,
                "face {face} at {uv:?}: cage fold gives {folded:?}, eval gives \
                 {evaluated:?} ({distance} apart)",
            );
        }
    }
}

// -- Deep isolation and the depth cap -----------------------------------------

/// A feature quad of the once-refined cube and the CSR corner of its
/// extraordinary (valence-3) vertex.
fn cube_ev_corner() -> (EvaluatedCase, u32, usize) {
    let case = cube_case(false);
    let ours = evaluated_case(&case, 1);
    let valence = |vi: u32| {
        ours.result.adjacency.vertex_edge_offsets[vi as usize + 1]
            - ours.result.adjacency.vertex_edge_offsets[vi as usize]
    };
    let (face, k) = (0..ours.result.topology.face_vertex_counts.len() as u32)
        .find_map(|face| {
            ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4]
                .iter()
                .position(|&c| valence(c) == 3)
                .map(|k| (face, k))
        })
        .expect("a quad touching an extraordinary vertex");
    (ours, face, k)
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
fn deep_isolation_and_depth_cap_rows_match_eval() {
    let (ours, face, k) = cube_ev_corner();
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    // ~14 isolation levels down to a regular patch.
    assert_row_gates(&evaluator, &ours.refined, face, off_corner(k, 1e-4));
    // Dyadic bits outlast the cap: the nearest-deepest-corner fallback.
    assert_row_gates(&evaluator, &ours.refined, face, off_corner(k, 1e-9));
}

// -- Clamping ------------------------------------------------------------------

#[test]
fn out_of_range_uv_rows_are_clamped() {
    let ours = evaluated_case(&cube_case(false), 1);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    assert_eq!(
        evaluator.weights_at(0, [-1.0, 2.0]).expect("clamped row"),
        evaluator.weights_at(0, [0.0, 1.0]).expect("corner row"),
    );
}

// -- Determinism ----------------------------------------------------------------

#[test]
fn repeated_rows_are_bit_identical() {
    let ours = evaluated_case(&cube_case(true), 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let face_count = ours.result.topology.face_vertex_counts.len() as u32;
    let samples: Vec<(u32, [f32; 2])> = (0..face_count)
        .filter(|&f| evaluator.quad_class(f) == QuadClass::Feature)
        .flat_map(|f| {
            [[0.5f32, 0.5], [0.3, 0.7], [1e-4, 1e-4]]
                .into_iter()
                .map(move |uv| (f, uv))
        })
        .collect();
    let first: Vec<_> = samples
        .iter()
        .map(|&(f, uv)| evaluator.weights_at(f, uv).expect("first pass"))
        .collect();
    let second: Vec<_> = samples
        .iter()
        .map(|&(f, uv)| evaluator.weights_at(f, uv).expect("second pass"))
        .collect();
    assert_eq!(first, second, "cached isolation changed a repeated row");
}
