//! Feature-quad limit evaluation vs the OpenSubdiv oracle
//! (limit-surface SDF design s2, gates per its §7).
//!
//! Every *feature* refined quad -- the s1 classification complement --
//! must evaluate the exact limit surface through the recursive
//! isolation of `LimitEvaluator`, at the corners, on the edges, and at
//! arbitrary interior `(u, v)`. The refined quad -> ptex
//! sub-rectangle correspondence is recovered convention-free from
//! corner limit positions (`common::recover_ptex_frames`, shared with
//! the s1 gate), the oracle is evaluated at the mapped images of the
//! sample set, and positions must agree within `1e-4` with `du`/`dv`
//! agreeing through the affine chain rule -- directions within 0.5
//! degrees and magnitudes to 1%.
//!
//! Cases: a cube at level 1 (every quad is feature -- EV isolation
//! through the submesh recursion) and a cube with an infinitely sharp
//! top rim at level 2 (crease-line snapping: corner and edge samples
//! on the rim resolve to the per-sector crease masks, whose tangents
//! are exactly the crease patches' parametric derivatives).
//!
//! # Documented exclusions (positions still asserted)
//!
//! Derivative comparisons are skipped at corner samples whose vertex
//! is a *smooth extraordinary vertex* -- one excluded corner per cube
//! L1 quad (24) and one per bottom-EV-touching creased-cube quad (12).
//! The parametric derivative does not converge there: approaching a
//! valence-n EV, patch derivatives scale by `(2 lambda)^depth` per
//! isolation level (`lambda ~ 0.41` at valence 3, so they vanish at
//! the corner; they diverge for `n > 4`), while the oracle reports
//! finite values from its end-cap approximation whose tangents drift
//! with the adaptive isolation level (the end-cap drift probed in
//! `tests/limit_osd_oracle.rs`). Our evaluator returns the analytic
//! sector tangent plane there instead (correct normal, no parametric
//! scale; `tests/limit_eval.rs` gates it against the sectored
//! stencils). Cone points (pinned multi-face sectors, the other
//! non-converging family) do not occur in these cages -- their
//! exclusion story lives with the sectored-stencil gates.
//!
//! The deep-recursion gate (a query 1e-4 off an EV corner, ~14
//! isolation levels) also compares position only: the sample sits
//! inside the oracle's end-cap region (2^-8 of the ptex face at
//! isolation 8).

mod common;

use common::{
    Case, POSITION_TOLERANCE, add, angle_deg, cube_case, lattice_locations, length,
    oracle_at_frame_samples, oracle_samples, recover_ptex_frames, scale, sub, v3,
};
use std::num::NonZeroU8;
use subdiv_kernels::{
    LimitEvaluator, QuadClass, RefinementResult, Refiner, Scheme, UniformRefine,
};

const DERIVATIVE_TOLERANCE_DEG: f64 = 0.5;
const DERIVATIVE_MAGNITUDE_RELATIVE_TOLERANCE: f64 = 1e-2;

/// In-quad samples per the s2 brief: the four corners, an edge
/// midpoint, the center, and asymmetric strictly interior points.
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

struct FeatureCase {
    result: RefinementResult,
    refined: Vec<[f32; 3]>,
    limit_positions: Vec<[f32; 3]>,
    feature_faces: Vec<u32>,
}

fn feature_case(case: &Case, level: u8) -> FeatureCase {
    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement");
    let refined = result.interpolate(&case.positions);
    let limit = result.limit_stencils().expect("limit stencils");
    let limit_positions = limit.position.interpolate(&refined);
    let table = result.patch_table().expect("patch table");
    let feature_faces: Vec<u32> = (0..result.topology.face_vertex_counts.len() as u32)
        .filter(|&face| table.quad_class(face) == QuadClass::Feature)
        .collect();
    FeatureCase {
        result,
        refined,
        limit_positions,
        feature_faces,
    }
}

/// Whether refined vertex `vi` is a smooth extraordinary vertex (the
/// derivative-exclusion predicate of the module docs).
fn smooth_extraordinary(ours: &FeatureCase, vi: u32) -> bool {
    let start = ours.result.adjacency.vertex_edge_offsets[vi as usize] as usize;
    let end = ours.result.adjacency.vertex_edge_offsets[vi as usize + 1] as usize;
    end - start != 4
        && !ours.result.adjacency.vertex_is_boundary[vi as usize]
        && ours.result.adjacency.vertex_edges[start..end]
            .iter()
            .all(|&e| ours.result.topology.edge_creases[e as usize] <= 0.0)
}

/// The s2 oracle gate: positions and chain-ruled derivatives of every
/// feature quad against petite at all `UV_SAMPLES`; returns the number
/// of excluded derivative comparisons.
fn assert_feature_quads_match_oracle(case: &Case, level: u8, expected_feature: usize) -> usize {
    let ours = feature_case(case, level);
    assert_eq!(
        ours.feature_faces.len(),
        expected_feature,
        "unexpected feature quad count",
    );
    let evaluator: LimitEvaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let lattice = oracle_samples(case, &lattice_locations(case.face_count(), level));
    let frames = recover_ptex_frames(
        &lattice,
        level,
        &ours.feature_faces,
        &ours.result.face_root,
        &ours.result.topology.face_vertex_indices,
        &ours.limit_positions,
    );
    let (oracle, rows) = oracle_at_frame_samples(case, &frames, &UV_SAMPLES);

    let mut excluded = 0;
    for ((&face, frame), face_rows) in ours.feature_faces.iter().zip(&frames).zip(&rows) {
        let corners =
            &ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
        for (k, (uv, &row)) in UV_SAMPLES.iter().zip(face_rows).enumerate() {
            let (position, du, dv) = evaluator
                .eval_with_derivatives(face, *uv)
                .expect("feature eval");
            let sample = &oracle[row];

            let distance = length(sub(v3(position), sample.position));
            assert!(
                distance <= POSITION_TOLERANCE,
                "face {face} at {uv:?}: position {position:?} is {distance} from oracle \
                 {:?} (ptex face {})",
                sample.position,
                frame.root,
            );

            // s5-p1 W-row check: the `weights_at` stencil reproduces
            // the oracle position through the refined positions.
            let weights = evaluator.weights_at(face, *uv).expect("weights row");
            let folded = weights.iter().fold([0.0f64; 3], |acc, &(i, w)| {
                let p = ours.refined[i as usize];
                [
                    acc[0] + w as f64 * p[0] as f64,
                    acc[1] + w as f64 * p[1] as f64,
                    acc[2] + w as f64 * p[2] as f64,
                ]
            });
            let distance = length(sub(folded, sample.position));
            assert!(
                distance <= POSITION_TOLERANCE,
                "face {face} at {uv:?}: weights row gives {folded:?}, {distance} from \
                 oracle {:?}",
                sample.position,
            );

            // Corner samples at smooth EVs: no convergent parametric
            // derivative on either side (see the module docs).
            if k < 4 && smooth_extraordinary(&ours, corners[k]) {
                excluded += 1;
                continue;
            }

            // Chain rule through the affine (u, v) -> (s, t) map: the
            // oracle's ds/dt derivatives combine with the frame columns.
            let expected_du = add(scale(sample.du, frame.e_u[0]), scale(sample.dv, frame.e_u[1]));
            let expected_dv = add(scale(sample.du, frame.e_v[0]), scale(sample.dv, frame.e_v[1]));
            for (name, evaluated, expected) in
                [("du", v3(du), expected_du), ("dv", v3(dv), expected_dv)]
            {
                let angle = angle_deg(evaluated, expected);
                assert!(
                    angle <= DERIVATIVE_TOLERANCE_DEG,
                    "face {face} at {uv:?}: {name} {evaluated:?} is {angle} degrees off the \
                     chain-ruled oracle derivative {expected:?}",
                );
                let magnitude_error = (length(evaluated) - length(expected)).abs();
                assert!(
                    magnitude_error
                        <= DERIVATIVE_MAGNITUDE_RELATIVE_TOLERANCE * length(expected),
                    "face {face} at {uv:?}: {name} magnitude {} vs oracle {}",
                    length(evaluated),
                    length(expected),
                );
            }
        }
    }
    excluded
}

#[test]
fn cube_feature_quads_match_opensubdiv_oracle() {
    // Once-refined cube: all 24 quads touch a valence-3 vertex; one
    // smooth-EV corner exclusion each.
    let excluded = assert_feature_quads_match_oracle(&cube_case(false), 1, 24);
    assert_eq!(excluded, 24, "unexpected smooth-EV exclusion count");
}

#[test]
fn creased_cube_feature_quads_match_opensubdiv_oracle() {
    // 96 refined quads, 56 regular (the s1 gate); 40 feature: 28
    // rim-touching plus 12 around the four bottom EVs -- the latter
    // contribute one smooth-EV corner exclusion each, while every
    // crease corner/edge sample is fully derivative-checked through
    // the per-sector snap.
    let excluded = assert_feature_quads_match_oracle(&cube_case(true), 2, 40);
    assert_eq!(excluded, 12, "unexpected smooth-EV exclusion count");
}

#[test]
fn deep_ev_query_matches_opensubdiv_oracle() {
    // The s2 deep-recursion gate: (1e-4, 1e-4) off an EV corner
    // isolates ~14 levels down; position against the oracle, 1e-4
    // (derivatives are inside the end-cap region; see the module docs).
    let case = cube_case(false);
    let ours = feature_case(&case, 1);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let lattice = oracle_samples(&case, &lattice_locations(case.face_count(), 1));
    let frames = recover_ptex_frames(
        &lattice,
        1,
        &ours.feature_faces,
        &ours.result.face_root,
        &ours.result.topology.face_vertex_indices,
        &ours.limit_positions,
    );

    let (slot, face, k) = ours
        .feature_faces
        .iter()
        .enumerate()
        .find_map(|(slot, &face)| {
            ours.result.topology.face_vertex_indices
                [face as usize * 4..face as usize * 4 + 4]
                .iter()
                .position(|&c| smooth_extraordinary(&ours, c))
                .map(|k| (slot, face, k))
        })
        .expect("a quad corner at an extraordinary vertex");
    let corner = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][k];
    let uv = [
        if corner[0] == 0.0 { 1e-4 } else { 1.0 - 1e-4 },
        if corner[1] == 0.0 { 1e-4 } else { 1.0 - 1e-4 },
    ];
    let (oracle, rows) = oracle_at_frame_samples(&case, &frames[slot..slot + 1], &[uv]);

    let position = evaluator.eval(face, uv).expect("deep eval");
    let sample = &oracle[rows[0][0]];
    let distance = length(sub(v3(position), sample.position));
    assert!(
        distance <= POSITION_TOLERANCE,
        "face {face} at {uv:?}: deep position {position:?} is {distance} from oracle {:?}",
        sample.position,
    );
}
