//! Regular B-spline patches vs the OpenSubdiv oracle (limit-surface
//! SDF design s1, gates per its §7).
//!
//! Every regular refined quad's bicubic patch must evaluate the exact
//! limit surface, at the corners *and* at arbitrary interior `(u, v)`.
//! The refined quad -> ptex sub-rectangle correspondence is recovered
//! convention-free, as in the sectored gate of
//! `tests/limit_osd_oracle.rs`: each quad corner's limit position is
//! matched against the level-`L` lattice samples of its root ptex face
//! (`face_root`), which yields the affine in-quad `(u, v)` -> ptex
//! `(s, t)` map (asserted to be an axis-aligned `(1/2^L)^2` square
//! cell, possibly rotated/mirrored). The oracle is then evaluated at
//! the mapped images of corner, edge, and strictly interior non-lattice
//! samples; positions must agree within `1e-4` and `du`/`dv` must agree
//! through the affine chain rule -- directions within 0.5 degrees and
//! magnitudes to 1%.
//!
//! Cases: a warped open 6x6 grid at level 1 (regular patches whose 4x4
//! neighborhoods reach boundary vertices), a cube at level 2 (patches
//! around isolated extraordinary vertices), and a cube with an
//! infinitely sharp top rim at level 2 (patches whose neighborhoods
//! reach crease vertices).

mod common;

use common::{
    Case, POSITION_TOLERANCE, add, angle_deg, cube_case, grid_case, lattice_locations, length,
    oracle_at_frame_samples, oracle_samples, recover_ptex_frames, scale, sub, v3,
};
use std::num::NonZeroU8;
use subdiv_kernels::{PatchTable, RefinementResult, Refiner, Scheme, UniformRefine};

const DERIVATIVE_TOLERANCE_DEG: f64 = 0.5;
const DERIVATIVE_MAGNITUDE_RELATIVE_TOLERANCE: f64 = 1e-2;

/// In-quad samples: the four corners (lattice points at the evaluated
/// level), an edge midpoint, the center, and asymmetric strictly
/// interior non-lattice points.
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

/// Our refinement + patch table + the data both gates share.
struct PatchedCase {
    result: RefinementResult,
    table: PatchTable,
    /// Refined control positions (the patch eval input).
    refined: Vec<[f32; 3]>,
    /// Per-refined-vertex limit positions (for lattice matching).
    limit_positions: Vec<[f32; 3]>,
}

fn patched_case(case: &Case, level: u8) -> PatchedCase {
    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement");
    let table = result.patch_table().expect("patch table");
    let refined = result.interpolate(&case.positions);
    let limit = result.limit_stencils().expect("limit stencils");
    let limit_positions = limit.position.interpolate(&refined);
    PatchedCase {
        result,
        table,
        refined,
        limit_positions,
    }
}

/// The s1 oracle gate: positions and chain-ruled derivatives of every
/// regular patch against petite at all `UV_SAMPLES` (frame recovery
/// and oracle plumbing shared with the s2 gates via `common`).
fn assert_patches_match_oracle(case: &Case, level: u8, expected_regular: usize) {
    let ours = patched_case(case, level);
    assert_eq!(
        ours.table.faces.len(),
        expected_regular,
        "unexpected regular patch count",
    );
    let lattice = oracle_samples(case, &lattice_locations(case.face_count(), level));
    let frames = recover_ptex_frames(
        &lattice,
        level,
        &ours.table.faces,
        &ours.result.face_root,
        &ours.result.topology.face_vertex_indices,
        &ours.limit_positions,
    );
    let (oracle, rows) = oracle_at_frame_samples(case, &frames, &UV_SAMPLES);

    for (patch, (frame, patch_rows)) in frames.iter().zip(&rows).enumerate() {
        for (uv, &row) in UV_SAMPLES.iter().zip(patch_rows) {
            let (position, du, dv) =
                ours.table
                    .eval_with_derivatives(patch, *uv, &ours.refined);
            let sample = &oracle[row];

            let distance = length(sub(v3(position), sample.position));
            assert!(
                distance <= POSITION_TOLERANCE,
                "patch {patch} at {uv:?}: position {position:?} is {distance} from oracle \
                 {:?} (ptex face {})",
                sample.position,
                frame.root,
            );

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
                    "patch {patch} at {uv:?}: {name} {evaluated:?} is {angle} degrees off the \
                     chain-ruled oracle derivative {expected:?}",
                );
                let magnitude_error = (length(evaluated) - length(expected)).abs();
                assert!(
                    magnitude_error
                        <= DERIVATIVE_MAGNITUDE_RELATIVE_TOLERANCE * length(expected),
                    "patch {patch} at {uv:?}: {name} magnitude {} vs oracle {}",
                    length(evaluated),
                    length(expected),
                );
            }
        }
    }
}

#[test]
fn warped_grid_patches_match_opensubdiv_oracle() {
    let case = grid_case(6, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    });
    // 12x12 refined quads, 10x10 of them away from the boundary.
    assert_patches_match_oracle(&case, 1, 100);
}

#[test]
fn cube_patches_match_opensubdiv_oracle() {
    // 96 refined quads minus 3 per valence-3 vertex.
    assert_patches_match_oracle(&cube_case(false), 2, 72);
}

#[test]
fn creased_cube_patches_match_opensubdiv_oracle() {
    // Top face: 16 - 12 rim-touching; sides: 16 - 4 rim-touching - 2
    // EV corners; bottom: 16 - 4 EV corners. 4 + 4 * 10 + 12 = 56.
    assert_patches_match_oracle(&cube_case(true), 2, 56);
}
