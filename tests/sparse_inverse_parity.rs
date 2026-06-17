//! P2 sparse-vs-dense parity (compute-only), the §7 gate of the sparse design:
//! the inverse stencil map must bound every change. For a control-point edit,
//! every output that actually changes lies in `affected_outputs(edit)`, and
//! splicing the freshly-evaluated affected rows into the previous output
//! reproduces a full dense re-evaluation bit-for-bit. No tolerance -- same
//! stencils, same ops on the two inputs.

use std::collections::HashSet;
use std::num::NonZeroU8;
use subdiv_kernels::{
    RefinementResult, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine,
};

/// A closed unit cube: 8 vertices, 6 quad faces, 12 edges, no creases.
/// Faces are consistently wound; every undirected edge is shared by exactly two
/// faces (verified by construction).
fn cube() -> (Mesh, Vec<[f32; 3]>) {
    let positions = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let face_vertex_indices = vec![
        0, 1, 2, 3, // -Z
        7, 6, 5, 4, // +Z
        0, 4, 5, 1, // -Y
        3, 2, 6, 7, // +Y
        1, 5, 6, 2, // +X
        0, 3, 7, 4, // -X
    ];
    let edge_vertices = vec![
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [6, 7],
        [5, 6],
        [4, 5],
        [4, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    let topology = Mesh {
        vertex_count: 8,
        face_vertex_counts: vec![4; 6],
        face_vertex_indices,
        edge_creases: vec![0.0; edge_vertices.len()],
        edge_vertices,
        vertex_corners: vec![0.0; 8],
    };
    (topology, positions)
}

fn refine_cube(levels: u8) -> (RefinementResult, Vec<[f32; 3]>) {
    let (topology, positions) = cube();
    let refiner = Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default())
        .expect("cube topology is valid");
    let req = UniformRefine {
        levels: NonZeroU8::new(levels).expect("levels >= 1"),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).expect("refinement succeeds");
    (result, positions)
}

/// Output indices where two buffers differ (exact equality on `[f32; 3]`).
fn differing_rows(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<u32> {
    a.iter()
        .zip(b)
        .enumerate()
        .filter_map(|(i, (x, y))| (x != y).then_some(i as u32))
        .collect()
}

#[test]
fn affected_outputs_bounds_every_changed_output() {
    for levels in 1..=3u8 {
        let (result, base) = refine_cube(levels);
        let out_base = result.interpolate(&base);

        // Move each base control point in turn; the outputs that actually change
        // must all lie within affected_outputs(that input).
        for moved in 0..base.len() as u32 {
            let mut edited = base.clone();
            edited[moved as usize][0] += 0.5;
            edited[moved as usize][1] -= 0.25;
            edited[moved as usize][2] += 0.125;
            let out_edited = result.interpolate(&edited);

            let affected: HashSet<u32> = result.affected_outputs(&[moved]).into_iter().collect();
            let changed = differing_rows(&out_base, &out_edited);

            // Soundness: nothing outside `affected` may change.
            for &row in &changed {
                assert!(
                    affected.contains(&row),
                    "level {levels}, moved input {moved}: output {row} changed but is not \
                     in affected_outputs ({} affected, {} changed)",
                    affected.len(),
                    changed.len()
                );
            }
            // Non-vacuity: a moved control point changes at least one output.
            assert!(
                !changed.is_empty(),
                "level {levels}, moved input {moved}: edit changed no outputs (vacuous test)"
            );
            // Locality: the affected set is a strict subset of the dense mesh.
            assert!(
                affected.len() < out_base.len(),
                "level {levels}, moved input {moved}: affected set is not sparse ({} of {})",
                affected.len(),
                out_base.len()
            );
        }
    }
}

#[test]
fn splicing_affected_rows_reproduces_dense() {
    // The operational guarantee: starting from the previous dense output and
    // overwriting only `affected_outputs` with their freshly-evaluated values
    // equals a full dense re-evaluation, bit-for-bit. Uses the composed table so
    // the dense path is a single interpolate, cross-checking the per-level chain.
    let (result, base) = refine_cube(2);
    let composed = result.compose_stencils(base.len());

    let mut edited = base.clone();
    edited[0][0] += 1.0;

    let dense_base = composed.interpolate(&base);
    let dense_full = composed.interpolate(&edited);

    let affected = result.affected_outputs(&[0]);

    // Real sparse update: recompute only the affected rows from the edited
    // inputs, straight into the previous output buffer.
    let mut sparse = dense_base.clone();
    composed.interpolate_rows(&edited, &affected, &mut sparse);

    assert_eq!(
        sparse, dense_full,
        "sparse update via interpolate_rows over affected_outputs([0]) did not \
         reproduce the dense re-evaluation -- affected set unsound or row eval wrong"
    );
}
