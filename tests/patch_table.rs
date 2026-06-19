//! Regular-patch gates that need no external oracle (limit-surface SDF
//! design s1).
//!
//! Three properties pin the patch extraction and the B-spline basis:
//!
//! - **Coverage**: a refined quad is regular exactly when all four of
//!   its corners are interior, valence-4, and free of sharpness -- on a
//!   once-refined open grid that is every quad not touching a boundary
//!   vertex; on a once-refined cube no quad qualifies (every quad
//!   touches a valence-3 corner descendant); after two refinements the
//!   extraordinary vertices are isolated and the interior quads become
//!   regular.
//! - **Affine reproduction**: a uniform cubic B-spline reproduces
//!   affine data, so on a flat uniformly spaced grid a patch must be
//!   exactly the bilinear chart of its quad -- this pins the
//!   `(u, v)` orientation (`u` along corner 0 -> 1, `v` along corner
//!   0 -> 3) and the in-quad derivative scale.
//! - **Limit-stencil consistency**: at the four patch corners the
//!   B-spline corner evaluation and the per-vertex limit stencils are
//!   two independent routes to the same limit point.

use std::num::NonZeroU8;
use subdiv_kernels::{
    QuadClass, RefinementResult, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine,
};

/// B-spline corner eval vs the per-vertex limit position stencils.
const CORNER_TOLERANCE: f32 = 1e-5;

// -- Cage builders (the `tests/limit_surface.rs` layouts) ---------------

/// An `n` x `n` quad grid in the y = height(i, j) sheet over the xz
/// plane.
fn quad_grid(n: u32, height: impl Fn(u32, u32) -> f32) -> (Mesh, Vec<[f32; 3]>) {
    let stride = n + 1;
    let height = &height;
    let positions: Vec<[f32; 3]> = (0..stride)
        .flat_map(|i| (0..stride).map(move |j| [i as f32, height(i, j), j as f32]))
        .collect();

    let vid = |i: u32, j: u32| i * stride + j;
    let face_vertex_indices: Vec<u32> = (0..n)
        .flat_map(|i| {
            (0..n).flat_map(move |j| [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
        })
        .collect();

    let topo = Mesh {
        vertex_count: stride * stride,
        face_vertex_counts: vec![4; (n * n) as usize],
        face_vertex_indices,
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; (stride * stride) as usize],
    };
    (topo, positions)
}

/// Unit-ish cube with outward winding.
fn cube() -> (Mesh, Vec<[f32; 3]>) {
    let positions: Vec<[f32; 3]> = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let face_vertex_indices: Vec<u32> = vec![
        0, 3, 2, 1, // z = -1, seen from below.
        4, 5, 6, 7, // z = +1.
        0, 1, 5, 4, // y = -1.
        1, 2, 6, 5, // x = +1.
        2, 3, 7, 6, // y = +1.
        3, 0, 4, 7, // x = -1.
    ];
    let topo = Mesh {
        vertex_count: 8,
        face_vertex_counts: vec![4; 6],
        face_vertex_indices,
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; 8],
    };
    (topo, positions)
}

fn refine(topo: Mesh, levels: u8) -> RefinementResult {
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default())
        .expect("refiner");
    refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(levels).expect("non-zero"),
        ))
        .expect("refinement")
}

/// Valence-3 refined vertices (the cube's extraordinary vertices).
fn valence_three(result: &RefinementResult) -> Vec<bool> {
    (0..result.topology.vertex_count as usize)
        .map(|vi| {
            let start = result.adjacency.vertex_edge_offsets[vi];
            let end = result.adjacency.vertex_edge_offsets[vi + 1];
            end - start == 3
        })
        .collect()
}

// -- Coverage ------------------------------------------------------------

#[test]
fn grid_interior_quads_are_regular_boundary_adjacent_are_feature() {
    let (topo, _) = quad_grid(6, |_, _| 0.0);
    let result = refine(topo, 1);
    let table = result.patch_table().expect("patch table");

    // A once-refined open grid has no extraordinary vertices and no
    // creases, so the only feature is the boundary: a quad is regular
    // exactly when none of its corners is a boundary vertex.
    let face_count = result.topology.face_vertex_counts.len();
    assert_eq!(face_count, 144);
    for face in 0..face_count {
        let corners = &result.topology.face_vertex_indices[face * 4..face * 4 + 4];
        let touches_boundary = corners
            .iter()
            .any(|&c| result.adjacency.vertex_is_boundary[c as usize]);
        let expected = if touches_boundary {
            QuadClass::Feature
        } else {
            QuadClass::Regular
        };
        assert_eq!(
            table.quad_class(face as u32),
            expected,
            "refined quad {face}",
        );
        assert_eq!(
            table.face_patch(face as u32).is_some(),
            expected == QuadClass::Regular,
            "refined quad {face}",
        );
    }

    // 12x12 refined quads; the 10x10 interior block is regular.
    assert_eq!(table.control_points.len(), 100);
    assert_eq!(table.faces.len(), 100);
    for (patch, &face) in table.faces.iter().enumerate() {
        assert_eq!(table.face_patch(face), Some(patch as u32));
    }
}

#[test]
fn once_refined_cube_has_no_regular_patches() {
    let (topo, _) = cube();
    let result = refine(topo, 1);
    let table = result.patch_table().expect("patch table");

    // Every once-refined cube quad touches a valence-3 corner
    // descendant.
    assert!(table.control_points.is_empty());
    assert!(table.faces.is_empty());
    assert_eq!(result.topology.face_vertex_counts.len(), 24);
    for face in 0..24 {
        assert_eq!(table.quad_class(face), QuadClass::Feature);
    }
}

#[test]
fn twice_refined_cube_isolates_extraordinary_vertices() {
    let (topo, _) = cube();
    let result = refine(topo, 2);
    let table = result.patch_table().expect("patch table");

    // 96 refined quads; the 8 valence-3 vertices have 3 incident quads
    // each and are isolated (no quad touches two), so 24 quads are
    // feature and 72 regular.
    let face_count = result.topology.face_vertex_counts.len();
    assert_eq!(face_count, 96);
    assert_eq!(table.control_points.len(), 72);

    let extraordinary = valence_three(&result);
    for face in 0..face_count {
        let corners = &result.topology.face_vertex_indices[face * 4..face * 4 + 4];
        let touches_ev = corners.iter().any(|&c| extraordinary[c as usize]);
        let expected = if touches_ev {
            QuadClass::Feature
        } else {
            QuadClass::Regular
        };
        assert_eq!(
            table.quad_class(face as u32),
            expected,
            "refined quad {face}",
        );
    }
}

// -- Affine reproduction ---------------------------------------------------

#[test]
fn flat_grid_patch_is_the_bilinear_chart_of_its_quad() {
    let (topo, positions) = quad_grid(6, |_, _| 0.0);
    let result = refine(topo, 1);
    let table = result.patch_table().expect("patch table");
    let refined = result.interpolate(&positions);

    // A central patch, away from the cage corners (the only refined
    // vertices a flat grid displaces): the quad covering
    // x, z in [3.0, 3.5].
    let (patch, face) = table
        .faces
        .iter()
        .enumerate()
        .find(|&(_, &face)| {
            let corners = &result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
            corners.iter().all(|&c| {
                let p = refined[c as usize];
                (3.0..=3.5).contains(&p[0]) && (3.0..=3.5).contains(&p[2])
            })
        })
        .map(|(patch, &face)| (patch, face))
        .expect("central patch");

    let corners = &result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
    let c0 = refined[corners[0] as usize];
    let c1 = refined[corners[1] as usize];
    let c3 = refined[corners[3] as usize];

    for &(u, v) in &[(0.0f32, 0.0f32), (1.0, 0.0), (0.5, 0.5), (0.3, 0.7), (0.85, 0.15)] {
        let (position, du, dv) = table.eval_with_derivatives(patch, [u, v], &refined);
        for c in 0..3 {
            let expected = c0[c] + u * (c1[c] - c0[c]) + v * (c3[c] - c0[c]);
            assert!(
                (position[c] - expected).abs() < 1e-6,
                "patch {patch} at ({u}, {v}): position {position:?} is not the bilinear chart",
            );
            assert!(
                (du[c] - (c1[c] - c0[c])).abs() < 1e-6,
                "patch {patch} at ({u}, {v}): du {du:?} is not corner 0 -> 1",
            );
            assert!(
                (dv[c] - (c3[c] - c0[c])).abs() < 1e-6,
                "patch {patch} at ({u}, {v}): dv {dv:?} is not corner 0 -> 3",
            );
        }
    }
}

// -- Limit-stencil consistency ---------------------------------------------

/// Patch corner evals against the per-vertex limit stencils: two
/// independent routes to the same limit point.
fn assert_corners_match_limit_stencils(result: &RefinementResult, positions: &[[f32; 3]]) {
    let table = result.patch_table().expect("patch table");
    assert!(!table.faces.is_empty(), "case has no regular patches");
    let refined = result.interpolate(positions);
    let limit = result.limit_stencils().expect("limit stencils");
    let limit_positions = limit.position.interpolate(&refined);

    for (patch, &face) in table.faces.iter().enumerate() {
        let corners = &result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4];
        for (k, &(u, v)) in [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
        {
            let evaluated = table.eval(patch, [u, v], &refined);
            let stencil = limit_positions[corners[k] as usize];
            for c in 0..3 {
                assert!(
                    (evaluated[c] - stencil[c]).abs() < CORNER_TOLERANCE,
                    "patch {patch} corner {k}: eval {evaluated:?} vs limit stencil {stencil:?}",
                );
            }
        }
    }
}

#[test]
fn cube_patch_corners_match_limit_stencils() {
    let (topo, positions) = cube();
    let result = refine(topo, 2);
    assert_corners_match_limit_stencils(&result, &positions);
}

#[test]
fn warped_grid_patch_corners_match_limit_stencils() {
    let (topo, positions) = quad_grid(6, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    });
    let result = refine(topo, 1);
    assert_corners_match_limit_stencils(&result, &positions);
}
