use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, Mesh, Refiner, Scheme, SchemeOptions,
    UniformRefine, VertexOrigin,
};
use core::num::NonZeroU8;

fn cube_topology() -> Mesh {
    Mesh {
        vertex_count: 8,
        face_vertex_counts: vec![4; 6],
        face_vertex_indices: vec![
            0, 1, 2, 3, // front
            4, 5, 6, 7, // back
            0, 1, 5, 4, // bottom
            2, 3, 7, 6, // top
            0, 3, 7, 4, // left
            1, 2, 6, 5, // right
        ],
        edge_vertices: vec![
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        edge_creases: vec![0.0; 12],
        vertex_corners: vec![0.0; 8],
    }
}

fn cube_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
}

/// On a closed cube (every vertex interior, no seam) the smooth modes must
/// reproduce the positional refinement at level 1 — the strong parity oracle.
/// This also guards the regression where CC's edge/vertex face-varying points
/// dropped their face-centroid terms (a pure-midpoint edge instead of the CC
/// smooth edge point).
fn assert_no_seam_matches_positional(mode: FaceVaryingInterpolation) {
    let topo = cube_topology();
    let positions = cube_positions();
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();
    let refined_pos = result.interpolate(&positions);
    let tables = refiner.face_varying_stencils(&req, &channel, mode).unwrap();
    let fvar_out = tables
        .iter()
        .fold(positions.clone(), |d, t| t.interpolate(&d));
    for (c, &v) in result.topology.face_vertex_indices.iter().enumerate() {
        assert!(
            close3(fvar_out[c], refined_pos[v as usize]),
            "{mode:?} corner {c} (vertex {v}): fvar {:?} != positional {:?}",
            fvar_out[c],
            refined_pos[v as usize],
        );
    }
}

/// Seam-along-a-crease oracle: a position-valued channel seamed around the
/// cube's front face, refined under a smooth mode, must equal the positional
/// refinement with those four edges marked as infinite creases.
fn assert_smooth_crease_matches_creased_positional(mode: FaceVaryingInterpolation) {
    let topo = cube_topology();
    let positions = cube_positions();
    // Front-face boundary loop: each loop vertex has exactly two seam edges.
    let seam = [[0u32, 1], [1, 2], [2, 3], [0, 3]];
    let mut creased = topo.clone();
    creased.edge_creases = crate::test_support::creases_for(&topo, &seam);
    let refiner = Refiner::new(creased, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();
    let refined_pos = result.interpolate(&positions);
    let (channel, values) = crate::test_support::seamed_position_channel(&topo, &seam, &positions);
    let tables = refiner.face_varying_stencils(&req, &channel, mode).unwrap();
    let fvar_out = tables.iter().fold(values, |d, t| t.interpolate(&d));
    for (c, &v) in result.topology.face_vertex_indices.iter().enumerate() {
        assert!(
            close3(fvar_out[c], refined_pos[v as usize]),
            "{mode:?} corner {c} (vertex {v}): fvar {:?} != creased-positional {:?}",
            fvar_out[c],
            refined_pos[v as usize],
        );
    }
}

#[test]
fn fvar_boundaries_no_seam_matches_positional() {
    assert_no_seam_matches_positional(FaceVaryingInterpolation::SmoothWithLinearBoundaries);
}

#[test]
fn fvar_smooth_no_seam_matches_positional() {
    assert_no_seam_matches_positional(FaceVaryingInterpolation::Smooth);
}

#[test]
fn fvar_corners_only_no_seam_matches_positional() {
    assert_no_seam_matches_positional(FaceVaryingInterpolation::SmoothWithLinearCorners);
}

// Only `Smooth` admits the seam-along-a-crease parity oracle on the cube: the
// cube's only 4-edge loops are face boundaries, so one side is a single-face
// chart whose values are face-varying corners — `SmoothWithLinearCorners`
// pins those (≠ positional crease) by design. Its corner behavior is covered
// by `fvar_corners_only_pins_corner_unlike_smooth` and by the octahedron
// crease-parity tests for Loop/√3 (closed seam loops, no corners).
#[test]
fn fvar_smooth_crease_matches_creased_positional() {
    assert_smooth_crease_matches_creased_positional(FaceVaryingInterpolation::Smooth);
}

/// Two quads sharing edge (1,2), a single fvar chart. Boundary edges are
/// seams; corner vertex 0 (valence-1) sits on two of them. Under `Smooth` it
/// follows the crease curve (3/4·v0 + 1/8·v1 + 1/8·v3); under
/// `SmoothWithLinearCorners` the corner is pinned (copy of v0). The two modes
/// must therefore disagree at vertex 0's child vertex-point.
#[test]
fn fvar_corners_only_pins_corner_unlike_smooth() {
    let topo = Mesh {
        vertex_count: 6,
        face_vertex_counts: vec![4, 4],
        face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 5, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [4, 5], [2, 5]],
        edge_creases: vec![0.0; 7],
        vertex_corners: vec![0.0; 6],
    };
    let positions: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();

    let eval = |mode| {
        let tables = refiner.face_varying_stencils(&req, &channel, mode).unwrap();
        tables
            .iter()
            .fold(positions.clone(), |d, t| t.interpolate(&d))
    };
    let smooth = eval(FaceVaryingInterpolation::Smooth);
    let corners = eval(FaceVaryingInterpolation::SmoothWithLinearCorners);

    // Locate vertex 0's child vertex-point corner.
    let c0 = result
        .topology
        .face_vertex_indices
        .iter()
        .position(|&v| {
            matches!(
                result.lineage.vertex_origin[v as usize],
                VertexOrigin::Vertex(0)
            )
        })
        .expect("vertex-0 child point present");

    // CornersOnly pins the valence-1 corner to v0 = [0,0,0].
    assert!(
        close3(corners[c0], [0.0, 0.0, 0.0]),
        "corner-pin got {:?}",
        corners[c0]
    );
    // Smooth follows the crease curve: 3/4·[0,0,0] + 1/8·[1,0,0] + 1/8·[0,1,0].
    assert!(
        close3(smooth[c0], [0.125, 0.125, 0.0]),
        "crease curve got {:?}",
        smooth[c0]
    );
}

fn quad_topology() -> Mesh {
    Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![4],
        face_vertex_indices: vec![0, 1, 2, 3],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [0, 3]],
        edge_creases: vec![0.0; 4],
        vertex_corners: vec![0.0; 4],
    }
}

fn quad_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
}

fn positions_match(a: &[[f32; 3]], b: &[[f32; 3]], eps: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(pa, pb)| pa.iter().zip(pb.iter()).all(|(x, y)| (x - y).abs() < eps))
}

#[test]
fn quad_refines_to_four_quads() {
    let topo = quad_topology();
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(1).unwrap(),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.topology.vertex_count, 9);
    assert_eq!(result.topology.face_vertex_counts.len(), 4);
    assert!(result.topology.face_vertex_counts.iter().all(|&n| n == 4));
}

#[test]
fn level2_produces_correct_vertex_count() {
    let topo = quad_topology();
    let pos = quad_positions();
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(2).unwrap(),
            ..Default::default()
        })
        .unwrap();

    let out = result.interpolate(&pos);
    assert_eq!(out.len(), result.topology.vertex_count as usize);
    // Level 2 of a single quad: 25 vertices.
    assert_eq!(result.topology.vertex_count, 25);
}

#[test]
fn crease_propagated_through_stencils() {
    let mut topo = quad_topology();
    topo.edge_creases[0] = 2.0;
    topo.vertex_corners[0] = 2.0;

    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(1).unwrap(),
            ..Default::default()
        })
        .unwrap();

    let creased_edges = result
        .topology
        .edge_creases
        .iter()
        .filter(|&&c| c > 0.0)
        .count();
    assert!(creased_edges >= 2);

    // Vertex point for corner vertex should have propagated corner value.
    // CC layout: [face_pts(1), edge_pts(4), vertex_pts(4)]. Vertex 0 is at index 5.
    assert!(result.topology.vertex_corners[5] > 0.0);
}

#[test]
fn interpolate_f64_positions() {
    let topo = quad_topology();
    let pos: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];

    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(1).unwrap(),
            ..Default::default()
        })
        .unwrap();

    let out = result.interpolate(&pos);
    assert_eq!(out.len(), 9);
    // Face point = centroid = (0.5, 0.5, 0.0)
    assert!((out[0][0] - 0.5).abs() < 1e-10);
    assert!((out[0][1] - 0.5).abs() < 1e-10);
}

// ── Face-varying stencil tests ────────────────────────────────────────

#[test]
fn fvar_all_linear_quad_level1() {
    let topo = quad_topology();
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3],
        value_count: 4,
    };
    let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine {
                levels: NonZeroU8::new(1).unwrap(),
                ..Default::default()
            },
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();

    let out = tables
        .iter()
        .fold(uvs.clone(), |data, t| t.interpolate(&data));
    assert_eq!(out.len(), 16);

    // Face-point UV = centroid (0.5, 0.5)
    assert!((out[0][0] - 0.5).abs() < 1e-5);
    assert!((out[0][1] - 0.5).abs() < 1e-5);

    // Vertex-point (AllLinear) = copy at index 2
    assert!((out[2][0] - 0.0).abs() < 1e-5);
    assert!((out[2][1] - 0.0).abs() < 1e-5);
}

#[test]
fn fvar_all_linear_preserves_seam() {
    let topo = Mesh {
        vertex_count: 6,
        face_vertex_counts: vec![4, 4],
        face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 5, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [4, 5], [2, 5]],
        edge_creases: vec![0.0; 7],
        vertex_corners: vec![0.0; 6],
    };

    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
        value_count: 8,
    };

    let uvs: Vec<[f32; 2]> = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];

    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine {
                levels: NonZeroU8::new(1).unwrap(),
                ..Default::default()
            },
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();

    let out = tables
        .iter()
        .fold(uvs.clone(), |data, t| t.interpolate(&data));
    assert_eq!(out.len(), 32);

    // Face 0 centroid
    assert!((out[0][0] - 0.5).abs() < 1e-5);
    // Face 1 centroid (starts at index 16)
    assert!((out[16][0] - 0.5).abs() < 1e-5);
}

#[test]
fn fvar_boundaries_matches_all_linear_on_single_quad() {
    let topo = quad_topology();
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3],
        value_count: 4,
    };
    let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(1).unwrap(),
        ..Default::default()
    };

    let al_tables = refiner
        .face_varying_stencils(&req, &fvar, FaceVaryingInterpolation::Linear)
        .unwrap();
    let al = al_tables
        .iter()
        .fold(uvs.clone(), |data, t| t.interpolate(&data));

    let bd_tables = refiner
        .face_varying_stencils(
            &req,
            &fvar,
            FaceVaryingInterpolation::SmoothWithLinearBoundaries,
        )
        .unwrap();
    let bd = bd_tables
        .iter()
        .fold(uvs.clone(), |data, t| t.interpolate(&data));

    assert_eq!(al.len(), bd.len());
    al.iter()
        .zip(bd.iter())
        .enumerate()
        .for_each(|(i, (a, b))| {
            assert!(
                (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5,
                "mismatch at {i}: AllLinear={a:?} Boundaries={b:?}"
            );
        });
}

#[test]
fn face_root_chains_to_the_base_mesh() {
    // Two quads sharing an edge; level 2 -> 8 -> 32 refined faces.
    let topo = Mesh {
        vertex_count: 6,
        face_vertex_counts: vec![4, 4],
        face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 5, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [4, 5], [2, 5]],
        edge_creases: vec![0.0; 7],
        vertex_corners: vec![0.0; 6],
    };
    let req = UniformRefine {
        levels: NonZeroU8::new(2).unwrap(),
        ..Default::default()
    };
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).unwrap();
    let result = refiner.refine_uniform(&req).unwrap();

    // Every refined face roots in the base mesh, 16 descendants per quad.
    assert_eq!(result.face_root.len(), 32);
    assert_eq!(result.face_root.iter().filter(|&&r| r == 0).count(), 16);
    assert_eq!(result.face_root.iter().filter(|&&r| r == 1).count(), 16);

    // And it equals the manual per-level lineage fold adapters would
    // otherwise have to do themselves.
    let refined = refiner.refine_topology(&req).unwrap();
    let mut root: Vec<u32> = (0..2).collect();
    for step in 0..refined.refinement_steps() {
        let lineage = refined.level_lineage(step).unwrap();
        root = lineage
            .face_parent
            .iter()
            .map(|&parent| root[parent as usize])
            .collect();
    }
    assert_eq!(result.face_root, root);
}

// ── Analytic uniform topology builder (perf #1) ─────────────────────────

/// `build_cc_refined_uniform` must be bit-identical to the generic
/// `build_topology_from` on the same child mesh, across several base meshes and
/// refinement levels (incl. triangle/mixed-valence parents and boundaries).
fn assert_analytic_matches_generic(base: Mesh, levels: usize) {
    use crate::catmull_clark::topology::{build_cc_refined_uniform, build_topology_from};

    let mut parent = build_topology_from(&base).unwrap();
    let mut pv_count = base.vertex_count as usize;

    for level in 0..levels {
        let analytic = build_cc_refined_uniform(&parent, pv_count);
        let child_vertex_count = parent.faces.len() + parent.edge_vertices.len() + pv_count;

        // Reference child built generically from the analytic child's faces.
        let child_mesh = Mesh {
            vertex_count: child_vertex_count as u32,
            face_vertex_counts: analytic.face_vertex_counts.clone(),
            face_vertex_indices: analytic.face_vertex_indices.clone(),
            edge_vertices: vec![],
            edge_creases: vec![],
            vertex_corners: vec![0.0; child_vertex_count],
        };
        let generic = build_topology_from(&child_mesh).unwrap();
        let a = &analytic.topology;

        assert_eq!(a.faces, generic.faces, "faces @level {level}");
        assert_eq!(
            a.face_edges, generic.face_edges,
            "face_edges @level {level}"
        );
        assert_eq!(
            a.edge_vertices, generic.edge_vertices,
            "edge_vertices @level {level}"
        );
        assert_eq!(
            a.edge_faces, generic.edge_faces,
            "edge_faces @level {level}"
        );
        assert_eq!(
            a.vertex_edges, generic.vertex_edges,
            "vertex_edges @level {level}"
        );
        assert_eq!(
            a.vertex_faces, generic.vertex_faces,
            "vertex_faces @level {level}"
        );
        assert_eq!(
            a.edge_creases, generic.edge_creases,
            "edge_creases @level {level}"
        );
        assert!(
            a.edge_key_to_index == generic.edge_key_to_index,
            "edge_key_to_index @level {level}"
        );

        parent = analytic.topology;
        pv_count = child_vertex_count;
    }
}

#[test]
fn analytic_uniform_topology_matches_generic_quad() {
    assert_analytic_matches_generic(quad_topology(), 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_cube() {
    assert_analytic_matches_generic(cube_topology(), 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_triangle() {
    // Triangle parent exercises non-quad (valence-3) base faces.
    let tri = Mesh {
        vertex_count: 3,
        face_vertex_counts: vec![3],
        face_vertex_indices: vec![0, 1, 2],
        edge_vertices: vec![[0, 1], [1, 2], [0, 2]],
        edge_creases: vec![0.0; 3],
        vertex_corners: vec![0.0; 3],
    };
    assert_analytic_matches_generic(tri, 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_mixed() {
    // Triangle + quad sharing an edge: mixed valence and a boundary.
    let mixed = Mesh {
        vertex_count: 5,
        face_vertex_counts: vec![3, 4],
        face_vertex_indices: vec![0, 1, 2, 1, 3, 4, 2],
        edge_vertices: vec![[0, 1], [1, 2], [0, 2], [1, 3], [3, 4], [2, 4]],
        edge_creases: vec![0.0; 6],
        vertex_corners: vec![0.0; 5],
    };
    assert_analytic_matches_generic(mixed, 3);
}
