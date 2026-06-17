use core::num::NonZeroU8;

use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, Mesh, Refiner, Scheme, SchemeOptions,
    StencilTable, UniformRefine, VertexOrigin,
};

/// Apply a chain of face-varying stencil tables to a UV array.
fn apply_fvar(tables: &[StencilTable], uvs: &[[f32; 2]]) -> Vec<[f32; 2]> {
    tables.iter().fold(uvs.to_vec(), |d, t| t.interpolate(&d))
}

/// Assert two UVs agree componentwise.
fn approx(a: [f32; 2], b: [f32; 2]) {
    assert!(
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5,
        "{a:?} != {b:?}",
    );
}

#[test]
fn fvar_linear_triangle_level1() {
    let topo = triangle_topology();
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2],
        value_count: 3,
    };
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine::from(NonZeroU8::new(1).unwrap()),
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();
    let out = apply_fvar(&tables, &uvs);
    assert_eq!(out.len(), 12); // 4 child triangles * 3 corners
    // Child triangle 0 = [v0, mid(0,1), mid(2,0)].
    approx(out[0], [0.0, 0.0]);
    approx(out[1], [0.5, 0.0]);
    approx(out[2], [0.0, 0.5]);
    // Child triangle 3 (center) = [mid(0,1), mid(1,2), mid(2,0)].
    approx(out[9], [0.5, 0.0]);
    approx(out[10], [0.5, 0.5]);
    approx(out[11], [0.0, 0.5]);
}

#[test]
fn fvar_linear_preserves_seam() {
    // Two triangles sharing edge (1,2); all face-varying indices distinct,
    // so the shared edge is a seam. Face 0's values live in x∈[0,1], face
    // 1's in x∈[10,11]; Linear is face-local, so no refined value may land
    // in the (1,10) gap.
    let topo = Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![3, 3],
        face_vertex_indices: vec![0, 1, 2, 1, 3, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]],
        edge_creases: vec![0.0; 5],
        vertex_corners: vec![0.0; 4],
    };
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3, 4, 5],
        value_count: 6,
    };
    let uvs = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [10.0, 1.0],
    ];
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine::from(NonZeroU8::new(1).unwrap()),
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();
    let out = apply_fvar(&tables, &uvs);
    for uv in &out {
        assert!(
            uv[0] <= 1.0001 || uv[0] >= 9.9999,
            "cross-seam mixing produced {uv:?}",
        );
    }
}

fn triangle_topology() -> Mesh {
    Mesh {
        vertex_count: 3,
        face_vertex_counts: vec![3],
        face_vertex_indices: vec![0, 1, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 0]],
        edge_creases: vec![0.0; 3],
        vertex_corners: vec![0.0; 3],
    }
}

fn triangle_positions() -> Vec<[f32; 3]> {
    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}

#[test]
fn triangle_refines_to_four_triangles() {
    let topo = triangle_topology();
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(1).unwrap(),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.topology.vertex_count, 6);
    assert_eq!(result.topology.face_vertex_counts, vec![3, 3, 3, 3]);

    let lineage = &result.lineage;
    assert_eq!(lineage.vertex_origin.len(), 6);
    assert_eq!(lineage.face_parent.len(), 4);
    assert!(
        lineage
            .vertex_origin
            .iter()
            .take(3)
            .all(|origin| matches!(origin, VertexOrigin::Edge(_)))
    );
    assert!(
        lineage
            .vertex_origin
            .iter()
            .skip(3)
            .all(|origin| matches!(origin, VertexOrigin::Vertex(_)))
    );
}

#[test]
fn level2_vertex_count() {
    let topo = triangle_topology();
    let pos = triangle_positions();
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(2).unwrap(),
            ..Default::default()
        })
        .unwrap();

    let out = result.interpolate(&pos);
    assert_eq!(out.len(), result.topology.vertex_count as usize);
}

#[test]
fn edge_polylines_split_at_each_level() {
    // Loop inserts a midpoint on every edge; each parent-edge polyline
    // grows from length 2 at the base to (2^N + 1) at level N.
    let topo = triangle_topology();
    let base_edge_count = topo.edge_vertices.len();

    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(2).unwrap(),
            edge_polylines: true,
            ..Default::default()
        })
        .unwrap();

    let polylines = result
        .edge_polylines
        .expect("edge_polylines should be Some when requested");

    assert_eq!(polylines.len(), base_edge_count);

    let vertex_count = result.topology.vertex_count;
    for (ei, poly) in polylines.iter().enumerate() {
        assert_eq!(
            poly.len(),
            5,
            "parent edge {ei} should have 2^2 + 1 = 5 points at level 2, got {}",
            poly.len()
        );
        for &vi in poly {
            assert!(
                vi < vertex_count,
                "polyline vertex index {vi} out of range (vertex_count={vertex_count})"
            );
        }
    }
}

#[test]
fn crease_propagated() {
    let mut topo = triangle_topology();
    topo.edge_creases = vec![2.0, 0.0, 0.0];
    topo.vertex_corners = vec![2.0, 0.0, 0.0];

    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(1).unwrap(),
            ..Default::default()
        })
        .unwrap();

    // Corner vertex (index 3 in Loop layout: [edge_pts(3), vertex_pts(3)])
    assert!((result.topology.vertex_corners[3] - 2.0).abs() < 1e-6);

    let creased_edges = result
        .topology
        .edge_creases
        .iter()
        .filter(|&&c| c > 0.0)
        .count();
    assert!(creased_edges >= 2);
}

/// Closed tetrahedron: every vertex is an interior valence-3 vertex, so the
/// smooth face-varying path is exercised.
fn tetrahedron_topology() -> Mesh {
    Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![3; 4],
        face_vertex_indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        edge_vertices: vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        edge_creases: vec![0.0; 6],
        vertex_corners: vec![0.0; 4],
    }
}

#[test]
fn fvar_boundaries_no_seam_matches_positional() {
    // With no seams (fvar index == vertex index), SmoothWithLinearBoundaries
    // must reproduce the positional smooth refinement at level 1.
    let topo = tetrahedron_topology();
    let positions: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();
    let refined_pos = result.interpolate(&positions);
    let tables = refiner
        .face_varying_stencils(
            &req,
            &channel,
            FaceVaryingInterpolation::SmoothWithLinearBoundaries,
        )
        .unwrap();
    let fvar_out = tables
        .iter()
        .fold(positions.clone(), |d, t| t.interpolate(&d));
    for (c, &v) in result.topology.face_vertex_indices.iter().enumerate() {
        let a = fvar_out[c];
        let b = refined_pos[v as usize];
        assert!(
            (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5,
            "corner {c} (vertex {v}): fvar {a:?} != positional {b:?}",
        );
    }
}

// ── Smooth / SmoothWithLinearCorners coverage ───────────────────────────

/// Closed octahedron: top/bottom poles (valence 4, no seam) plus four
/// equator vertices, each with exactly two equator edges — the regular
/// 2-seam crease-curve case.
fn octahedron_topology() -> Mesh {
    Mesh {
        vertex_count: 6,
        face_vertex_counts: vec![3; 8],
        face_vertex_indices: vec![
            0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, // top cap
            5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4, // bottom cap
        ],
        edge_vertices: vec![
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4], // pole→equator (top)
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1], // equator loop
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4], // pole→equator (bottom)
        ],
        edge_creases: vec![0.0; 12],
        vertex_corners: vec![0.0; 6],
    }
}

fn octahedron_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 1.0],  // 0 top
        [1.0, 0.0, 0.0],  // 1
        [0.0, 1.0, 0.0],  // 2
        [-1.0, 0.0, 0.0], // 3
        [0.0, -1.0, 0.0], // 4
        [0.0, 0.0, -1.0], // 5 bottom
    ]
}

fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
}

fn assert_no_seam_matches_positional(mode: FaceVaryingInterpolation) {
    let topo = tetrahedron_topology();
    let positions: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
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

/// Seam-along-a-crease oracle: a position-valued channel seamed along the
/// equator, refined under a smooth mode, must equal the positional refinement
/// with the equator marked as infinite creases.
fn assert_smooth_crease_matches_creased_positional(mode: FaceVaryingInterpolation) {
    let topo = octahedron_topology();
    let positions = octahedron_positions();
    let seam = [[1u32, 2], [2, 3], [3, 4], [4, 1]];
    let mut creased = topo.clone();
    creased.edge_creases = crate::test_support::creases_for(&topo, &seam);
    let refiner = Refiner::new(creased, Scheme::Loop, SchemeOptions::default()).unwrap();
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
fn fvar_smooth_no_seam_matches_positional() {
    assert_no_seam_matches_positional(FaceVaryingInterpolation::Smooth);
}

#[test]
fn fvar_corners_only_no_seam_matches_positional() {
    assert_no_seam_matches_positional(FaceVaryingInterpolation::SmoothWithLinearCorners);
}

#[test]
fn fvar_smooth_crease_matches_creased_positional() {
    assert_smooth_crease_matches_creased_positional(FaceVaryingInterpolation::Smooth);
}

#[test]
fn fvar_corners_only_crease_matches_creased_positional() {
    assert_smooth_crease_matches_creased_positional(
        FaceVaryingInterpolation::SmoothWithLinearCorners,
    );
}

#[test]
fn fvar_smooth_preserves_seam() {
    // Two triangles, all six fvar indices distinct, so the shared edge and
    // both outer boundaries are seams. Face 0's values live in x∈[0,1], face
    // 1's in x∈[10,11]; the smooth crease-curve rule must stay home-side, so
    // no refined value may land in the (1,10) gap.
    let topo = Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![3, 3],
        face_vertex_indices: vec![0, 1, 2, 1, 3, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]],
        edge_creases: vec![0.0; 5],
        vertex_corners: vec![0.0; 4],
    };
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3, 4, 5],
        value_count: 6,
    };
    let uvs = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [10.0, 1.0],
    ];
    let refiner = Refiner::new(topo, Scheme::Loop, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine::from(NonZeroU8::new(1).unwrap()),
            &fvar,
            FaceVaryingInterpolation::Smooth,
        )
        .unwrap();
    let out = apply_fvar(&tables, &uvs);
    for uv in &out {
        assert!(
            uv[0] <= 1.0001 || uv[0] >= 9.9999,
            "cross-seam mixing produced {uv:?}"
        );
    }
}

// ── Analytic uniform topology builder (perf) ────────────────────────────

/// `build_loop_refined_uniform` must be bit-identical to the generic
/// `build_topology_from` on the same child mesh, across several triangle base
/// meshes and refinement levels (incl. a boundary).
fn assert_loop_analytic_matches_generic(base: Mesh, levels: usize) {
    use crate::loop_subdivision::topology::{build_loop_refined_uniform, build_topology_from};

    let mut parent = build_topology_from(&base).unwrap();
    let mut pv_count = base.vertex_count as usize;

    for level in 0..levels {
        let analytic = build_loop_refined_uniform(&parent, pv_count);
        let child_vertex_count = parent.edge_vertices.len() + pv_count;

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
            a.edge_is_boundary, generic.edge_is_boundary,
            "edge_is_boundary @level {level}"
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
            a.vertex_is_boundary, generic.vertex_is_boundary,
            "vertex_is_boundary @level {level}"
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
fn analytic_uniform_topology_matches_generic_triangle() {
    assert_loop_analytic_matches_generic(triangle_topology(), 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_tetrahedron() {
    assert_loop_analytic_matches_generic(tetrahedron_topology(), 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_octahedron() {
    assert_loop_analytic_matches_generic(octahedron_topology(), 3);
}

#[test]
fn analytic_uniform_topology_matches_generic_boundary() {
    // Two triangles sharing edge (1,2): open mesh with a boundary.
    let strip = Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![3, 3],
        face_vertex_indices: vec![0, 1, 2, 1, 3, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]],
        edge_creases: vec![0.0; 5],
        vertex_corners: vec![0.0; 4],
    };
    assert_loop_analytic_matches_generic(strip, 3);
}
