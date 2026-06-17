use core::num::NonZeroU8;

use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, Mesh, Refiner, Scheme, SchemeOptions,
    StencilTable, UniformRefine,
};

/// Apply a chain of face-varying stencil tables to a UV array.
fn apply_fvar(tables: &[StencilTable], uvs: &[[f32; 2]]) -> Vec<[f32; 2]> {
    tables.iter().fold(uvs.to_vec(), |d, t| t.interpolate(&d))
}

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
    let uvs = vec![[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]];
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine::from(NonZeroU8::new(1).unwrap()),
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();
    let out = apply_fvar(&tables, &uvs);
    assert_eq!(out.len(), 9); // 3 child triangles * 3 corners
    // Single triangle: no interior edges, so no flip. Each child triangle
    // is [centroid, vp_i, vp_{i+1}]; centroid = (c0+c1+c2)/3 = (1,1).
    approx(out[0], [1.0, 1.0]);
    approx(out[1], [0.0, 0.0]);
    approx(out[2], [3.0, 0.0]);
    approx(out[3], [1.0, 1.0]);
    approx(out[4], [3.0, 0.0]);
    approx(out[5], [0.0, 3.0]);
}

#[test]
fn fvar_linear_preserves_seam() {
    // Two triangles sharing edge (1,2); the shared edge is an interior
    // non-crease edge so √3 flips it, but Linear values stay face-local
    // (centroids average only their own face; vertex points copy the home
    // face), so no refined value crosses from x∈[0,1] into x∈[10,11].
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
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
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

/// Single triangle.
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

/// Tetrahedron (4 triangles, 4 vertices).
fn tetrahedron_topology() -> Mesh {
    Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![3, 3, 3, 3],
        face_vertex_indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        edge_vertices: vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        edge_creases: vec![0.0; 6],
        vertex_corners: vec![0.0; 4],
    }
}

fn tetrahedron_positions() -> Vec<[f32; 3]> {
    vec![
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
}

#[test]
fn single_triangle_topology() {
    let topo = triangle_topology();
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();

    // SAFETY: 1 is non-zero.
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();

    // 1 triangle → 1 centroid + 3 vertex points = 4 output vertices
    assert_eq!(result.topology.vertex_count, 4);
    // 3 child triangles (boundary triangle — no edge flips)
    assert_eq!(result.topology.face_vertex_counts.len(), 3);
    assert!(result.topology.face_vertex_counts.iter().all(|&n| n == 3));
}

#[test]
fn tetrahedron_level1() {
    let topo = tetrahedron_topology();
    let pos = tetrahedron_positions();
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();

    // SAFETY: 1 is non-zero.
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();

    // 4 faces → 4 centroids, 4 vertices → 4 vertex points = 8 total
    assert_eq!(result.topology.vertex_count, 8);

    let refined_pos = result.interpolate(&pos);
    assert_eq!(refined_pos.len(), 8);

    // First 4 are face centroids — average of face corners.
    // Face 0 = [0,1,2] → centroid = avg of tetra positions = known value.
    // Just verify they're not NaN.
    refined_pos.iter().for_each(|p| {
        assert!(p.iter().all(|c| c.is_finite()));
    });
}

#[test]
fn tetrahedron_level2() {
    let topo = tetrahedron_topology();
    let pos = tetrahedron_positions();
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();

    // SAFETY: 2 is non-zero.
    let req = UniformRefine::from(NonZeroU8::new(2).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();

    let refined_pos = result.interpolate(&pos);
    assert_eq!(refined_pos.len(), result.topology.vertex_count as usize);

    // All positions should be finite.
    refined_pos.iter().for_each(|p| {
        assert!(p.iter().all(|c| c.is_finite()));
    });
}

#[test]
fn interpolate_f64_positions() {
    let topo = tetrahedron_topology();
    let pos: Vec<[f64; 3]> = vec![
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];

    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
    // SAFETY: 1 is non-zero.
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let result = refiner.refine_uniform(&req).unwrap();
    let refined = result.interpolate(&pos);

    assert_eq!(refined.len(), result.topology.vertex_count as usize);
    // All centroid positions should be finite and within the bounding box.
    refined.iter().for_each(|p| {
        assert!(p.iter().all(|c| c.is_finite() && c.abs() < 2.0));
    });
}

#[test]
fn fvar_boundaries_no_seam_matches_positional() {
    // No seams ⇒ SmoothWithLinearBoundaries equals the positional smooth
    // refinement (level 1). Tetrahedron: all vertices interior.
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
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
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
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
        ],
        edge_creases: vec![0.0; 12],
        vertex_corners: vec![0.0; 6],
    }
}

fn octahedron_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
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
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
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

fn assert_smooth_crease_matches_creased_positional(mode: FaceVaryingInterpolation) {
    let topo = octahedron_topology();
    let positions = octahedron_positions();
    let seam = [[1u32, 2], [2, 3], [3, 4], [4, 1]];
    let mut creased = topo.clone();
    creased.edge_creases = crate::test_support::creases_for(&topo, &seam);
    let refiner = Refiner::new(creased, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
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
    // Two triangles, all six fvar indices distinct ⇒ shared edge + boundaries
    // are seams. The √3 edge flip rearranges connectivity, but every smooth
    // stencil stays home-side, so no refined value lands in the (1,10) gap.
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
    let refiner = Refiner::new(topo, Scheme::Sqrt3, SchemeOptions::default()).unwrap();
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
