use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, Mesh, Refiner, Scheme, SchemeOptions,
    StencilTable, UniformRefine,
};
use core::num::NonZeroU8;

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
fn fvar_linear_single_quad_level1() {
    // A lone quad has only its F-face (no interior edges/vertices), so each
    // refined corner copies its source corner's value.
    let topo = single_quad();
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3],
        value_count: 4,
    };
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let tables = refiner
        .face_varying_stencils(
            &UniformRefine::from(NonZeroU8::new(1).unwrap()),
            &fvar,
            FaceVaryingInterpolation::Linear,
        )
        .unwrap();
    let out = apply_fvar(&tables, &uvs);
    assert_eq!(out.len(), 4);
    for (o, u) in out.iter().zip(&uvs) {
        approx(*o, *u);
    }
}

#[test]
fn fvar_linear_preserves_seam() {
    // Two quads sharing edge (1,2). The interior edge spawns an E-face whose
    // corners copy from both faces, but each corner is a pure copy, so no
    // refined value crosses from x∈[0,1] into x∈[10,11].
    let topo = Mesh {
        vertex_count: 6,
        face_vertex_counts: vec![4, 4],
        face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 5, 2],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5], [5, 2]],
        edge_creases: vec![0.0; 7],
        vertex_corners: vec![0.0; 6],
    };
    let fvar = FaceVaryingChannel {
        indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
        value_count: 8,
    };
    let uvs = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [11.0, 1.0],
        [10.0, 1.0],
    ];
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
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

fn single_quad() -> Mesh {
    Mesh {
        vertex_count: 4,
        face_vertex_counts: vec![4],
        face_vertex_indices: vec![0, 1, 2, 3],
        edge_vertices: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        edge_creases: vec![0.0; 4],
        vertex_corners: vec![0.0; 4],
    }
}

fn cube() -> Mesh {
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

#[test]
fn single_quad_topology() {
    let topo = single_quad();
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(1).unwrap(),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).unwrap();

    // Single quad: 4 corners → 4 face-vertex points.
    assert_eq!(result.topology.vertex_count, 4);

    // All boundary edges → no E-faces, no V-faces. Only 1 F-face.
    let f_count = result.topology.face_vertex_counts.len();
    assert_eq!(f_count, 1);
}

#[test]
fn cube_topology_level1() {
    let topo = cube();
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(1).unwrap(),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).unwrap();

    // 6 faces × 4 corners = 24 face-vertex points.
    assert_eq!(result.topology.vertex_count, 24);

    // F-faces: 6, E-faces: 12 (one per interior edge), V-faces: 8.
    let f_count = result.topology.face_vertex_counts.len();
    assert_eq!(f_count, 6 + 12 + 8);
}

#[test]
fn weight_sum_is_one() {
    let topo = cube();
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(1).unwrap(),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).unwrap();

    // Every stencil's weights should sum to 1.0.
    let table = &result.level_stencils[0];
    for i in 0..table.output_count() {
        let start = table.offsets[i] as usize;
        let end = table.offsets[i + 1] as usize;
        let sum: f32 = table.weights[start..end].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "stencil {i} weight sum = {sum}",);
    }
}

#[test]
fn multi_level_no_nan() {
    let topo = cube();
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(2).unwrap(),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).unwrap();

    // Interpolate with unit-cube positions.
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
    let refined = result.interpolate(&positions);

    assert_eq!(refined.len(), result.topology.vertex_count as usize);
    assert!(
        refined.iter().all(|p| p.iter().all(|v| v.is_finite())),
        "NaN or Inf in refined positions",
    );
}

#[test]
fn interpolate_f64_positions() {
    let topo = cube();
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine {
        levels: NonZeroU8::new(1).unwrap(),
        ..Default::default()
    };
    let result = refiner.refine_uniform(&req).unwrap();

    let positions: Vec<[f64; 3]> = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let refined = result.interpolate(&positions);
    assert_eq!(refined.len(), result.topology.vertex_count as usize);
    assert!(refined.iter().all(|p| p.iter().all(|v| v.is_finite())));
}

#[test]
fn fvar_boundaries_no_seam_matches_positional() {
    // No seams ⇒ SmoothWithLinearBoundaries equals Doo-Sabin's positional
    // refinement (level 1). Cube: all vertices interior (closed mesh).
    let topo = cube();
    let positions: Vec<[f32; 3]> = (0..topo.vertex_count)
        .map(|i| {
            let i = i as f32;
            [i, 2.0 * i, 3.0 * i + 1.0]
        })
        .collect();
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
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
            (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4 && (a[2] - b[2]).abs() < 1e-4,
            "corner {c} (vertex {v}): fvar {a:?} != positional {b:?}",
        );
    }
}

#[test]
fn fvar_smooth_modes_coincide() {
    // Doo-Sabin's face-local smooth rule makes all three smooth modes equal.
    let topo = cube();
    let positions: Vec<[f32; 3]> = (0..topo.vertex_count)
        .map(|i| {
            let i = i as f32;
            [i, 2.0 * i, 3.0 * i + 1.0]
        })
        .collect();
    let channel = FaceVaryingChannel {
        indices: topo.face_vertex_indices.clone(),
        value_count: topo.vertex_count,
    };
    let refiner = Refiner::new(topo, Scheme::DooSabin, SchemeOptions::default()).unwrap();
    let req = UniformRefine::from(NonZeroU8::new(1).unwrap());
    let apply = |mode| {
        let tables = refiner.face_varying_stencils(&req, &channel, mode).unwrap();
        tables
            .iter()
            .fold(positions.clone(), |d, t| t.interpolate(&d))
    };
    let b = apply(FaceVaryingInterpolation::SmoothWithLinearBoundaries);
    let c = apply(FaceVaryingInterpolation::SmoothWithLinearCorners);
    let s = apply(FaceVaryingInterpolation::Smooth);
    assert_eq!(b.len(), c.len());
    for ((vb, vc), vs) in b.iter().zip(&c).zip(&s) {
        assert_eq!(vb, vc);
        assert_eq!(vb, vs);
    }
}
