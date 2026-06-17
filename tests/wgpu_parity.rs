#![cfg(feature = "wgpu")]
//! P0 GPU parity gate: the GPU stencil-eval compute path matches the CPU
//! `StencilTable::interpolate` bit-close on a refined cube. Skips gracefully
//! when no GPU adapter is available.

use std::num::NonZeroU8;
use subdiv_kernels::{GpuContext, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine};

/// A closed unit cube: 8 vertices, 6 quad faces, 12 edges, no creases.
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

#[test]
fn gpu_eval_matches_cpu_interpolate() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU adapter available; skipping gpu_eval_matches_cpu_interpolate");
        return;
    };

    for levels in 1..=2u8 {
        let (topology, base) = cube();
        let refiner = Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default())
            .expect("cube topology is valid");
        let result = refiner
            .refine_uniform(&UniformRefine {
                levels: NonZeroU8::new(levels).unwrap(),
                ..Default::default()
            })
            .expect("refinement succeeds");
        let composed = result.compose_stencils(base.len());

        // CPU baseline.
        let cpu: Vec<[f32; 3]> = composed.interpolate(&base);
        let cpu_flat: Vec<f32> = cpu.iter().flat_map(|p| *p).collect();

        // GPU.
        let input_flat: Vec<f32> = base.iter().flat_map(|p| *p).collect();
        let gpu_flat = ctx
            .evaluate(&composed, &input_flat, 3)
            .expect("gpu evaluate succeeds");

        assert_eq!(
            gpu_flat.len(),
            cpu_flat.len(),
            "level {levels} output length"
        );
        for (i, (c, g)) in cpu_flat.iter().zip(&gpu_flat).enumerate() {
            assert!(
                (c - g).abs() < 1e-5,
                "level {levels} component {i}: cpu {c} vs gpu {g}"
            );
        }
    }
}

#[test]
fn gpu_sparse_matches_dense() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU adapter available; skipping gpu_sparse_matches_dense");
        return;
    };

    let (topology, base) = cube();
    let refiner = Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default())
        .expect("cube topology is valid");
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(2).unwrap(),
            ..Default::default()
        })
        .expect("refinement succeeds");
    let composed = result.compose_stencils(base.len());

    let base_flat: Vec<f32> = base.iter().flat_map(|p| *p).collect();
    let mut edited = base.clone();
    edited[0][0] += 1.0;
    let edited_flat: Vec<f32> = edited.iter().flat_map(|p| *p).collect();

    // Dense GPU evaluations of the old and new cages.
    let dense_base = ctx
        .evaluate(&composed, &base_flat, 3)
        .expect("gpu dense base");
    let dense_full = ctx
        .evaluate(&composed, &edited_flat, 3)
        .expect("gpu dense full");

    // Sparse GPU update: recompute only the affected rows into the prior output;
    // it must reproduce the full dense re-evaluation.
    let affected = result.affected_outputs(&[0]);
    let sparse = ctx
        .evaluate_sparse(&composed, &edited_flat, 3, &affected, &dense_base)
        .expect("gpu sparse");

    assert_eq!(sparse.len(), dense_full.len());
    for (i, (s, d)) in sparse.iter().zip(&dense_full).enumerate() {
        assert!(
            (s - d).abs() < 1e-5,
            "component {i}: sparse {s} vs dense {d}"
        );
    }
}
