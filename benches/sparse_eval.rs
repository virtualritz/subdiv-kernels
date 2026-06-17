//! Local-edit latency: dense re-dispatch vs sparse re-evaluation (the §7
//! benchmark of the sparse subdivision design). For a single moved control
//! point, "dense" recomputes every refined output via the composed stencil
//! table; "sparse" recomputes only the rows the inverse stencil map flags as
//! affected. The crossover -- where sparse wins -- widens as cage size and
//! subdivision level grow. Each `BenchmarkId` label carries the dense output
//! count (`out`) and affected-row count (`aff`) so the ratio is visible.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::num::NonZeroU8;
use subdiv_kernels::{
    AffectedScratch, InverseStencilChain, Refiner, Scheme, SchemeOptions, StencilTable,
    Mesh, UniformRefine,
};

/// An `n` x `n` quad grid cage: `(n+1)^2` vertices, `n^2` faces, all its edges
/// (uncreased). An open mesh, so boundary stencils participate.
fn grid(n: u32) -> (Mesh, Vec<[f32; 3]>) {
    let stride = n + 1;
    let vid = |i: u32, j: u32| i * stride + j;

    let mut positions = Vec::with_capacity((stride * stride) as usize);
    for i in 0..stride {
        for j in 0..stride {
            positions.push([i as f32, j as f32, 0.0]);
        }
    }

    let mut face_vertex_indices = Vec::with_capacity((n * n * 4) as usize);
    for i in 0..n {
        for j in 0..n {
            face_vertex_indices.extend_from_slice(&[
                vid(i, j),
                vid(i, j + 1),
                vid(i + 1, j + 1),
                vid(i + 1, j),
            ]);
        }
    }

    let mut edge_vertices = Vec::new();
    for i in 0..stride {
        for j in 0..n {
            edge_vertices.push([vid(i, j), vid(i, j + 1)]); // horizontal
        }
    }
    for i in 0..n {
        for j in 0..stride {
            edge_vertices.push([vid(i, j), vid(i + 1, j)]); // vertical
        }
    }

    let topology = Mesh {
        vertex_count: stride * stride,
        face_vertex_counts: vec![4; (n * n) as usize],
        face_vertex_indices,
        edge_creases: vec![0.0; edge_vertices.len()],
        edge_vertices,
        vertex_corners: vec![0.0; (stride * stride) as usize],
    };
    (topology, positions)
}

struct Prepared {
    composed: StencilTable,
    chain: InverseStencilChain,
    affected: Vec<u32>,
    base_out: Vec<[f32; 3]>,
    edited: Vec<[f32; 3]>,
    moved: u32,
}

fn prepare(n: u32, level: u8) -> Prepared {
    let (topology, base) = grid(n);
    let refiner =
        Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default()).expect("valid grid");
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(level).unwrap(),
            ..Default::default()
        })
        .expect("refinement");

    let composed = result.compose_stencils(base.len());
    let base_out = composed.interpolate(&base);

    let moved = (base.len() / 2) as u32; // a central-ish control vertex
    let mut edited = base.clone();
    edited[moved as usize][2] += 0.5;

    let chain = result.inverse_stencil_chain();
    let affected = chain.affected_outputs(&[moved]);

    Prepared {
        composed,
        chain,
        affected,
        base_out,
        edited,
        moved,
    }
}

fn local_edit(c: &mut Criterion) {
    // (cage grid size n, subdivision level)
    let configs = [(8u32, 2u8), (16, 2), (32, 2), (8, 3)];
    let mut group = c.benchmark_group("local_edit");

    for &(n, level) in &configs {
        let p = prepare(n, level);
        let label = format!(
            "grid{n}_lvl{level}_out{}_aff{}",
            p.base_out.len(),
            p.affected.len()
        );

        // Dense re-dispatch: recompute every refined output.
        group.bench_with_input(BenchmarkId::new("dense", &label), &p, |b, p| {
            b.iter(|| black_box(p.composed.interpolate(black_box(&p.edited))));
        });

        // Sparse: recompute only the affected rows into the persistent buffer
        // (affected set precomputed -- the steady-state drag of one vertex).
        group.bench_with_input(BenchmarkId::new("sparse", &label), &p, |b, p| {
            let mut buf = p.base_out.clone();
            b.iter(|| {
                p.composed
                    .interpolate_rows(black_box(&p.edited), black_box(&p.affected), &mut buf);
                black_box(&buf);
            });
        });

        // Sparse including the affected-set gather (the moved vertex varies each
        // edit, so the inverse map is re-queried per edit).
        group.bench_with_input(
            BenchmarkId::new("sparse_with_gather", &label),
            &p,
            |b, p| {
                let mut buf = p.base_out.clone();
                b.iter(|| {
                    let affected = p.chain.affected_outputs(black_box(&[p.moved]));
                    p.composed
                        .interpolate_rows(black_box(&p.edited), &affected, &mut buf);
                    black_box(&buf);
                });
            },
        );

        // Same, but with a reused AffectedScratch -- zero per-query allocation.
        group.bench_with_input(
            BenchmarkId::new("sparse_with_gather_scratch", &label),
            &p,
            |b, p| {
                let mut buf = p.base_out.clone();
                let mut scratch = AffectedScratch::default();
                let mut affected = Vec::new();
                b.iter(|| {
                    p.chain.affected_outputs_into(
                        black_box(&[p.moved]),
                        &mut scratch,
                        &mut affected,
                    );
                    p.composed
                        .interpolate_rows(black_box(&p.edited), &affected, &mut buf);
                    black_box(&buf);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, local_edit);
criterion_main!(benches);
