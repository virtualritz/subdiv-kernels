//! Closest-point query latency on the limit surface (the design's
//! section-7 benchmark for s3; per-query budget context: the
//! amendment drag wants ~10^2-10^3 queries per interactive frame and
//! the SDF sampler batches far more).
//!
//! Cases:
//!
//! - `grid100_L1` -- the heavy cage: a 100x100 quad grid (~10k control
//!   points) refined one level to 40k quads, regular-patch-dominated
//!   with an open boundary ring.
//! - `creased_cube_L2` -- feature-heavy: the infinitely-creased cube
//!   at level 2 (96 quads, every EV/crease path hot).
//!
//! Batches of 10^2..10^4 deterministic queries (near-surface, inside,
//! and far-field mixed via a fixed LCG); the BVH and the feature
//! isolation caches are warmed by one pass before timing, matching the
//! many-query consumers the index exists for. Criterion reports
//! element throughput, i.e. queries/second.
//!
//! Measured 2026-06-10 (release, Linux, Xeon E-2276M 2.8 GHz):
//! `grid100_L1` ~22-28 us/query across batch sizes (regular-patch
//! Gauss-Newton dominates; boundary-foot queries cost the feature-line
//! slide on top), `creased_cube_L2` ~250-280 us/query (feature-quad
//! isolation evals; the box-prune-defeating inside/medial queries of
//! the mix run ~1.3 ms, far-field ~0.13 ms). Against the ~10^2-10^3
//! queries/frame interactive budget: the heavy regular-dominated cage
//! fits at both ends (2.2 ms per 10^2, 28 ms per 10^3); feature-dense
//! cages fit at 10^2. The known next lever is memoizing the s2 corner
//! snaps/isolation descents that feature evals re-derive per sample.
//!
//! Re-measured 2026-06-11 after the corner-snap memoization (the s4
//! perf lever: `corner_limit_sector` results cached per
//! `(face, corner)` on the evaluator and on each isolation node):
//! `creased_cube_L2` dropped 41-43% to ~170-190 us/query
//! (same-session before: ~320-330 us); `grid100_L1` dropped 28-39%
//! against its 2026-06-10 numbers to ~13-20 us/query -- its
//! open-boundary feet snap boundary corners through the feature-line
//! slide, so the memoization reaches it too.
//!
//! Run with `cargo bench --bench closest_point`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::num::NonZeroU8;
use subdiv_kernels::{
    RefinementResult, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine,
};

/// An `n` x `n` quad grid cage (the `benches/gpu_eval.rs` layout).
fn grid_topology(n: u32) -> (Mesh, Vec<[f32; 3]>) {
    let stride = n + 1;
    let positions: Vec<[f32; 3]> = (0..stride)
        .flat_map(|i| (0..stride).map(move |j| [i as f32, 0.0, j as f32]))
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

/// The infinitely-creased cube cage (the test suites' layout).
fn creased_cube() -> (Mesh, Vec<[f32; 3]>) {
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
    let topo = Mesh {
        vertex_count: 8,
        face_vertex_counts: vec![4; 6],
        face_vertex_indices: vec![
            0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7,
        ],
        edge_vertices: vec![[4, 5], [5, 6], [6, 7], [7, 4]],
        edge_creases: vec![f32::INFINITY; 4],
        vertex_corners: vec![0.0; 8],
    };
    (topo, positions)
}

fn refine(topo: Mesh, level: u8) -> RefinementResult {
    let refiner =
        Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
    refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement")
}

/// Deterministic query batch in `bounds` (fixed-seed LCG; benches are
/// free to use one, unlike the test convention).
fn queries(count: usize, bounds: [[f32; 2]; 3]) -> Vec<[f32; 3]> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..count)
        .map(|_| {
            [
                bounds[0][0] + next() * (bounds[0][1] - bounds[0][0]),
                bounds[1][0] + next() * (bounds[1][1] - bounds[1][0]),
                bounds[2][0] + next() * (bounds[2][1] - bounds[2][0]),
            ]
        })
        .collect()
}

/// One bench case: label, refined cage, control positions, query box.
type BenchCase = (&'static str, RefinementResult, Vec<[f32; 3]>, [[f32; 2]; 3]);

fn bench_closest_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("closest_point");
    group.sample_size(10);

    let cases: Vec<BenchCase> = {
        let (grid_topo, grid_positions) = grid_topology(100);
        let (cube_topo, cube_positions) = creased_cube();
        vec![
            (
                "grid100_L1",
                refine(grid_topo, 1),
                grid_positions,
                [[-5.0, 105.0], [-8.0, 8.0], [-5.0, 105.0]],
            ),
            (
                "creased_cube_L2",
                refine(cube_topo, 2),
                cube_positions,
                [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]],
            ),
        ]
    };

    for (label, result, control_positions, bounds) in &cases {
        let refined = result.interpolate(control_positions);
        let evaluator = result.limit_evaluator(&refined).expect("limit evaluator");
        let batch = queries(10_000, *bounds);
        // Warm the BVH and the feature-isolation caches once, like the
        // many-query consumers do.
        for q in &batch {
            evaluator.closest_point(*q).expect("warmup query");
        }
        for &n in &[100usize, 1_000, 10_000] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(*label, n), &batch[..n], |b, queries| {
                b.iter(|| {
                    for &q in queries {
                        black_box(evaluator.closest_point(black_box(q)).expect("query"));
                    }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_closest_point);
criterion_main!(benches);
