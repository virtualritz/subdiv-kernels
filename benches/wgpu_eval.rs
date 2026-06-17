//! Control-edit latency: CPU stencil re-evaluation vs cached-table GPU
//! re-dispatch. For a control-point edit on a static topology:
//!
//! - `cpu/stencil_interpolate` -- the composed stencil table applied on the
//!   CPU, the lower bound for any stencil-cached CPU path.
//! - `gpu/redispatch` -- upload the edited control positions, dispatch the
//!   cached `StencilTableGpu` through the cached pipeline, read back. Buffers
//!   are created per dispatch.
//!
//! The design's win condition: an order-of-magnitude drop on re-dispatch vs
//! the CPU path. Run with `cargo bench --features wgpu --bench wgpu_eval`; the
//! GPU group no-op skips when no adapter is present.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::num::NonZeroU8;
use subdiv_kernels::{
    BufferDescriptor, GpuContext, Mesh, Refiner, Scheme, SchemeOptions, StencilEvalPipeline,
    StencilTable, StencilTableGpu, UniformRefine, evaluate_stencils,
};
use wgpu::util::DeviceExt;

/// The benched cage sizes (quad-grid side) and subdivision levels: a 100x100
/// grid (~10k control points) at display levels 1 and 2, and a 316x316 one
/// (~100k, the design §7 "heavy cage") at level 1 -- the level auto-subdiv
/// realistically picks for a cage already that dense. (316 at L2 is omitted:
/// a 1.6M-quad display mesh is far past the display path's operating range
/// and the native-subdivide samples alone take minutes.)
const CASES: &[(u32, u8)] = &[(100, 1), (100, 2), (316, 1)];

/// An `n` x `n` quad grid cage as a kernel [`Mesh`] + positions.
fn grid_topology(n: u32) -> (Mesh, Vec<[f32; 3]>) {
    let stride = n + 1;
    let positions: Vec<[f32; 3]> = (0..stride)
        .flat_map(|i| (0..stride).map(move |j| [i as f32, 0.0, j as f32]))
        .collect();

    let vid = |i: u32, j: u32| i * stride + j;
    let mut face_vertex_indices = Vec::with_capacity((n * n * 4) as usize);
    let mut edge_vertices: Vec<[u32; 2]> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            face_vertex_indices.extend_from_slice(&[
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]);
        }
    }
    for i in 0..stride {
        for j in 0..stride {
            if i + 1 < stride {
                edge_vertices.push([vid(i, j), vid(i + 1, j)]);
            }
            if j + 1 < stride {
                edge_vertices.push([vid(i, j), vid(i, j + 1)]);
            }
        }
    }
    let edge_count = edge_vertices.len();
    let topo = Mesh {
        vertex_count: stride * stride,
        face_vertex_counts: vec![4; (n * n) as usize],
        face_vertex_indices,
        edge_vertices,
        edge_creases: vec![0.0; edge_count],
        vertex_corners: vec![0.0; (stride * stride) as usize],
    };
    (topo, positions)
}

/// The composed cage -> refined-level stencil table cached on the GPU.
fn composed_table(topo: Mesh, level: u8, input_count: usize) -> StencilTable {
    let refiner =
        Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine {
            levels: NonZeroU8::new(level).expect("nonzero"),
            ..Default::default()
        })
        .expect("refinement");
    result.compose_stencils(input_count)
}

fn bench_control_edit(c: &mut Criterion) {
    let gpu = GpuContext::new();
    if gpu.is_none() {
        eprintln!("gpu_eval: no GPU adapter; the gpu group is skipped");
    }

    let mut group = c.benchmark_group("control_edit");
    group.sample_size(10);

    for &(n, level) in CASES {
        let (topo, mut positions) = grid_topology(n);
        let table = composed_table(topo, level, positions.len());
        let label = format!("grid{n}x{n}_L{level}_out{}", table.output_count());

        // The stencil-cached CPU lower bound.
        group.bench_with_input(
            BenchmarkId::new("cpu_stencil_interpolate", &label),
            &(),
            |b, ()| {
                b.iter(|| {
                    positions[0][1] += 1.0e-6;
                    black_box(table.interpolate(black_box(&positions)))
                });
            },
        );

        // The §6 GPU cache hit: per-edit upload + dispatch + readback
        // through the cached pipeline + table.
        if let Some(ctx) = &gpu {
            let pipeline = StencilEvalPipeline::new(&ctx.device);
            let gpu_table = StencilTableGpu::from((&ctx.device, &table));
            let output_count = gpu_table.output_count() as usize;
            group.bench_with_input(BenchmarkId::new("gpu_redispatch", &label), &(), |b, ()| {
                b.iter(|| {
                    positions[0][1] += 1.0e-6;
                    let src = ctx
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("bench::src"),
                            contents: bytemuck::cast_slice(&positions),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        });
                    let dst_size = ((output_count * 3 * size_of::<f32>()) as u64).max(4);
                    let dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bench::dst"),
                        size: dst_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let desc = BufferDescriptor::packed(3);
                    evaluate_stencils(
                        &ctx.device,
                        &ctx.queue,
                        &pipeline,
                        &gpu_table,
                        &src,
                        &dst,
                        desc,
                        desc,
                        0..output_count as u32,
                    )
                    .expect("dispatch");
                    black_box(readback(&ctx.device, &ctx.queue, &dst, dst_size))
                });
            });
        }
    }
    group.finish();
}

/// Blocking buffer readback (stage -> copy -> map -> poll) so the benched
/// dispatch includes the full round-trip cost.
fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size_bytes: u64,
) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench::readback"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().expect("map callback").expect("map");
    bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec()
}

criterion_group!(benches, bench_control_edit);
criterion_main!(benches);
