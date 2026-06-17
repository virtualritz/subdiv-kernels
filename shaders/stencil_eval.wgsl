// Uniform stencil evaluation compute kernel.
//
// Ported from opensubdiv-petite's `shaders/wgsl/stencil_eval.wgsl` (which itself
// mirrors OpenSubdiv's `glslComputeKernel.glsl`), specialised to subdiv-kernels'
// CSR `StencilTable`: row `i`'s entries span `offsets[i]..offsets[i+1]`, so the
// per-row size is derived from the offsets and no separate `sizes` buffer is
// needed. Positions/primvars only -- the limit-derivative (b1) path is separate.

override WORKGROUP_SIZE: u32 = 64u;

struct Params {
    // All offsets/strides are in floats.
    src_offset: u32,
    dst_offset: u32,
    src_stride: u32,
    dst_stride: u32,
    length: u32,       // components per element (e.g. 3 for xyz)
    batch_start: u32,  // first output row (inclusive)
    batch_end: u32,    // last output row (exclusive)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_buffer: array<f32>;
// CSR row offsets, length = output_count + 1.
@group(0) @binding(3) var<storage, read> stencil_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> stencil_indices: array<u32>;
@group(0) @binding(5) var<storage, read> stencil_weights: array<f32>;
// Indirection: output-row indices for the sparse (indexed) entry point.
@group(0) @binding(6) var<storage, read> stencil_indirection: array<u32>;

// Cap per-element component count (matches the host MAX_COMPONENTS).
const MAX_LENGTH: u32 = 32u;

// Evaluate output row `current` into dst_buffer.
fn eval_row(current: u32) {
    let row_start = stencil_offsets[current];
    let row_end = stencil_offsets[current + 1u];
    let dst_base = params.dst_offset + current * params.dst_stride;

    for (var c: u32 = 0u; c < params.length && c < MAX_LENGTH; c = c + 1u) {
        var sum: f32 = 0.0;
        for (var si: u32 = row_start; si < row_end; si = si + 1u) {
            let vi = params.src_offset + stencil_indices[si] * params.src_stride + c;
            sum = sum + stencil_weights[si] * src_buffer[vi];
        }
        dst_buffer[dst_base + c] = sum;
    }
}

// Dense: invocation gid.x handles output row gid.x + batch_start, in
// [batch_start, batch_end).
@compute @workgroup_size(WORKGROUP_SIZE)
fn eval_stencils(@builtin(global_invocation_id) gid: vec3<u32>) {
    let current = gid.x + params.batch_start;
    if (current >= params.batch_end) {
        return;
    }
    eval_row(current);
}

// Sparse: invocation gid.x handles the output row named by the indirection
// buffer; `batch_end` is the number of indirection entries.
@compute @workgroup_size(WORKGROUP_SIZE)
fn eval_stencils_indexed(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch_end) {
        return;
    }
    eval_row(stencil_indirection[gid.x]);
}
