//! GPU (wgpu) compute path for uniform stencil evaluation.
//!
//! Ported from opensubdiv-petite's `osd::wgpu` (which mirrors OpenSubdiv's
//! `glslComputeKernel.glsl`), specialised to this crate's CSR [`StencilTable`]
//! and to positions/primvar evaluation. The limit-derivative (b1) path -- du/dv
//! and second derivatives -- is a separate future addition.
//!
//! This module is behind the `gpu` feature. The output of
//! [`GpuContext::evaluate`] is bit-close to [`StencilTable::interpolate`] for
//! `[f32; components]` data (same ops, run on the GPU).

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable, bytes_of};
use wgpu::util::DeviceExt;

use crate::{KernelError, StencilTable};

/// Canonical WGSL source for the stencil-eval compute kernel.
pub const STENCIL_EVAL_WGSL: &str = include_str!("../shaders/stencil_eval.wgsl");

/// Maximum primvar components per element the kernel supports (matches the
/// shader's `MAX_LENGTH`).
pub const MAX_COMPONENTS: u32 = 32;

const DEFAULT_WORKGROUP_SIZE: u32 = 64;

/// A headless wgpu device + queue for running the compute kernel.
///
/// Callers that already own a `wgpu::Device` should build [`StencilEvalPipeline`]
/// and [`StencilTableGpu`] directly; this is a convenience for standalone use
/// and tests. [`new`](Self::new) returns `None` when no adapter is available.
#[derive(Debug)]
pub struct GpuContext {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu queue.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a headless compute context, or `None` if no GPU adapter is
    /// available (so callers can gracefully fall back to the CPU path).
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Async variant of [`new`](Self::new).
    pub async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        // The kernel binds 5 storage buffers; downlevel defaults allow only 4.
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 8,
            ..wgpu::Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("subdiv-kernels::wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .ok()?;

        Some(Self { device, queue })
    }

    /// Evaluate `table` over a tightly-packed input buffer of
    /// `components`-vectors, returning the packed output.
    ///
    /// Equivalent to [`StencilTable::interpolate`] for `[f32; components]` data.
    /// Builds all transient GPU resources per call; for repeated evaluation hold
    /// a [`StencilEvalPipeline`] + [`StencilTableGpu`] and reuse buffers via
    /// [`evaluate_stencils`].
    pub fn evaluate(
        &self,
        table: &StencilTable,
        input: &[f32],
        components: u32,
    ) -> Result<Vec<f32>, KernelError> {
        if components == 0 || components > MAX_COMPONENTS {
            return Err(KernelError::Gpu(format!(
                "components {components} out of range 1..={MAX_COMPONENTS}"
            )));
        }
        let device = &self.device;
        let output_count = table.output_count();

        let gpu_table = StencilTableGpu::from((device, table));
        let pipeline = StencilEvalPipeline::new(device);

        let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subdiv-kernels::eval_src"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let dst_len = output_count * components as usize;
        let dst_size = ((dst_len * std::mem::size_of::<f32>()) as u64).max(4);
        let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("subdiv-kernels::eval_dst"),
            size: dst_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let desc = BufferDescriptor::packed(components);
        evaluate_stencils(
            device,
            &self.queue,
            &pipeline,
            &gpu_table,
            &src_buffer,
            &dst_buffer,
            desc,
            desc,
            0..output_count as u32,
        )?;

        let data = readback(device, &self.queue, &dst_buffer, dst_size);
        Ok(data[..dst_len].to_vec())
    }

    /// Sparse re-evaluation: recompute only the `affected` output rows from
    /// `input`, splicing them into `prior_output` (the previous dense result).
    /// Pair with `RefinementResult::affected_outputs` for incremental edits.
    ///
    /// The result equals a full dense re-evaluation when `affected` covers every
    /// output row that actually changed.
    pub fn evaluate_sparse(
        &self,
        table: &StencilTable,
        input: &[f32],
        components: u32,
        affected: &[u32],
        prior_output: &[f32],
    ) -> Result<Vec<f32>, KernelError> {
        if components == 0 || components > MAX_COMPONENTS {
            return Err(KernelError::Gpu(format!(
                "components {components} out of range 1..={MAX_COMPONENTS}"
            )));
        }
        let device = &self.device;
        let dst_len = table.output_count() * components as usize;
        if prior_output.len() != dst_len {
            return Err(KernelError::Gpu(format!(
                "prior_output length {} != output_count*components {dst_len}",
                prior_output.len()
            )));
        }

        let gpu_table = StencilTableGpu::from((device, table));
        let pipeline = StencilEvalPipeline::new(device);

        let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subdiv-kernels::sparse_src"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Seed dst with the prior output; only affected rows get overwritten.
        let dst_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subdiv-kernels::sparse_dst"),
            contents: bytemuck::cast_slice(prior_output),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let indirection = storage_buffer(
            device,
            "subdiv-kernels::sparse_indirection",
            bytemuck::cast_slice(affected),
        );

        let desc = BufferDescriptor::packed(components);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("subdiv-kernels::evaluate_sparse"),
        });
        pipeline.encode_indexed(
            device,
            &mut encoder,
            &gpu_table,
            &src_buffer,
            &dst_buffer,
            &indirection,
            desc,
            desc,
            affected.len() as u32,
        )?;
        self.queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let dst_size = ((dst_len * std::mem::size_of::<f32>()) as u64).max(4);
        let data = readback(device, &self.queue, &dst_buffer, dst_size);
        Ok(data[..dst_len].to_vec())
    }
}

/// Layout of a primvar buffer for the kernel, in floats.
#[derive(Debug, Clone, Copy)]
pub struct BufferDescriptor {
    /// Offset to the first element, in floats.
    pub offset: u32,
    /// Stride between consecutive elements, in floats.
    pub stride: u32,
    /// Components per element (e.g. 3 for xyz).
    pub length: u32,
}

impl BufferDescriptor {
    /// A tightly-packed buffer of `components`-vectors starting at offset 0.
    pub fn packed(components: u32) -> Self {
        Self {
            offset: 0,
            stride: components,
            length: components,
        }
    }
}

/// GPU-resident CSR stencil table (row offsets, indices, weights).
#[derive(Debug)]
pub struct StencilTableGpu {
    output_count: u32,
    offsets: wgpu::Buffer,
    indices: wgpu::Buffer,
    weights: wgpu::Buffer,
}

/// Upload a [`StencilTable`] into GPU storage buffers.
impl<'a, 'b> From<(&'a wgpu::Device, &'b StencilTable)> for StencilTableGpu {
    fn from((device, table): (&'a wgpu::Device, &'b StencilTable)) -> Self {
        Self {
            output_count: table.output_count() as u32,
            offsets: storage_buffer(
                device,
                "subdiv-kernels::stencil_offsets",
                bytemuck::cast_slice(&table.offsets),
            ),
            indices: storage_buffer(
                device,
                "subdiv-kernels::stencil_indices",
                bytemuck::cast_slice(&table.indices),
            ),
            weights: storage_buffer(
                device,
                "subdiv-kernels::stencil_weights",
                bytemuck::cast_slice(&table.weights),
            ),
        }
    }
}

impl StencilTableGpu {
    /// Number of output rows this table produces.
    pub fn output_count(&self) -> u32 {
        self.output_count
    }
}

/// Upload `bytes` as a storage buffer, substituting a 4-byte zero buffer when
/// empty (wgpu rejects zero-sized buffers).
fn storage_buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    const FALLBACK: [u8; 4] = [0u8; 4];
    let contents = if bytes.is_empty() {
        &FALLBACK[..]
    } else {
        bytes
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    src_offset: u32,
    dst_offset: u32,
    src_stride: u32,
    dst_stride: u32,
    length: u32,
    batch_start: u32,
    batch_end: u32,
    // Pad to 32 bytes: a uniform struct rounds up to a 16-byte multiple.
    _pad: u32,
}

/// Compute pipeline + bind-group layout for stencil evaluation.
#[derive(Debug)]
pub struct StencilEvalPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    indexed_bind_group_layout: wgpu::BindGroupLayout,
    indexed_pipeline: wgpu::ComputePipeline,
    workgroup_size: u32,
}

impl StencilEvalPipeline {
    /// Build the pipeline with the default workgroup size (64).
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_workgroup_size(device, DEFAULT_WORKGROUP_SIZE)
    }

    /// Build the pipeline with a specific workgroup size.
    pub fn with_workgroup_size(device: &wgpu::Device, workgroup_size: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subdiv-kernels::stencil_eval"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(STENCIL_EVAL_WGSL)),
        });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Bindings 0..=5, shared by the dense and indexed kernels.
        let base_entries = vec![
            // 0: uniform params
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShaderParams>() as u64,
                    ),
                },
                count: None,
            },
            // 1: src (read-only)
            storage_ro(1),
            // 2: dst (read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: offsets, 4: indices, 5: weights
            storage_ro(3),
            storage_ro(4),
            storage_ro(5),
        ];

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("subdiv-kernels::stencil_eval_bgl"),
            entries: &base_entries,
        });

        // The indexed kernel adds binding 6: the indirection buffer.
        let mut indexed_entries = base_entries.clone();
        indexed_entries.push(storage_ro(6));
        let indexed_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("subdiv-kernels::stencil_eval_indexed_bgl"),
                entries: &indexed_entries,
            });

        let make_pipeline = |bgl: &wgpu::BindGroupLayout, entry: &str, label: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WORKGROUP_SIZE", workgroup_size as f64)],
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            })
        };

        let pipeline = make_pipeline(
            &bind_group_layout,
            "eval_stencils",
            "subdiv-kernels::stencil_eval_pipeline",
        );
        let indexed_pipeline = make_pipeline(
            &indexed_bind_group_layout,
            "eval_stencils_indexed",
            "subdiv-kernels::stencil_eval_indexed_pipeline",
        );

        Self {
            bind_group_layout,
            pipeline,
            indexed_bind_group_layout,
            indexed_pipeline,
            workgroup_size,
        }
    }

    /// Encode a stencil-evaluation dispatch for the output rows in
    /// `batch_range` into `encoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_table: &StencilTableGpu,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        src_desc: BufferDescriptor,
        dst_desc: BufferDescriptor,
        batch_range: std::ops::Range<u32>,
    ) -> Result<(), KernelError> {
        if dst_desc.length > MAX_COMPONENTS {
            return Err(KernelError::Gpu(format!(
                "primvar length {} exceeds kernel capacity {MAX_COMPONENTS}",
                dst_desc.length
            )));
        }

        let params = ShaderParams {
            src_offset: src_desc.offset,
            dst_offset: dst_desc.offset,
            src_stride: src_desc.stride,
            dst_stride: dst_desc.stride,
            length: dst_desc.length,
            batch_start: batch_range.start,
            batch_end: batch_range.end,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subdiv-kernels::stencil_params"),
            contents: bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subdiv-kernels::stencil_eval_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_table.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_table.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_table.weights.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("subdiv-kernels::stencil_eval"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let invocations = batch_range.end.saturating_sub(batch_range.start);
        let groups = invocations.div_ceil(self.workgroup_size);
        if groups > 0 {
            pass.dispatch_workgroups(groups, 1, 1);
        }
        drop(pass);
        Ok(())
    }

    /// Encode a sparse (indexed) dispatch: recompute only the output rows named
    /// by `indirection` (e.g. from `affected_outputs`), leaving the other rows of
    /// `dst_buffer` untouched. `count` is the number of indirection entries.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indexed(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_table: &StencilTableGpu,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        indirection: &wgpu::Buffer,
        src_desc: BufferDescriptor,
        dst_desc: BufferDescriptor,
        count: u32,
    ) -> Result<(), KernelError> {
        if dst_desc.length > MAX_COMPONENTS {
            return Err(KernelError::Gpu(format!(
                "primvar length {} exceeds kernel capacity {MAX_COMPONENTS}",
                dst_desc.length
            )));
        }

        let params = ShaderParams {
            src_offset: src_desc.offset,
            dst_offset: dst_desc.offset,
            src_stride: src_desc.stride,
            dst_stride: dst_desc.stride,
            length: dst_desc.length,
            batch_start: 0,
            batch_end: count,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subdiv-kernels::stencil_params"),
            contents: bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subdiv-kernels::stencil_eval_indexed_bg"),
            layout: &self.indexed_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_table.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_table.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_table.weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: indirection.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("subdiv-kernels::stencil_eval_indexed"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.indexed_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = count.div_ceil(self.workgroup_size);
        if groups > 0 {
            pass.dispatch_workgroups(groups, 1, 1);
        }
        drop(pass);
        Ok(())
    }
}

/// One-shot: encode, submit, and wait. The result lands in `dst_buffer`
/// (which must be at least `output_count * dst_desc.stride` floats).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_stencils(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &StencilEvalPipeline,
    gpu_table: &StencilTableGpu,
    src_buffer: &wgpu::Buffer,
    dst_buffer: &wgpu::Buffer,
    src_desc: BufferDescriptor,
    dst_desc: BufferDescriptor,
    batch_range: std::ops::Range<u32>,
) -> Result<(), KernelError> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("subdiv-kernels::evaluate_stencils"),
    });
    pipeline.encode(
        device,
        &mut encoder,
        gpu_table,
        src_buffer,
        dst_buffer,
        src_desc,
        dst_desc,
        batch_range,
    )?;
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    Ok(())
}

/// Copy a GPU buffer back to a `Vec<f32>` (blocking).
fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size_bytes: u64,
) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("subdiv-kernels::readback"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("subdiv-kernels::readback_copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv()
        .expect("map_async callback dropped")
        .expect("buffer map failed");

    let data = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    data
}
