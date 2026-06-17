#![cfg_attr(docsrs, feature(doc_cfg))]
//! Subdivision surface kernels — Catmull–Clark, Loop, √3, and Doo–Sabin.
//!
//! Subdivision refines a coarse polygon **control mesh** into a finer, smoother
//! one: each step splits the faces and places the new points as weighted
//! averages of their neighbors, converging to a smooth *limit surface*.
//!
//! This crate computes that refinement's connectivity and weights and returns
//! [`StencilTable`]s — sparse maps where each output point is a weighted sum of
//! a few input points. Supply a [`topology::Mesh`] (the control cage) and apply
//! the stencils to your own per-vertex data (positions, UVs, colors, …); the
//! crate holds no geometry and needs no host mesh type.
//!
//! # Example
//!
//! Refine a tetrahedron and exercise the main pieces of the API — one-shot
//! interpolation, composed-table re-evaluation, sparse-edit queries,
//! face-varying (UV) channels, the cached refinement handle, and limit
//! stencils.
//!
//! ```
//! use std::num::NonZeroU8;
//! use subdiv_kernels::{
//!     topology::{FaceVaryingChannel, Mesh},
//!     FaceVaryingInterpolation, Refiner, Scheme, SchemeOptions, UniformRefine,
//! };
//!
//! // A tetrahedron: 4 vertices, 4 triangular faces, 6 edges (closed surface).
//! let face_vertex_indices = vec![0, 1, 2, /**/ 0, 2, 3, /**/ 0, 3, 1, /**/ 1, 3, 2];
//! let mesh = Mesh {
//!     vertex_count: 4,
//!     face_vertex_counts: vec![3; 4],
//!     face_vertex_indices: face_vertex_indices.clone(),
//!     edge_vertices: vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
//!     edge_creases: vec![0.0; 6],
//!     vertex_corners: vec![0.0; 4],
//! };
//! let positions: Vec<[f32; 3]> =
//!     vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
//!
//! let refiner = Refiner::new(mesh, Scheme::CatmullClark, SchemeOptions::default())?;
//! let req = UniformRefine::from(NonZeroU8::new(2).unwrap());
//!
//! // One-shot: interpolate any per-vertex data through all levels.
//! let result = refiner.refine_uniform(&req)?;
//! let refined = result.interpolate(&positions);
//! assert_eq!(refined.len(), result.topology.vertex_count as usize);
//!
//! // Animation: compose the per-level stencils once, re-evaluate each frame.
//! // Same surface as the chained path (up to f32 rounding).
//! let composed = result.compose_stencils(positions.len());
//! let composed_positions = composed.interpolate(&positions);
//! assert!(composed_positions.iter().zip(&refined).all(|(a, b)| {
//!     a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
//! }));
//!
//! // Sparse edits: which refined outputs move when control point 0 moves?
//! assert!(!result.affected_outputs(&[0]).is_empty());
//!
//! // Face-varying UVs, smooth interior with linear island boundaries.
//! let uvs: Vec<[f32; 2]> = (0..4).map(|i| [i as f32, 0.0]).collect();
//! let uv_channel = FaceVaryingChannel { indices: face_vertex_indices, value_count: 4 };
//! let fvar_tables = refiner.face_varying_stencils(
//!     &req,
//!     &uv_channel,
//!     FaceVaryingInterpolation::SmoothWithLinearBoundaries,
//! )?;
//! let refined_uvs = fvar_tables.iter().fold(uvs, |d, t| t.interpolate(&d));
//! assert_eq!(refined_uvs.len(), result.topology.face_vertex_indices.len());
//!
//! // Cached handle: query per level without recomputing topology, then take
//! // the owned final mesh + adjacency.
//! let refinement = refiner.refine_topology(&req)?;
//! let parts = refinement.into_final_parts();
//! assert_eq!(parts.topology.vertex_count, result.topology.vertex_count);
//!
//! // Limit surface: stencils for limit positions and tangents/normals.
//! let _limit = result.limit_stencils()?;
//!
//! // Write the refined surface as a Wavefront OBJ (vertices, then faces).
//! let mut obj = String::new();
//! for [x, y, z] in &refined {
//!     obj += &format!("v {x} {y} {z}\n");
//! }
//! let mut corner = 0;
//! for &n in &result.topology.face_vertex_counts {
//!     obj += "f";
//!     for k in 0..n as usize {
//!         // OBJ indices are 1-based.
//!         obj += &format!(" {}", result.topology.face_vertex_indices[corner + k] + 1);
//!     }
//!     obj += "\n";
//!     corner += n as usize;
//! }
//! // std::fs::write("surface.obj", &obj)?;  // ← persist to disk
//! assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), refined.len());
//! # Ok::<(), subdiv_kernels::KernelError>(())
//! ```
//!
//! # Performance
//!
//! [`RefinementResult::interpolate`] chains the per-level stencils — the same
//! algorithmic cost as direct subdivision, best for a one-shot refine. For
//! animation (static topology, changing data),
//! [`RefinementResult::compose_stencils`] precomputes a single table mapping
//! control points straight to the final level, so each frame is one
//! [`StencilTable::interpolate`] call. Either path applies to any number of
//! data buffers (positions, UVs, …) that share the topology.
//!
//! # Implementing [`Interpolatable`]
//!
//! Any type with a weighted add can be subdivided. The crate ships impls for
//! `f32`, `f64`, and `[f32; N]` / `[f64; N]`; for your own types:
//!
//! ```
//! use subdiv_kernels::Interpolatable;
//!
//! #[derive(Default, Clone)]
//! struct Color { r: f32, g: f32, b: f32, a: f32 }
//!
//! impl Interpolatable for Color {
//!     fn add_with_weight(&mut self, src: &Self, weight: f32) {
//!         self.r += src.r * weight;
//!         self.g += src.g * weight;
//!         self.b += src.b * weight;
//!         self.a += src.a * weight;
//!     }
//! }
//! ```

mod catmull_clark;
mod closest_point;
pub(crate) mod csr;
mod doo_sabin;
mod error;
mod face_varying;
mod interpolate;
mod inverse;
mod limit;
mod limit_eval;
mod loop_subdivision;
mod options;
mod output;
mod patch;
mod refiner;
pub(crate) mod sharpness;
mod sqrt3;
mod stencil;
#[cfg(test)]
mod test_support;
pub mod topology;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use closest_point::ClosestPoint;
pub use error::KernelError;
pub use interpolate::Interpolatable;
pub use inverse::{AffectedScratch, InverseStencilChain, InverseStencilMap};
pub use limit::{LimitStencils, SectoredLimitStencils};
pub use limit_eval::{LimitEvaluator, LimitSample, MAX_ISOLATION_DEPTH};
pub use options::{
    BoundaryInterpolation, CornerRule, CreaseComputationMethod, FaceVaryingInterpolation, Scheme,
    SchemeOptions, TriangleSubdivisionRule, UniformRefine,
};
pub use output::{LineageMaps, RefinementResult, VertexOrigin};
pub use patch::{PatchTable, QuadClass};
pub use refiner::{RefinedFinalParts, Refinement, Refiner};
pub use stencil::StencilTable;
// Canonical home is the `topology` module (`topology::Mesh`); these are also
// re-exported at the crate root for terse `use subdiv_kernels::Mesh` sites.
pub use topology::{Adjacency, FaceVaryingChannel, Mesh};
#[cfg(feature = "wgpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "wgpu")))]
pub use wgpu::{
    BufferDescriptor, GpuContext, MAX_COMPONENTS, STENCIL_EVAL_WGSL, StencilEvalPipeline,
    StencilTableGpu, evaluate_stencils,
};

/// Common imports for typical use: `use subdiv_kernels::prelude::*;`.
pub mod prelude {
    pub use crate::{
        BoundaryInterpolation, CornerRule, CreaseComputationMethod, FaceVaryingChannel,
        FaceVaryingInterpolation, Interpolatable, KernelError, Mesh, Refiner, RefinementResult,
        Scheme, SchemeOptions, StencilTable, TriangleSubdivisionRule, UniformRefine,
    };
}
