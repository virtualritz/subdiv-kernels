//! Loop subdivision vertex stencil extraction.
//!
//! Mirrors the CC stencils layout:
//!
//! - [`sparse`]: `Sparse` alias + merge/pack plumbing.
//! - [`points`]: per-point stencil rules (edge, vertex, boundary,
//!   crease blending).
//! - [`level`]: [`LoopLevelData`] + base-level data-gathering helpers.
//! - [`refine`]: [`refine_topology_once`] + [`vertex_stencils_from_level`].

mod level;
mod points;
mod refine;
mod sparse;

pub(crate) use level::{LoopLevelData, base_level_data};
pub(crate) use refine::{refine_topology_once, vertex_stencils_from_level};
