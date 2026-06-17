//! Catmull-Clark vertex stencil extraction.
//!
//! Split across four submodules:
//!
//! - [`sparse`]: `Sparse` alias + merge/pack/identity plumbing.
//! - [`points`]: per-point stencil rules (face, edge, vertex).
//! - [`level`]: [`CcLevelData`] + base-level data-gathering helpers.
//! - [`refine`]: [`refine_topology_once`] + [`vertex_stencils_from_level`].
//!
//! Each level's topology is computed once via
//! [`refine_topology_once`] and cached in [`CcLevelData`]; vertex
//! stencils are then extracted on demand via
//! [`vertex_stencils_from_level`] without rebuilding topology.

mod level;
mod points;
mod refine;
mod sparse;

pub(crate) use level::{CcLevelData, base_level_data};
pub(crate) use refine::{refine_topology_once, vertex_stencils_from_level};
pub(crate) use sparse::{Sparse, merge, pack};
