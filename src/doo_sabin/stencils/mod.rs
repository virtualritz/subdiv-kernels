//! Doo-Sabin vertex stencil extraction.
//!
//! Mirrors the CC/Loop/Sqrt3 stencils layout: `sparse`, `points`,
//! `level`, `refine`. See [`super::stencils`] docs for the overall
//! two-phase design.

mod level;
mod points;
mod refine;
mod sparse;

pub(crate) use level::{DooSabinLevelData, base_level_data};
pub(crate) use refine::{refine_topology_once, vertex_stencils_from_level};
