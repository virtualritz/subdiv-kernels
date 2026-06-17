//! Loop face-varying stencil computation.
//!
//! One stencil per refined face-corner, in `child.mesh.face_vertex_indices`
//! order. `Linear` is face-local and seam-preserving;
//! `SmoothWithLinearBoundaries` uses Loop's positional masks in the interior
//! (see [`crate::face_varying`]).

use crate::face_varying::{all_linear_via_origin, pack, smooth_modes};
use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, KernelError, SchemeOptions, StencilTable,
};

use super::stencils::{LoopLevelData, vertex_stencils_from_level};

/// Face-varying stencils for one level of Loop refinement, mapping
/// `channel` (the parent level's face-varying values) to the refined
/// corners of `child`.
pub(crate) fn fvar_stencils_once(
    parent: &LoopLevelData,
    child: &LoopLevelData,
    channel: &FaceVaryingChannel,
    mode: FaceVaryingInterpolation,
    options: &SchemeOptions,
) -> Result<StencilTable, KernelError> {
    let stencils = match mode {
        FaceVaryingInterpolation::Linear => all_linear_via_origin(
            &parent.mesh,
            &parent.topo.edge_vertices,
            &child.mesh,
            &child.lineage,
            channel,
        ),
        FaceVaryingInterpolation::Smooth
        | FaceVaryingInterpolation::SmoothWithLinearCorners
        | FaceVaryingInterpolation::SmoothWithLinearBoundaries => {
            let pos = vertex_stencils_from_level(parent, options);
            smooth_modes(
                &parent.mesh,
                &parent.topo.edge_vertices,
                &child.mesh,
                &child.lineage,
                &pos,
                channel,
                mode,
            )
        }
    };

    Ok(pack(&stencils))
}
