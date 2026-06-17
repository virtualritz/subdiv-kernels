//! √3 face-varying stencil computation.
//!
//! One stencil per refined face-corner, in `child.mesh.face_vertex_indices`
//! order. √3 children are centroids (face averages) and vertex points; the
//! shared origin-based builder reads the post-flip child connectivity so the
//! edge-flip step needs no special handling here.

use crate::face_varying::{all_linear_via_origin, pack, smooth_modes};
use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, KernelError, SchemeOptions, StencilTable,
};

use super::stencils::{Sqrt3LevelData, vertex_stencils_from_level};

/// Face-varying stencils for one level of √3 refinement, mapping `channel`
/// (the parent level's values) to the refined corners of `child`.
pub(crate) fn fvar_stencils_once(
    parent: &Sqrt3LevelData,
    child: &Sqrt3LevelData,
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
