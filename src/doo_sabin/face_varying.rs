//! Doo-Sabin face-varying stencil computation.
//!
//! One stencil per refined face-corner, in `child.mesh.face_vertex_indices`
//! order. Every Doo-Sabin child vertex is a face-vertex point whose index is
//! the parent corner index: `Linear` copies each source corner's value; the
//! smooth modes use Doo-Sabin's face-local positional mask (see
//! [`crate::face_varying::smooth_doo_sabin`]).

use crate::face_varying::{all_linear_copy_corners, pack, smooth_doo_sabin};
use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, KernelError, SchemeOptions, StencilTable,
};

use super::stencils::{DooSabinLevelData, vertex_stencils_from_level};

/// Face-varying stencils for one level of Doo-Sabin refinement, mapping
/// `channel` (the parent level's values) to the refined corners of `child`.
pub(crate) fn fvar_stencils_once(
    parent: &DooSabinLevelData,
    child: &DooSabinLevelData,
    channel: &FaceVaryingChannel,
    mode: FaceVaryingInterpolation,
    options: &SchemeOptions,
) -> Result<StencilTable, KernelError> {
    let stencils = match mode {
        FaceVaryingInterpolation::Linear => all_linear_copy_corners(&child.mesh, channel),
        // Doo-Sabin's smooth rule is face-local (each face-vertex point reads
        // only its own face), so seams never mix and all three smooth modes
        // coincide.
        FaceVaryingInterpolation::SmoothWithLinearBoundaries
        | FaceVaryingInterpolation::SmoothWithLinearCorners
        | FaceVaryingInterpolation::Smooth => {
            let pos = vertex_stencils_from_level(parent, options);
            smooth_doo_sabin(&parent.mesh, &child.mesh, &pos, channel)
        }
    };

    Ok(pack(&stencils))
}
