//! CC face-varying stencil computation.
//!
//! Face-varying stencils operate on per-face-corner values using
//! [`FaceVaryingChannel`] for seam detection. The output has one
//! stencil per face-corner in the refined mesh.
//!
//! `Linear` keeps CC's own purely face-local rule. The smooth-spectrum modes
//! (`Smooth`, `SmoothWithLinearCorners`, `SmoothWithLinearBoundaries`) reuse
//! the scheme-agnostic [`crate::face_varying::smooth_modes`], which remaps
//! CC's *positional* per-child-vertex stencils home-side into face-varying
//! space — so the face/edge/vertex-point rules (and their crease, corner and
//! smooth-triangle handling) are inherited from the validated positional path
//! rather than re-derived here.

use crate::face_varying::smooth_modes;
use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, KernelError, Mesh, SchemeOptions, StencilTable,
};

use super::stencils::{CcLevelData, vertex_stencils_from_level};
use super::topology::Topology;

// ── Sparse stencil helpers ─────────────────────────────────────────────

type Sparse = Vec<(u32, f32)>;

fn pack(stencils: &[Sparse]) -> StencilTable {
    let mut offsets = Vec::with_capacity(stencils.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();

    offsets.push(0u32);
    stencils.iter().for_each(|s| {
        s.iter().for_each(|&(idx, w)| {
            indices.push(idx);
            weights.push(w);
        });
        offsets.push(indices.len() as u32);
    });

    StencilTable {
        offsets,
        indices,
        weights,
    }
}

/// Compute CSR-style face offsets from face_vertex_counts.
fn face_offsets(topo: &Mesh) -> Vec<usize> {
    std::iter::once(0)
        .chain(topo.face_vertex_counts.iter().scan(0usize, |acc, &c| {
            *acc += c as usize;
            Some(*acc)
        }))
        .collect()
}

// ── AllLinear stencils ─────────────────────────────────────────────────

/// AllLinear: purely face-local, seam-preserving by construction.
///
/// For each selected parent face corner `i` in face `fi`:
/// - face_point: centroid of this face's FVar corners
/// - ep_prev: midpoint of corner (i-1) and corner i
/// - vertex_point: copy of corner i
/// - ep_curr: midpoint of corner i and corner (i+1)
fn all_linear_stencils(
    topo: &Topology,
    fvar: &FaceVaryingChannel,
    offsets: &[usize],
    face_selected: &[bool],
) -> Vec<Sparse> {
    let mut out = Vec::new();

    (0..topo.faces.len()).for_each(|fi| {
        let start = offsets[fi];
        let n = topo.faces.row_len(fi);

        if face_selected[fi] {
            let inv_n = 1.0 / n as f32;
            // Face-point stencil (shared across child quads of this face)
            let face_pt: Sparse = (0..n).map(|k| (fvar.indices[start + k], inv_n)).collect();

            (0..n).for_each(|i| {
                let prev = (i + n - 1) % n;
                let next = (i + 1) % n;

                let fv_prev = fvar.indices[start + prev];
                let fv_curr = fvar.indices[start + i];
                let fv_next = fvar.indices[start + next];

                // Child quad corners: [face_pt, ep_prev, vertex_pt, ep_curr]
                out.push(face_pt.clone());
                out.push(vec![(fv_prev, 0.5), (fv_curr, 0.5)]);
                out.push(vec![(fv_curr, 1.0)]);
                out.push(vec![(fv_curr, 0.5), (fv_next, 0.5)]);
            });
        } else {
            // Unselected: identity copy of each face corner
            (0..n).for_each(|k| {
                out.push(vec![(fvar.indices[start + k], 1.0)]);
            });
        }
    });

    out
}

// ── Public entry point ─────────────────────────────────────────────────

/// Compute face-varying stencils for one level of CC refinement.
///
/// Returns a [`StencilTable`] mapping input FVar values → refined FVar values.
/// The output has one entry per face-corner in the refined mesh (in the same
/// order as `child.mesh.face_vertex_indices`).
pub(crate) fn fvar_stencils_once(
    parent: &CcLevelData,
    child: &CcLevelData,
    channel: &FaceVaryingChannel,
    mode: FaceVaryingInterpolation,
    options: &SchemeOptions,
) -> Result<StencilTable, KernelError> {
    let stencils = match mode {
        FaceVaryingInterpolation::Linear => {
            let offsets = face_offsets(&parent.mesh);
            all_linear_stencils(&parent.topo, channel, &offsets, &parent.face_selected)
        }
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
