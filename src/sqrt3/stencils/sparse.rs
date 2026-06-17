//! Sparse stencil type + packing helper.
//!
//! Sqrt3 reuses the crate-root `sharpness::merge` helper instead of
//! defining its own, so this module only carries the `Sparse` alias
//! and `pack`.

use crate::StencilTable;

/// Sparse stencil: (source_index, weight) pairs.
pub(super) type Sparse = Vec<(u32, f32)>;

/// Pack sparse stencils into a CSR [`StencilTable`].
pub(super) fn pack(stencils: &[Sparse]) -> StencilTable {
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
