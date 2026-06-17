//! Sparse stencil type + accumulation / packing helpers.

use crate::StencilTable;

/// Sparse stencil: (source_index, weight) pairs.
pub(super) type Sparse = Vec<(u32, f32)>;

/// Accumulate `source` scaled by `w` into `stencil`.
pub(super) fn merge(stencil: &mut Sparse, source: &[(u32, f32)], w: f32) {
    source.iter().for_each(|&(idx, sw)| {
        if let Some(entry) = stencil.iter_mut().find(|(i, _)| *i == idx) {
            entry.1 += sw * w;
        } else {
            stencil.push((idx, sw * w));
        }
    });
}

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
