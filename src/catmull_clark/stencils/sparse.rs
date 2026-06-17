//! Sparse stencil type + accumulation / packing helpers.
//!
//! Scheme-local plumbing shared by [`points`](super::points) (which
//! produces the sparse stencils) and [`refine`](super::refine) (which
//! packs them into the final [`StencilTable`]).

use crate::StencilTable;

/// Sparse stencil: (source_index, weight) pairs.
pub(crate) type Sparse = Vec<(u32, f32)>;

/// Accumulate `source` scaled by `w` into `stencil`.
pub(crate) fn merge(stencil: &mut Sparse, source: &[(u32, f32)], w: f32) {
    source.iter().for_each(|&(idx, sw)| {
        if let Some(entry) = stencil.iter_mut().find(|(i, _)| *i == idx) {
            entry.1 += sw * w;
        } else {
            stencil.push((idx, sw * w));
        }
    });
}

/// Single source with weight 1.0.
pub(super) fn identity_entry(idx: u32) -> Sparse {
    vec![(idx, 1.0)]
}

/// Pack sparse stencils into a CSR [`StencilTable`].
pub(crate) fn pack(stencils: &[Sparse]) -> StencilTable {
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
