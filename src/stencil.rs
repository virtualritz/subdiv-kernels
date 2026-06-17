//! CSR-packed stencil tables for subdivision interpolation.

use crate::Interpolatable;

/// A sparse linear map from input points to output points.
///
/// Each output point is a weighted sum of a few input points — its *stencil*.
/// Apply it to any [`Interpolatable`] data with [`interpolate()`](Self::interpolate).
///
/// Stored compressed (CSR): output `i`'s source indices and weights are the
/// slices `indices[offsets[i]..offsets[i + 1]]` and the matching `weights`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct StencilTable {
    /// CSR row offsets. Length = output_count + 1.
    pub offsets: Vec<u32>,
    /// Source indices (into the input buffer), flat.
    pub indices: Vec<u32>,
    /// Source weights (parallel to `indices`).
    pub weights: Vec<f32>,
}

impl StencilTable {
    /// Number of output points this table produces.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Apply stencils to an input buffer, producing one output value per stencil.
    #[inline]
    pub fn interpolate<T: Interpolatable>(&self, input: &[T]) -> Vec<T> {
        (0..self.output_count())
            .map(|i| {
                let start = self.offsets[i] as usize;
                let end = self.offsets[i + 1] as usize;
                let mut result = T::default();
                self.indices[start..end]
                    .iter()
                    .zip(&self.weights[start..end])
                    .for_each(|(&idx, &w)| result.add_with_weight(&input[idx as usize], w));
                result
            })
            .collect()
    }

    /// Apply stencils for only `rows`, scattering each result into
    /// `output[row]` and leaving every other entry untouched.
    ///
    /// The CPU analogue of an indexed (sparse) dispatch: pair with
    /// [`affected_outputs`](crate::InverseStencilMap::affected_outputs) to
    /// re-evaluate only the outputs a control-point edit changed, splicing them
    /// into the previous output buffer. Bit-identical to
    /// [`interpolate`](Self::interpolate) on the recomputed rows.
    ///
    /// `output` must have at least [`output_count`](Self::output_count) entries
    /// and every index in `rows` must be `< output_count`.
    pub fn interpolate_rows<T: Interpolatable>(&self, input: &[T], rows: &[u32], output: &mut [T]) {
        for &row in rows {
            let r = row as usize;
            let start = self.offsets[r] as usize;
            let end = self.offsets[r + 1] as usize;
            let mut result = T::default();
            self.indices[start..end]
                .iter()
                .zip(&self.weights[start..end])
                .for_each(|(&idx, &w)| result.add_with_weight(&input[idx as usize], w));
            output[r] = result;
        }
    }

    /// Compose two stencil tables: `self` maps A→B, `other` maps B→C.
    /// The result maps A→C by substituting B's stencils into C's.
    pub fn compose(&self, other: &StencilTable) -> Self {
        let mut offsets = Vec::with_capacity(other.output_count() + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        offsets.push(0);

        (0..other.output_count()).for_each(|c| {
            // For output point c in `other`, accumulate the composed stencil.
            // other's stencil for c references points in B-space.
            // For each B-point, expand via self's stencil into A-space.
            let c_start = other.offsets[c] as usize;
            let c_end = other.offsets[c + 1] as usize;

            // Accumulate into a sparse map: A-index → combined weight.
            let mut combined: Vec<(u32, f32)> = Vec::new();

            other.indices[c_start..c_end]
                .iter()
                .zip(&other.weights[c_start..c_end])
                .for_each(|(&b_idx, &b_weight)| {
                    let b = b_idx as usize;
                    let b_start = self.offsets[b] as usize;
                    let b_end = self.offsets[b + 1] as usize;

                    self.indices[b_start..b_end]
                        .iter()
                        .zip(&self.weights[b_start..b_end])
                        .for_each(|(&a_idx, &a_weight)| {
                            let w = b_weight * a_weight;
                            // Merge into combined list.
                            if let Some(entry) = combined.iter_mut().find(|(idx, _)| *idx == a_idx)
                            {
                                entry.1 += w;
                            } else {
                                combined.push((a_idx, w));
                            }
                        });
                });

            combined.iter().for_each(|&(idx, w)| {
                indices.push(idx);
                weights.push(w);
            });
            offsets.push(indices.len() as u32);
        });

        Self {
            offsets,
            indices,
            weights,
        }
    }

    /// Identity table of `count` rows: each output copies its input unchanged.
    pub fn identity(count: usize) -> Self {
        let offsets = (0..=count as u32).collect();
        let indices = (0..count as u32).collect();
        let weights = vec![1.0; count];
        Self {
            offsets,
            indices,
            weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_preserves_input() {
        let table = StencilTable::identity(3);
        let input: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let output = table.interpolate(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn midpoint_stencil() {
        let table = StencilTable {
            offsets: vec![0, 2],
            indices: vec![0, 1],
            weights: vec![0.5, 0.5],
        };
        let input = vec![[0.0_f32, 0.0], [2.0, 4.0]];
        let output = table.interpolate(&input);
        assert_eq!(output, vec![[1.0, 2.0]]);
    }

    #[test]
    fn compose_identity_is_identity() {
        let a = StencilTable::identity(3);
        let b = StencilTable {
            offsets: vec![0, 2, 4],
            indices: vec![0, 1, 1, 2],
            weights: vec![0.5, 0.5, 0.5, 0.5],
        };
        let composed = a.compose(&b);
        let input = vec![1.0_f32, 3.0, 5.0];
        assert_eq!(composed.interpolate(&input), b.interpolate(&input));
    }

    #[test]
    fn compose_chains_correctly() {
        // A→B: 2 inputs → 3 outputs
        // B[0] = 0.5*A[0] + 0.5*A[1], B[1] = 1.0*A[1], B[2] = 1.0*A[0]
        let ab = StencilTable {
            offsets: vec![0, 2, 3, 4],
            indices: vec![0, 1, 1, 0],
            weights: vec![0.5, 0.5, 1.0, 1.0],
        };
        // B→C: 3 inputs → 1 output (average of all B points)
        let bc = StencilTable {
            offsets: vec![0, 3],
            indices: vec![0, 1, 2],
            weights: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        };

        let ac = ab.compose(&bc);
        let input = vec![2.0_f32, 8.0];

        let b = ab.interpolate(&input);
        let c_via_b = bc.interpolate(&b);
        let c_direct = ac.interpolate(&input);

        c_via_b
            .iter()
            .zip(c_direct.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn interpolate_f64() {
        let table = StencilTable {
            offsets: vec![0, 3],
            indices: vec![0, 1, 2],
            weights: vec![0.25, 0.5, 0.25],
        };
        let input: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [4.0, 8.0, 12.0], [0.0, 0.0, 0.0]];
        let output = table.interpolate(&input);
        assert_eq!(output, vec![[2.0, 4.0, 6.0]]);
    }

    #[test]
    fn interpolate_rows_matches_full_on_subset_and_preserves_others() {
        // out0 = in0, out1 = (in0+in1)/2, out2 = in1, out3 = (in1+in2)/2.
        let table = StencilTable {
            offsets: vec![0, 1, 3, 4, 6],
            indices: vec![0, 0, 1, 1, 1, 2],
            weights: vec![1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        };
        let input: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]];
        let full = table.interpolate(&input);

        let sentinel = [-9.0_f32, -9.0, -9.0];
        let mut out = vec![sentinel; table.output_count()];
        table.interpolate_rows(&input, &[1, 3], &mut out);

        // Recomputed rows match the full eval bit-for-bit...
        assert_eq!(out[1], full[1]);
        assert_eq!(out[3], full[3]);
        // ...and rows not listed are left exactly as they were.
        assert_eq!(out[0], sentinel);
        assert_eq!(out[2], sentinel);
    }
}
