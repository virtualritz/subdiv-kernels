//! Inverse stencil maps -- "which outputs change when these inputs change".
//!
//! A [`StencilTable`] maps input (control) points to output points: each output
//! row is a weighted sum of a bounded set of inputs. The *inverse* map is its
//! transpose -- for each input point, the set of output rows whose stencil
//! references it. Moving one control point changes exactly the outputs its
//! transpose lists; everything else is bit-identical. This is the locality
//! property the sparse / incremental evaluation path is built on.
//!
//! [`InverseStencilMap`] inverts a single table. The map is topology-only --
//! it depends on the stencils' sparsity pattern, not on any weights or data --
//! so it has the same lifetime as the stencil table: build it once when the
//! stencils are built, reuse it across every position edit.

use crate::StencilTable;
use crate::csr::CsrVec;

/// Transpose of a single [`StencilTable`].
///
/// [`affected_outputs`](Self::affected_outputs) maps a set of changed input
/// indices to the sorted, unique set of output rows whose stencils reference any
/// of them.
#[derive(Debug, Clone)]
pub struct InverseStencilMap {
    /// CSR transpose: row `i` lists the output rows referencing input `i`.
    /// `transpose.len()` is one past the largest referenced input index.
    transpose: CsrVec,
    output_count: usize,
}

impl<'a> From<&'a StencilTable> for InverseStencilMap {
    /// Build the inverse (transpose) of `table` in a single pass over its
    /// entries.
    fn from(table: &'a StencilTable) -> Self {
        let output_count = table.output_count();
        let input_count = table
            .indices
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);

        // Bucket each output row under every input its stencil references. One
        // pass over all stencil entries; visiting rows in increasing order
        // leaves each bucket sorted ascending.
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); input_count];
        for row in 0..output_count {
            let start = table.offsets[row] as usize;
            let end = table.offsets[row + 1] as usize;
            for &input in &table.indices[start..end] {
                buckets[input as usize].push(row as u32);
            }
        }

        Self {
            transpose: CsrVec::from_jagged_u32(&buckets),
            output_count,
        }
    }
}

impl InverseStencilMap {
    /// Number of output rows of the original table.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Output rows affected by changing `changed_inputs`, sorted ascending with
    /// duplicates removed. Input indices that no output references (or that are
    /// out of range) contribute nothing.
    pub fn affected_outputs(&self, changed_inputs: &[u32]) -> Vec<u32> {
        // Mark each affected row in a scratch bitset, collecting it on first
        // sight, then sort the (small) result. This drops the original's full
        // O(output_count) scan-to-collect -- the dominant cost on a large mesh --
        // without sorting the multiplicity-laden gather (which a plain
        // sort+dedup does, and which loses badly once the affected set is a
        // sizeable fraction). Robust across "tiny fraction of a huge mesh" and
        // "sizeable fraction of a small one"; a near-total change set should take
        // the dense path anyway (see the design doc's honest-payoff note).
        let mut seen = vec![false; self.output_count];
        let mut affected = Vec::new();
        for &input in changed_inputs {
            let i = input as usize;
            if i < self.transpose.len() {
                for &row in self.transpose.row(i) {
                    let r = row as usize;
                    if !seen[r] {
                        seen[r] = true;
                        affected.push(row);
                    }
                }
            }
        }
        affected.sort_unstable();
        affected
    }
}

/// Per-level inverse maps for a multi-level refinement.
///
/// `level_stencils[k]` maps level-k vertices to level-(k+1) vertices, so the
/// inverse of level `k` maps a changed set in level-k space to the affected set
/// in level-(k+1) space. [`affected_outputs`](Self::affected_outputs) threads a
/// base-cage change set forward through every level to the final-level outputs
/// it touches -- equivalent to inverting the composed table, but without paying
/// the composition cost.
///
/// The chain is topology-only: build it once when the stencils are built and
/// reuse it across every position edit.
#[derive(Debug, Clone)]
pub struct InverseStencilChain {
    levels: Vec<InverseStencilMap>,
}

impl<'a> From<&'a [StencilTable]> for InverseStencilChain {
    /// Build per-level inverse maps from a refinement's per-level stencil tables
    /// (e.g. [`RefinementResult::level_stencils`](crate::RefinementResult::level_stencils)).
    fn from(level_stencils: &'a [StencilTable]) -> Self {
        Self {
            levels: level_stencils.iter().map(InverseStencilMap::from).collect(),
        }
    }
}

impl InverseStencilChain {
    /// Number of refinement levels in the chain.
    #[inline]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Final-level output rows affected by changing the given base-level input
    /// (control) points, sorted ascending with duplicates removed.
    ///
    /// With no levels the chain is the identity: the affected outputs are the
    /// changed inputs themselves.
    pub fn affected_outputs(&self, changed_base_inputs: &[u32]) -> Vec<u32> {
        // One-shot: allocate a scratch and delegate to the reusable path. For a
        // hot edit loop, cache an `AffectedScratch` and call
        // `affected_outputs_into` directly to avoid this per-call allocation.
        let mut scratch = AffectedScratch::default();
        let mut out = Vec::new();
        self.affected_outputs_into(changed_base_inputs, &mut scratch, &mut out);
        out
    }

    /// Like [`affected_outputs`](Self::affected_outputs) but writes into `out`
    /// (cleared first) and reuses `scratch`, so a hot edit loop allocates
    /// nothing after the scratch has warmed up. Same sorted, deduped result.
    pub fn affected_outputs_into(
        &self,
        changed_base_inputs: &[u32],
        scratch: &mut AffectedScratch,
        out: &mut Vec<u32>,
    ) {
        let AffectedScratch {
            stamp,
            generation,
            front,
            back,
        } = scratch;

        out.clear();

        if self.levels.is_empty() {
            // Identity: the affected outputs are the changed inputs themselves.
            out.extend_from_slice(changed_base_inputs);
            out.sort_unstable();
            out.dedup();
            return;
        }

        let max_output = self
            .levels
            .iter()
            .map(|level| level.output_count)
            .max()
            .unwrap_or(0);
        if stamp.len() < max_output {
            stamp.resize(max_output, 0);
        }

        front.clear();
        front.extend_from_slice(changed_base_inputs);

        // Propagate forward, deduping each level's affected rows by generation
        // stamp (so the buffer is never cleared between calls).
        for level in &self.levels {
            *generation = generation.wrapping_add(1);
            if *generation == 0 {
                // Counter wrapped; reset so stale 0-stamps aren't read as seen.
                stamp.iter_mut().for_each(|s| *s = 0);
                *generation = 1;
            }
            let g = *generation;

            back.clear();
            for &input in front.iter() {
                let i = input as usize;
                if i < level.transpose.len() {
                    for &row in level.transpose.row(i) {
                        let r = row as usize;
                        if stamp[r] != g {
                            stamp[r] = g;
                            back.push(row);
                        }
                    }
                }
            }
            back.sort_unstable();
            std::mem::swap(front, back);
        }

        std::mem::swap(out, front);
    }
}

/// Reusable scratch for [`InverseStencilChain::affected_outputs_into`].
///
/// Holds a generation-stamped row buffer plus ping-pong work vectors so a hot
/// edit loop pays no per-query allocation: build one and reuse it across edits.
/// Safe to reuse across different chains/topologies -- the row buffer grows as
/// needed and the generation counter is monotonic.
#[derive(Debug, Clone, Default)]
pub struct AffectedScratch {
    /// Per-row "last seen" generation; a row is hit this pass iff its stamp
    /// equals the current generation. Sized to the largest level's output.
    stamp: Vec<u32>,
    generation: u32,
    /// Ping-pong buffers for the change set propagating through the levels.
    front: Vec<u32>,
    back: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small table: 4 outputs over 3 inputs.
    /// - out0 = in0
    /// - out1 = (in0 + in1) / 2
    /// - out2 = in1
    /// - out3 = (in1 + in2) / 2
    fn sample_table() -> StencilTable {
        StencilTable {
            offsets: vec![0, 1, 3, 4, 6],
            indices: vec![0, 0, 1, 1, 1, 2],
            weights: vec![1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        }
    }

    /// Brute-force oracle: an output is affected iff its stencil references any
    /// changed input.
    fn brute_force_affected(table: &StencilTable, changed: &[u32]) -> Vec<u32> {
        let set: std::collections::HashSet<u32> = changed.iter().copied().collect();
        (0..table.output_count() as u32)
            .filter(|&r| {
                let s = table.offsets[r as usize] as usize;
                let e = table.offsets[r as usize + 1] as usize;
                table.indices[s..e].iter().any(|i| set.contains(i))
            })
            .collect()
    }

    #[test]
    fn affected_outputs_for_single_input() {
        let inv = InverseStencilMap::from(&sample_table());
        // in0 feeds out0, out1.
        assert_eq!(inv.affected_outputs(&[0]), vec![0, 1]);
        // in1 feeds out1, out2, out3.
        assert_eq!(inv.affected_outputs(&[1]), vec![1, 2, 3]);
        // in2 feeds out3 only.
        assert_eq!(inv.affected_outputs(&[2]), vec![3]);
    }

    #[test]
    fn affected_outputs_is_sorted_unique_union() {
        let inv = InverseStencilMap::from(&sample_table());
        // Union of in0 -> {0,1} and in2 -> {3}, with a duplicate input.
        assert_eq!(inv.affected_outputs(&[2, 0, 0]), vec![0, 1, 3]);
    }

    #[test]
    fn affected_outputs_matches_brute_force() {
        let table = sample_table();
        let inv = InverseStencilMap::from(&table);
        for changed in [
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 1, 2],
        ] {
            assert_eq!(
                inv.affected_outputs(&changed),
                brute_force_affected(&table, &changed),
                "changed = {changed:?}"
            );
        }
    }

    #[test]
    fn out_of_range_or_empty_input_affects_nothing() {
        let inv = InverseStencilMap::from(&sample_table());
        assert!(inv.affected_outputs(&[99]).is_empty());
        assert!(inv.affected_outputs(&[]).is_empty());
    }

    /// Two chained tables: A(3) -> B(4) -> C(3).
    fn two_level_tables() -> (StencilTable, StencilTable) {
        let t0 = sample_table(); // A(3) -> B(4)
        // B(4) -> C(3): c0 = b0, c1 = (b1 + b2)/2, c2 = b3.
        let t1 = StencilTable {
            offsets: vec![0, 1, 3, 4],
            indices: vec![0, 1, 2, 3],
            weights: vec![1.0, 0.5, 0.5, 1.0],
        };
        (t0, t1)
    }

    #[test]
    fn chain_affected_outputs_matches_composed_table() {
        // Oracle: invert the *composed* A->C table by brute force. The chain's
        // forward propagation must give the identical affected set.
        let (t0, t1) = two_level_tables();
        let chain = InverseStencilChain::from([t0.clone(), t1.clone()].as_slice());
        let composed = t0.compose(&t1);
        for changed in [vec![], vec![0], vec![1], vec![2], vec![0, 2], vec![0, 1, 2]] {
            assert_eq!(
                chain.affected_outputs(&changed),
                brute_force_affected(&composed, &changed),
                "changed = {changed:?}"
            );
        }
    }

    #[test]
    fn chain_single_level_matches_single_map() {
        let t0 = sample_table();
        let chain = InverseStencilChain::from(std::slice::from_ref(&t0));
        let map = InverseStencilMap::from(&t0);
        for changed in [vec![0], vec![1], vec![2], vec![0, 1, 2]] {
            assert_eq!(
                chain.affected_outputs(&changed),
                map.affected_outputs(&changed)
            );
        }
    }

    #[test]
    fn chain_with_no_levels_is_identity() {
        let chain = InverseStencilChain::from(&[] as &[StencilTable]);
        assert_eq!(chain.affected_outputs(&[2, 0, 0]), vec![0, 2]);
        assert!(chain.affected_outputs(&[]).is_empty());
    }

    #[test]
    fn affected_outputs_into_matches_oracle_and_reuses_scratch() {
        let (t0, t1) = two_level_tables();
        let chain = InverseStencilChain::from([t0.clone(), t1.clone()].as_slice());
        let composed = t0.compose(&t1);
        let mut scratch = AffectedScratch::default();
        let mut out = Vec::new();
        // Reuse the SAME scratch across queries with different change sets; each
        // must be correct (the generation stamps must not leak between calls).
        for changed in [vec![0u32], vec![2], vec![], vec![0, 1, 2], vec![1]] {
            chain.affected_outputs_into(&changed, &mut scratch, &mut out);
            assert_eq!(
                out,
                brute_force_affected(&composed, &changed),
                "changed = {changed:?}"
            );
        }
    }

    #[test]
    fn affected_outputs_into_no_levels_is_identity_and_clears_out() {
        let chain = InverseStencilChain::from(&[] as &[StencilTable]);
        let mut scratch = AffectedScratch::default();
        let mut out = vec![999]; // must be cleared before filling
        chain.affected_outputs_into(&[2, 0, 0], &mut scratch, &mut out);
        assert_eq!(out, vec![0, 2]);
    }
}
