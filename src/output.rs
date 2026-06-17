use crate::{
    Adjacency, Interpolatable, InverseStencilChain, Mesh, Scheme, SchemeOptions, StencilTable,
};

/// Origin classification for refined vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexOrigin {
    /// Vertex descended from a coarse vertex index.
    Vertex(u32),

    /// Vertex descended from a coarse edge index.
    Edge(u32),

    /// Vertex descended from a coarse face index.
    Face(u32),
}

/// Refinement lineage maps for adapter-side propagation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineageMaps {
    /// Origin tag for each refined vertex index.
    pub vertex_origin: Vec<VertexOrigin>,

    /// Parent coarse face index for each refined face.
    pub face_parent: Vec<u32>,

    /// Parent coarse edge index for each refined edge.
    pub edge_parent: Vec<u32>,
}

/// Output of [`Refiner::refine_uniform`](crate::Refiner::refine_uniform).
///
/// Contains the refined topology, per-level stencil tables, and lineage
/// information. Use [`interpolate`](Self::interpolate) to apply
/// subdivision weights to any data buffer, or
/// [`compose_stencils`](Self::compose_stencils) to precompute a
/// single stencil table for amortized re-evaluation (animation).
///
/// # Performance model
///
/// - **One-shot**: call [`interpolate`](Self::interpolate) — chains
///   per-level stencil application. Same algorithmic cost as direct
///   subdivision. No exponential stencil growth.
/// - **Animation**: call [`compose_stencils`](Self::compose_stencils)
///   once, then [`StencilTable::interpolate`] each frame. Stencil
///   composition is O(output × entries²) but amortized over many frames.
/// - **Multiple buffers**: [`interpolate`](Self::interpolate) can be
///   called once per buffer (positions, UVs, colors, …) — all share
///   the same topology computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct RefinementResult {
    /// Refined topology (no positions).
    pub topology: Mesh,

    /// Per-level stencil tables. `level_stencils[i]` maps level-i
    /// vertices to level-(i+1) vertices. Length equals the number of
    /// refinement levels.
    pub level_stencils: Vec<StencilTable>,

    /// Ancestry tracking for adapter-side attribute propagation.
    /// Relative to the previous level (level N-1 -> N); for direct
    /// refined-face -> base-face ancestry use [`face_root`](Self::face_root).
    pub lineage: LineageMaps,

    /// Base-mesh (root) face index for each refined face -- the per-level
    /// `face_parent` chain pre-folded across all refinement levels, so an
    /// adapter can map any refined face straight to the input face it
    /// descends from (picking, per-face attribute propagation). Indexed by
    /// refined face; values index the faces of the mesh given to the
    /// [`Refiner`](crate::Refiner).
    pub face_root: Vec<u32>,

    /// Refined face selection mask (present when input had selection).
    pub selected_faces: Option<Vec<bool>>,

    /// For each input edge, the refined vertices lying along it, in order.
    ///
    /// `Some` only when the `edge_polylines` refinement option was set.
    /// Indices refer to [`topology`](Self::topology).
    pub edge_polylines: Option<Vec<Vec<u32>>>,

    /// Pre-built adjacency arrays for the refined topology.
    ///
    /// Allows adapter-side mesh construction without redundant edge
    /// discovery or adjacency analysis.
    pub adjacency: Adjacency,

    /// Scheme that produced this result. Recorded at
    /// [`refine_uniform`](crate::Refiner::refine_uniform) time so
    /// scheme-dependent post-processing
    /// ([`limit_stencils`](Self::limit_stencils)) needs no refiner
    /// handle.
    pub scheme: Scheme,

    /// Scheme options in effect during refinement (boundary and
    /// sharpness conventions for [`limit_stencils`](Self::limit_stencils)).
    pub options: SchemeOptions,
}

impl RefinementResult {
    /// Interpolate a data buffer through all refinement levels.
    ///
    /// Chains per-level stencil application: each level reads from the
    /// previous level's output and writes the next. This avoids the
    /// exponential stencil growth of [`compose_stencils`](Self::compose_stencils)
    /// and matches the performance of direct subdivision.
    ///
    /// The input buffer must have one entry per vertex in the **original**
    /// (pre-refinement) topology. The output has one entry per vertex in
    /// [`topology`](Self::topology).
    pub fn interpolate<T: Interpolatable>(&self, input: &[T]) -> Vec<T> {
        self.level_stencils
            .iter()
            .fold(input.to_vec(), |data, stencil| stencil.interpolate(&data))
    }

    /// Compose all per-level stencil tables into a single table mapping
    /// original vertices directly to final refined vertices.
    ///
    /// Use this when you need to re-evaluate the same topology with
    /// different data many times (e.g. animation with static topology).
    /// The composed table enables a single `StencilTable::interpolate`
    /// call per frame instead of chaining N levels.
    ///
    /// For one-shot subdivision, prefer [`interpolate`](Self::interpolate)
    /// which avoids the O(output × entries²) composition cost.
    /// Compose all per-level stencil tables into a single table mapping
    /// original vertices directly to final refined vertices.
    ///
    /// `input_vertex_count` must match the number of vertices in the
    /// original (pre-refinement) topology.
    pub fn compose_stencils(&self, input_vertex_count: usize) -> StencilTable {
        self.level_stencils.iter().fold(
            StencilTable::identity(input_vertex_count),
            |composed, level| composed.compose(level),
        )
    }

    /// Build the inverse stencil chain for this refinement -- the transpose of
    /// every level -- used to map changed control points to the refined output
    /// vertices they affect.
    ///
    /// The chain is topology-only, so build it once and reuse it across edits.
    /// For a single edit, [`affected_outputs`](Self::affected_outputs) is a
    /// convenience that builds and queries it in one call.
    pub fn inverse_stencil_chain(&self) -> InverseStencilChain {
        InverseStencilChain::from(self.level_stencils.as_slice())
    }

    /// Final refined output indices affected by changing the given original
    /// (pre-refinement) control-point indices, sorted ascending and deduped.
    ///
    /// `changed_inputs` are indices into the input buffer -- the same order as
    /// the vertices of the `Mesh` given to the `Refiner` and of
    /// [`interpolate`](Self::interpolate)'s input -- *not* host-mesh vertex IDs.
    /// A host that keys edits by a stable vertex ID must map those IDs to this
    /// dense input order first.
    ///
    /// Outputs *not* in this set are bit-identical under a change confined to
    /// `changed_inputs` -- this is the basis of sparse re-evaluation. This
    /// rebuilds the inverse chain on each call; for repeated edits, cache
    /// [`inverse_stencil_chain`](Self::inverse_stencil_chain) and call its
    /// `affected_outputs` directly.
    pub fn affected_outputs(&self, changed_inputs: &[u32]) -> Vec<u32> {
        self.inverse_stencil_chain()
            .affected_outputs(changed_inputs)
    }
}
