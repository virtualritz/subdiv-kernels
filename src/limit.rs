//! Analytic Catmull-Clark limit position and tangent stencils.
//!
//! [`RefinementResult::limit_stencils`] produces three [`StencilTable`]s
//! over the refined level's own vertices: row `i` reads refined vertices
//! and writes the limit position / tangents of refined vertex `i`. They
//! compose with the refinement stencils
//! ([`RefinementResult::compose_limit_stencils`]) and dispatch through
//! the existing CPU/GPU evaluation paths unchanged. The surface normal
//! is `tangent1 x tangent2`, oriented with the face winding.
//!
//! The mask weights are an exact port of OpenSubdiv's
//! `Sdc::Scheme<SCHEME_CATMARK>::assign{Corner,Crease,Smooth}Limit{Mask,
//! TangentMasks}` (`opensubdiv/sdc/catmarkScheme.h`), with face weights
//! applied to the diagonal (opposite) vertex of each incident quad
//! (`SetFaceWeightsForFaceCenters(false)`), which is why the level must
//! have been refined at least once: every Catmull-Clark face is then a
//! quad with a well-defined diagonal.
//!
//! # Rule selection (mirrors OpenSubdiv `Far`)
//!
//! OpenSubdiv selects limit masks from the subdivision rule *at the
//! evaluated level* (`far/primvarRefiner.h` ->
//! `Vtr::Level::getVertexRule` -> `Sdc::Crease::IsSharp`, i.e.
//! `sharpness > 0`). Mirrored here on the refined level's stored
//! crease/corner values:
//!
//! - An edge is sharp when its refined-level crease value is `> 0.0` or
//!   it is a boundary edge (OpenSubdiv sharpens boundary edges
//!   unconditionally; this crate's refiner applies the crease rule along
//!   boundaries under `EdgesOnly`/`EdgesAndCorners`).
//! - A vertex takes the corner masks when its refined-level corner
//!   value is `> 0.0`, when more than two incident edges are sharp, or
//!   -- under [`BoundaryInterpolation::EdgesAndCorners`] -- when it is a
//!   boundary vertex with a single incident face (exactly the vertices
//!   the refiner pins).
//! - Exactly two sharp edges select the crease masks; zero or one
//!   (dart) the smooth masks.
//!
//! Residual *finite* sharpness at the refined level therefore selects
//! the sharp masks, exactly as OpenSubdiv far does; once refinement has
//! decayed a semi-sharp value to zero, deeper results select the smooth
//! masks. "Infinitely sharp" is whatever never decays under the
//! refiner's configured propagation: `f32::INFINITY` always, the `10.0`
//! sentinel under [`CornerRule::OpenSubdivDeRose`], and every positive
//! value under `crease_normalize`/`corner_normalize`.
//!
//! One deliberate approximation against this crate's non-OpenSubdiv
//! conventions, OpenSubdiv-exact otherwise:
//!
//! - [`BoundaryInterpolation::Natural`] has no closed-form limit mask
//!   for boundary vertices (the refiner's natural rule averages a
//!   partial neighborhood); open meshes are rejected under it.
//!
//! # Sectored tangents ([`RefinementResult::sectored_limit_stencils`])
//!
//! At an infinitely sharp crease the limit surface has one normal per
//! *sector* -- the maximal contiguous fan of incident faces between
//! consecutive sharp edges around a vertex (sharp per the rule
//! selection above) -- so the single per-vertex tangent pair shades one
//! side's normal onto every adjacent face. [`SectoredLimitStencils`]
//! generalizes the tangent output to one row per sector:
//!
//! - Smooth and dart vertices keep one sector (the full ring, the
//!   per-vertex smooth masks); all their corners share it.
//! - A crease vertex has one sector per side: the counterclockwise span
//!   between its two sharp edges and the complementary span, each
//!   evaluated with the crease tangent masks ring-oriented so the
//!   evaluated span *is* that sector. A boundary crease vertex has one
//!   side. `tangent1` runs along each sector's own leading edge.
//! - A corner-rule vertex has one sector per fan between consecutive
//!   sharp edges, position pinned, tangents from the crease tangent
//!   masks of the fan's bounding edges. On a single-face fan that
//!   normal is exactly OpenSubdiv's (the corner-mask and crease-mask
//!   tangent pairs span the same plane, oracle-verified); on a
//!   multi-face fan the pinned limit is a cone point with no
//!   convergent normal -- OpenSubdiv's own patch evaluation reports
//!   per-face normals there that drift with the adaptive isolation
//!   level (probed in `tests/limit_osd_oracle.rs`) -- and the crease
//!   machinery is kept as a deterministic, fan-symmetric tangent plane
//!   (the naive bounding-edge cross product would degenerate on
//!   straight-through fans). A vertex pinned by vertex sharpness alone
//!   (fewer than two sharp edges) keeps the whole ring as one sector
//!   with the naive corner tangents, OpenSubdiv-far parity.
//!
//! Sector rows are deterministic: emitted vertex by vertex, ordered
//! within a vertex by the ring slot of each sector's leading edge.
//! [`SectoredLimitStencils::corner_sector`] maps every refined
//! face-corner to its row.
//!
//! [`CornerRule::OpenSubdivDeRose`]: crate::CornerRule::OpenSubdivDeRose

use std::f64::consts::PI;

use crate::catmull_clark::stencils::{Sparse, merge, pack};
use crate::{
    Adjacency, BoundaryInterpolation, KernelError, Mesh, RefinementResult, Scheme, SchemeOptions,
    StencilTable,
};

/// Limit-surface stencils over a refined level's vertices.
///
/// All three tables read the refined level's vertex data (or, when
/// produced by [`RefinementResult::compose_limit_stencils`], the
/// original control points) and write one value per refined vertex.
/// Tangents are unnormalized; `tangent1 x tangent2` is the (winding
/// oriented) surface normal and the tangent magnitudes match
/// OpenSubdiv's limit-mask scales.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitStencils {
    /// Refined-vertex -> limit position.
    pub position: StencilTable,
    /// Refined-vertex -> limit tangent 1 (unnormalized).
    pub tangent1: StencilTable,
    /// Refined-vertex -> limit tangent 2 (unnormalized).
    pub tangent2: StencilTable,
}

/// Per-sector limit-surface stencils over a refined level's vertices.
///
/// The position table matches [`LimitStencils`]'s row for row; the
/// tangent tables have one row per *sector* (see the module docs), so
/// each side of an infinitely sharp crease shades with its own normal.
/// Tangents are unnormalized and `tangent1 x tangent2` is the (winding
/// oriented) sector normal.
#[derive(Debug, Clone, PartialEq)]
pub struct SectoredLimitStencils {
    /// Refined-vertex -> limit position (same as
    /// [`LimitStencils::position`]).
    pub position: StencilTable,
    /// Per-sector limit tangents: row `s` is sector `s`'s tangent.
    pub tangent1: StencilTable,
    /// Per-sector limit tangents: row `s` is sector `s`'s tangent.
    pub tangent2: StencilTable,
    /// For each refined face-corner, in the CSR order of
    /// [`Mesh::face_vertex_indices`], the sector row its
    /// vertex's limit tangents live in. Smooth vertices have one sector
    /// shared by all their corners; a crease vertex has two; a
    /// corner-rule vertex one per fan between consecutive sharp edges.
    pub corner_sector: Vec<u32>,
}

impl RefinementResult {
    /// Limit masks over the refined level's own vertices (row `i` reads
    /// refined vertices, writes limit data for refined vertex `i`).
    ///
    /// Catmull-Clark only, and the result must come from at least one
    /// full (unselected) refinement -- see the module docs for the rule
    /// conventions and restrictions.
    pub fn limit_stencils(&self) -> Result<LimitStencils, KernelError> {
        build_limit_stencils(self)
    }

    /// The cage -> limit composition: each [`LimitStencils`] table
    /// composed onto [`compose_stencils`](Self::compose_stencils), so a
    /// host can upload three GPU tables and evaluate limit position and
    /// tangents straight from control points. The surface normal is
    /// `tangent1 x tangent2`.
    ///
    /// `input_vertex_count` must match the number of vertices in the
    /// original (pre-refinement) topology.
    pub fn compose_limit_stencils(
        &self,
        input_vertex_count: usize,
    ) -> Result<LimitStencils, KernelError> {
        let limit = self.limit_stencils()?;
        let cage_to_refined = self.compose_stencils(input_vertex_count);
        Ok(LimitStencils {
            position: cage_to_refined.compose(&limit.position),
            tangent1: cage_to_refined.compose(&limit.tangent1),
            tangent2: cage_to_refined.compose(&limit.tangent2),
        })
    }

    /// Per-sector limit masks over the refined level's own vertices:
    /// the position table is per refined vertex, the tangent tables per
    /// sector, with [`SectoredLimitStencils::corner_sector`] mapping
    /// each refined face-corner to its tangent row.
    ///
    /// Same scheme/option gating and errors as
    /// [`limit_stencils`](Self::limit_stencils).
    pub fn sectored_limit_stencils(&self) -> Result<SectoredLimitStencils, KernelError> {
        build_sectored_limit_stencils(self)
    }

    /// The cage -> limit composition of
    /// [`sectored_limit_stencils`](Self::sectored_limit_stencils): all
    /// three tables composed onto
    /// [`compose_stencils`](Self::compose_stencils), so a host uploads
    /// the tables and evaluates per-sector limit tangents straight from
    /// control points.
    ///
    /// `input_vertex_count` must match the number of vertices in the
    /// original (pre-refinement) topology.
    pub fn compose_sectored_limit_stencils(
        &self,
        input_vertex_count: usize,
    ) -> Result<SectoredLimitStencils, KernelError> {
        let sectored = self.sectored_limit_stencils()?;
        let cage_to_refined = self.compose_stencils(input_vertex_count);
        Ok(SectoredLimitStencils {
            position: cage_to_refined.compose(&sectored.position),
            tangent1: cage_to_refined.compose(&sectored.tangent1),
            tangent2: cage_to_refined.compose(&sectored.tangent2),
            corner_sector: sectored.corner_sector,
        })
    }
}

/// Counterclockwise one-ring of a refined vertex.
///
/// Slot `i` holds ring edge `e_i`; the quad between `e_i` and `e_(i+1)`
/// contributes its diagonal vertex as face slot `i`. Interior rings are
/// cyclic (`edges == faces`); boundary rings run from the leading
/// boundary edge around the surface fan to the trailing one
/// (`edges == faces + 1`), the orientation OpenSubdiv's crease tangent
/// masks assume (tangent1 along the leading edge, `tan1 x tan2`
/// outward).
pub(crate) struct Ring {
    /// Neighbor vertex at the far end of each ring edge.
    pub(crate) neighbors: Vec<u32>,
    /// Mesh edge index per ring slot, parallel to `neighbors`.
    pub(crate) edges: Vec<u32>,
    /// Diagonal vertex of the quad between ring edges `i` and `i + 1`.
    pub(crate) diagonals: Vec<u32>,
    /// Face index of the quad between ring edges `i` and `i + 1`,
    /// parallel to `diagonals`.
    pub(crate) faces: Vec<u32>,
    /// Boundary fan vs interior cycle.
    pub(crate) boundary: bool,
}

impl Ring {
    /// The ring re-origined so slot `by` becomes slot 0. Interior rings
    /// only -- a boundary fan's origin is its leading boundary edge.
    pub(crate) fn rotated(&self, by: usize) -> Self {
        debug_assert!(!self.boundary, "boundary fans cannot be rotated");
        let shift = |ring: &[u32]| -> Vec<u32> {
            (0..ring.len())
                .map(|i| ring[(i + by) % ring.len()])
                .collect()
        };
        Self {
            neighbors: shift(&self.neighbors),
            edges: shift(&self.edges),
            diagonals: shift(&self.diagonals),
            faces: shift(&self.faces),
            boundary: false,
        }
    }
}

/// Limit rule for one vertex, per the module-doc selection.
enum LimitRule {
    Smooth,
    /// Ring slots of the two sharp edges (leading, trailing).
    Crease([usize; 2]),
    /// Ring slots of all sharp edges (may be fewer than two when the
    /// vertex is pinned by vertex sharpness or boundary policy alone).
    Corner(Vec<usize>),
}

/// Per-incident-face corner data feeding the ring walk: within the face,
/// `out_edge` runs from the vertex to its next corner, `in_edge` to its
/// previous corner, and `diagonal` is the opposite quad corner. With
/// consistent winding, sweeping from `out_edge` across the face interior
/// to `in_edge` is the counterclockwise step around the vertex.
struct FanStep {
    out_edge: u32,
    in_edge: u32,
    diagonal: u32,
    face: u32,
    used: bool,
}

/// The shared gating of refined-level surface machinery (limit masks,
/// regular-patch extraction): Catmull-Clark, full (unselected)
/// refinement, and an all-quad refined level.
pub(crate) fn validate_refined_quads(result: &RefinementResult) -> Result<(), KernelError> {
    if result.scheme != Scheme::CatmullClark {
        return Err(KernelError::NotImplemented(
            "limit-surface machinery is only implemented for Catmull-Clark",
        ));
    }
    // `selected_faces` is an all-true mask for unselected refinements;
    // only an actually partial mask leaves unrefined n-gons behind.
    if result
        .selected_faces
        .as_ref()
        .is_some_and(|mask| mask.iter().any(|&selected| !selected))
    {
        return Err(KernelError::NotImplemented(
            "limit-surface machinery is not defined for partially selected refinements",
        ));
    }
    if result.topology.face_vertex_counts.iter().any(|&c| c != 4) {
        return Err(KernelError::NotImplemented(
            "limit-surface machinery needs an all-quad refined level (refine the cage at least \
             once)",
        ));
    }
    Ok(())
}

/// The gating of both limit builders: [`validate_refined_quads`] plus
/// no open mesh under the natural boundary rule (which has no
/// closed-form limit mask for boundary vertices).
fn validate_limit_topology(result: &RefinementResult) -> Result<(), KernelError> {
    validate_refined_quads(result)?;
    if result.options.boundary_interpolation == BoundaryInterpolation::Natural
        && result.adjacency.edge_is_boundary.iter().any(|&b| b)
    {
        return Err(KernelError::NotImplemented(
            "limit stencils for BoundaryInterpolation::Natural on open meshes",
        ));
    }
    Ok(())
}

fn build_limit_stencils(result: &RefinementResult) -> Result<LimitStencils, KernelError> {
    validate_limit_topology(result)?;
    let mesh = &result.topology;
    let adjacency = &result.adjacency;

    let vertex_count = mesh.vertex_count as usize;
    let mut position_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);
    let mut tangent1_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);
    let mut tangent2_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);

    for vi in 0..vertex_count {
        let ring = vertex_ring(vi, mesh, adjacency)?;
        let masks = match classify(vi, &ring, mesh, adjacency, &result.options) {
            LimitRule::Smooth => smooth_masks(vi as u32, &ring),
            LimitRule::Crease(ends) => crease_masks(vi as u32, &ring, ends),
            LimitRule::Corner(_) => corner_masks(vi as u32, &ring),
        };
        debug_assert!(
            (masks.position.iter().map(|&(_, w)| w).sum::<f32>() - 1.0).abs() < 1e-4,
            "limit position mask of vertex {vi} is not affine",
        );
        debug_assert!(
            masks.tangent1.iter().map(|&(_, w)| w).sum::<f32>().abs() < 1e-4
                && masks.tangent2.iter().map(|&(_, w)| w).sum::<f32>().abs() < 1e-4,
            "limit tangent masks of vertex {vi} are not derivations",
        );
        position_rows.push(masks.position);
        tangent1_rows.push(masks.tangent1);
        tangent2_rows.push(masks.tangent2);
    }

    Ok(LimitStencils {
        position: pack(&position_rows),
        tangent1: pack(&tangent1_rows),
        tangent2: pack(&tangent2_rows),
    })
}

fn build_sectored_limit_stencils(
    result: &RefinementResult,
) -> Result<SectoredLimitStencils, KernelError> {
    validate_limit_topology(result)?;
    let mesh = &result.topology;
    let adjacency = &result.adjacency;

    let vertex_count = mesh.vertex_count as usize;
    let mut position_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);
    let mut tangent1_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);
    let mut tangent2_rows: Vec<Sparse> = Vec::with_capacity(vertex_count);
    let mut corner_sector = vec![u32::MAX; mesh.face_vertex_indices.len()];

    for vi in 0..vertex_count {
        let ring = vertex_ring(vi, mesh, adjacency)?;
        let rule = classify(vi, &ring, mesh, adjacency, &result.options);
        let sectors = vertex_sectors(vi as u32, &ring, rule);
        debug_assert!(
            (sectors.position.iter().map(|&(_, w)| w).sum::<f32>() - 1.0).abs() < 1e-4,
            "limit position mask of vertex {vi} is not affine",
        );
        debug_assert!(
            sectors.tangents.iter().all(|(tangent1, tangent2)| {
                tangent1.iter().map(|&(_, w)| w).sum::<f32>().abs() < 1e-4
                    && tangent2.iter().map(|&(_, w)| w).sum::<f32>().abs() < 1e-4
            }),
            "a sector tangent mask of vertex {vi} is not a derivation",
        );

        // Scatter this vertex's fan slots into the per-corner map; each
        // face corner belongs to exactly one vertex ring.
        let first_row = tangent1_rows.len() as u32;
        for (fan_slot, &fi) in ring.faces.iter().enumerate() {
            let off = (fi * 4) as usize;
            let corner = mesh.face_vertex_indices[off..off + 4]
                .iter()
                .position(|&c| c == vi as u32)
                .ok_or(KernelError::InvalidTopology(
                    "vertex-face adjacency references a face without that vertex",
                ))?;
            corner_sector[off + corner] = first_row + sectors.fan_sector[fan_slot];
        }

        position_rows.push(sectors.position);
        for (tangent1, tangent2) in sectors.tangents {
            tangent1_rows.push(tangent1);
            tangent2_rows.push(tangent2);
        }
    }
    debug_assert!(
        corner_sector.iter().all(|&row| row != u32::MAX),
        "a refined face-corner was not covered by any vertex ring",
    );

    Ok(SectoredLimitStencils {
        position: pack(&position_rows),
        tangent1: pack(&tangent1_rows),
        tangent2: pack(&tangent2_rows),
        corner_sector,
    })
}

/// Sector decomposition of one vertex: the shared position mask, one
/// tangent-mask pair per sector (ordered by leading ring slot), and the
/// local sector index per ring fan slot (the face between ring edges
/// `i` and `i + 1`).
struct VertexSectors {
    position: Sparse,
    tangents: Vec<(Sparse, Sparse)>,
    fan_sector: Vec<u32>,
}

impl VertexSectors {
    /// The whole ring as one sector.
    fn single(masks: LimitMasks, fan_count: usize) -> Self {
        Self {
            position: masks.position,
            tangents: vec![(masks.tangent1, masks.tangent2)],
            fan_sector: vec![0; fan_count],
        }
    }
}

/// Split one vertex's ring into sectors and evaluate per-sector masks;
/// see the module docs for the per-rule decomposition.
fn vertex_sectors(vi: u32, ring: &Ring, rule: LimitRule) -> VertexSectors {
    let fan_count = ring.faces.len();
    match rule {
        LimitRule::Smooth => VertexSectors::single(smooth_masks(vi, ring), fan_count),
        // A boundary crease vertex has surface on one side only.
        LimitRule::Crease(ends) if ring.boundary => {
            VertexSectors::single(crease_masks(vi, ring, ends), fan_count)
        }
        // An interior crease vertex has one sector per side: the
        // counterclockwise span between the sharp edges and its
        // complement, the latter evaluated on the re-origined ring so
        // the crease masks see a forward span.
        LimitRule::Crease([lead, trail]) => {
            let near = crease_masks(vi, ring, [lead, trail]);
            let far = crease_masks(vi, &ring.rotated(trail), [0, fan_count - (trail - lead)]);
            VertexSectors {
                position: near.position,
                tangents: vec![(near.tangent1, near.tangent2), (far.tangent1, far.tangent2)],
                fan_sector: (0..fan_count)
                    .map(|slot| u32::from(!(lead..trail).contains(&slot)))
                    .collect(),
            }
        }
        // Pinned with fewer than two sharp edges: no fan is bounded, so
        // the whole ring stays one sector with OpenSubdiv far's naive
        // corner tangents.
        LimitRule::Corner(sharp_slots) if sharp_slots.len() < 2 => {
            VertexSectors::single(corner_masks(vi, ring), fan_count)
        }
        // A pinned vertex has one sector per fan between consecutive
        // sharp edges; the position interpolates the vertex and each
        // fan's tangents come from the crease tangent masks of its
        // bounding edges (see the module docs for why not the naive
        // bounding-edge differences).
        LimitRule::Corner(sharp_slots) => {
            let wrap = (!ring.boundary).then(|| {
                let first = sharp_slots[0];
                // SAFETY-free indexing: this arm has at least two slots.
                let last = sharp_slots[sharp_slots.len() - 1];
                (last, first + ring.edges.len())
            });
            let spans: Vec<(usize, usize)> = sharp_slots
                .windows(2)
                .map(|pair| (pair[0], pair[1]))
                .chain(wrap)
                .collect();
            let tangents = spans
                .iter()
                .map(|&(lead, trail)| {
                    let masks = if ring.boundary {
                        crease_masks(vi, ring, [lead, trail])
                    } else {
                        crease_masks(vi, &ring.rotated(lead), [0, trail - lead])
                    };
                    (masks.tangent1, masks.tangent2)
                })
                .collect();
            let fan_sector = (0..fan_count)
                .map(|slot| {
                    // Fan slot `slot` belongs to the sector led by the
                    // last sharp slot at or before it; earlier slots
                    // wrap into the trailing sector.
                    let led_by = sharp_slots.partition_point(|&sharp| sharp <= slot);
                    (if led_by == 0 {
                        spans.len() - 1
                    } else {
                        led_by - 1
                    }) as u32
                })
                .collect();
            VertexSectors {
                position: vec![(vi, 1.0)],
                tangents,
                fan_sector,
            }
        }
    }
}

/// Derivative payload of [`corner_limit_sector`].
pub(crate) enum SectorDerivatives {
    /// Parametric one-sided limit derivatives along the corner's two
    /// quad edges, with respect to the quad's own unit
    /// parameterization: `d_out` along the out-edge (corner -> next
    /// CSR corner), `d_in` along the in-edge (corner -> previous CSR
    /// corner). Available at crease-rule corners whose quad edges are
    /// each a sharp sector bound (the crease-curve derivative,
    /// `(lead - trail) / 2`) or the single interior edge of a regular
    /// two-face sector (the crease patch's cross derivative -- the
    /// `tangent2` mask is exactly `limit(row 1) - limit(row 0)` of the
    /// patch's control rows).
    Parametric { d_out: Sparse, d_in: Sparse },
    /// In-sector tangent pair only (smooth, dart, pinned, or irregular
    /// sector): `tangent1 x tangent2` is the winding-oriented sector
    /// normal, but the pair has no quad-parametric alignment or scale.
    Plane { tangent1: Sparse, tangent2: Sparse },
}

/// Limit position and in-sector derivative masks of one face-corner,
/// evaluated directly from its ring -- the row [`SectoredLimitStencils`]
/// would assign that corner, without building whole-mesh tables (the
/// feature-isolation submeshes of `limit_eval` have artificially
/// truncated border vertices whose rings must never be walked).
///
/// The corner's vertex must have a complete fan. Boundary vertices
/// under [`BoundaryInterpolation::Natural`] are rejected (no
/// closed-form limit mask; see the module docs).
pub(crate) fn corner_limit_sector(
    face: u32,
    corner: usize,
    mesh: &Mesh,
    adjacency: &Adjacency,
    options: &SchemeOptions,
) -> Result<(Sparse, SectorDerivatives), KernelError> {
    let off = (face * 4) as usize;
    let vi = mesh.face_vertex_indices[off + corner];
    let ring = vertex_ring(vi as usize, mesh, adjacency)?;
    if ring.boundary && options.boundary_interpolation == BoundaryInterpolation::Natural {
        return Err(KernelError::NotImplemented(
            "limit evaluation at boundary feature vertices under BoundaryInterpolation::Natural",
        ));
    }
    let rule = classify(vi as usize, &ring, mesh, adjacency, options);
    let crease_ends = match &rule {
        LimitRule::Crease(ends) => Some(*ends),
        _ => None,
    };
    let fan_slot =
        ring.faces
            .iter()
            .position(|&f| f == face)
            .ok_or(KernelError::InvalidTopology(
                "corner vertex ring does not contain the corner's face",
            ))?;
    let mut sectors = vertex_sectors(vi, &ring, rule);
    let row = sectors.fan_sector[fan_slot] as usize;
    let (tangent1, tangent2) = sectors.tangents.swap_remove(row);

    // The quad's sector span in ring slots (interior row 0 = the span
    // between the sharp edges, row 1 = its complement; a boundary
    // crease vertex has the one surface-side span).
    let span = crease_ends.map(|[lead, trail]| {
        if ring.boundary || row == 0 {
            (lead, trail - lead)
        } else {
            (trail, ring.edges.len() - (trail - lead))
        }
    });
    // Parametric derivative along ring edge `slot`, when the slot is a
    // sector bound (+-tangent1 along the crease curve) or the single
    // interior edge of a regular sector (+tangent2, the cross
    // derivative); `None` leaves the corner plane-only.
    let edge_derivative = |slot: usize| -> Option<Sparse> {
        let (lead, len) = span?;
        let rel = if ring.boundary {
            slot.checked_sub(lead)?
        } else {
            (slot + ring.edges.len() - lead) % ring.edges.len()
        };
        if rel == 0 {
            Some(tangent1.clone())
        } else if rel == len {
            Some(tangent1.iter().map(|&(i, w)| (i, -w)).collect())
        } else if len == 2 && rel == 1 {
            Some(tangent2.clone())
        } else {
            None
        }
    };
    let slot_of = |edge: u32| ring.edges.iter().position(|&e| e == edge);
    let out_slot = slot_of(adjacency.face_edges[off + corner]);
    let in_slot = slot_of(adjacency.face_edges[off + (corner + 3) % 4]);
    let derivatives = match (
        out_slot.and_then(&edge_derivative),
        in_slot.and_then(&edge_derivative),
    ) {
        (Some(d_out), Some(d_in)) => SectorDerivatives::Parametric { d_out, d_in },
        _ => SectorDerivatives::Plane { tangent1, tangent2 },
    };
    Ok((sectors.position, derivatives))
}

/// Walk the oriented fan around `vi` from the per-face corner data.
pub(crate) fn vertex_ring(
    vi: usize,
    mesh: &Mesh,
    adjacency: &Adjacency,
) -> Result<Ring, KernelError> {
    let face_start = adjacency.vert_face_offsets[vi] as usize;
    let face_end = adjacency.vert_face_offsets[vi + 1] as usize;
    let incident_faces = &adjacency.vert_faces[face_start..face_end];
    if incident_faces.is_empty() {
        return Err(KernelError::InvalidTopology(
            "vertex without incident faces has no limit ring",
        ));
    }

    let mut steps = incident_faces
        .iter()
        .map(|&fi| {
            // All faces are quads (validated by the caller), so corner
            // offsets are 4 * face.
            let off = (fi * 4) as usize;
            let corners = &mesh.face_vertex_indices[off..off + 4];
            corners
                .iter()
                .position(|&c| c == vi as u32)
                .map(|j| FanStep {
                    out_edge: adjacency.face_edges[off + j],
                    in_edge: adjacency.face_edges[off + (j + 3) % 4],
                    diagonal: corners[(j + 2) % 4],
                    face: fi,
                    used: false,
                })
                .ok_or(KernelError::InvalidTopology(
                    "vertex-face adjacency references a face without that vertex",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let boundary = adjacency.vertex_is_boundary[vi];
    let start = if boundary {
        // The leading boundary edge is the one face-outgoing boundary
        // edge; the walk then ends on the trailing boundary edge.
        steps
            .iter()
            .find(|s| adjacency.edge_is_boundary[s.out_edge as usize])
            .map(|s| s.out_edge)
            .ok_or(KernelError::InvalidTopology(
                "boundary vertex has no leading boundary edge (inconsistent face winding)",
            ))?
    } else {
        steps[0].out_edge
    };

    let face_count = steps.len();
    let mut edges = Vec::with_capacity(face_count + 1);
    let mut diagonals = Vec::with_capacity(face_count);
    let mut faces = Vec::with_capacity(face_count);
    let mut current = start;
    edges.push(current);
    for k in 0..face_count {
        let step = steps
            .iter_mut()
            .find(|s| !s.used && s.out_edge == current)
            .ok_or(KernelError::InvalidTopology(
                "vertex ring is not a single oriented fan",
            ))?;
        step.used = true;
        diagonals.push(step.diagonal);
        faces.push(step.face);
        current = step.in_edge;
        if boundary || k + 1 < face_count {
            edges.push(current);
        }
    }
    if !boundary && current != start {
        return Err(KernelError::InvalidTopology(
            "interior vertex ring does not close",
        ));
    }
    if boundary && !adjacency.edge_is_boundary[current as usize] {
        return Err(KernelError::InvalidTopology(
            "boundary vertex ring does not end on a boundary edge",
        ));
    }
    let incident_edge_count =
        (adjacency.vert_edge_offsets[vi + 1] - adjacency.vert_edge_offsets[vi]) as usize;
    if edges.len() != incident_edge_count {
        return Err(KernelError::InvalidTopology(
            "vertex ring does not cover all incident edges (non-manifold fan)",
        ));
    }

    let neighbors = edges
        .iter()
        .map(|&ei| {
            let [a, b] = mesh.edge_vertices[ei as usize];
            if a as usize == vi { b } else { a }
        })
        .collect();

    Ok(Ring {
        neighbors,
        edges,
        diagonals,
        faces,
        boundary,
    })
}

/// Select the limit rule for one vertex; see the module docs.
fn classify(
    vi: usize,
    ring: &Ring,
    mesh: &Mesh,
    adjacency: &Adjacency,
    options: &SchemeOptions,
) -> LimitRule {
    let sharp_slots: Vec<usize> = ring
        .edges
        .iter()
        .enumerate()
        .filter(|&(_, &ei)| {
            mesh.edge_creases[ei as usize] > 0.0 || adjacency.edge_is_boundary[ei as usize]
        })
        .map(|(slot, _)| slot)
        .collect();

    // Sharp vertex tag (Sdc::Crease::IsSharp on the refined value).
    if mesh.vertex_corners[vi] > 0.0 {
        return LimitRule::Corner(sharp_slots);
    }
    // EdgesAndCorners pins single-face boundary vertices, mirroring the
    // refiner's vertex_point_stencil.
    if ring.boundary
        && options.boundary_interpolation == BoundaryInterpolation::EdgesAndCorners
        && ring.diagonals.len() <= 1
    {
        return LimitRule::Corner(sharp_slots);
    }

    match sharp_slots.as_slice() {
        // Smooth and dart vertices share the smooth masks.
        [] | [_] => LimitRule::Smooth,
        [lead, trail] => LimitRule::Crease([*lead, *trail]),
        _ => LimitRule::Corner(sharp_slots),
    }
}

/// The three sparse limit masks of one vertex.
struct LimitMasks {
    position: Sparse,
    tangent1: Sparse,
    tangent2: Sparse,
}

/// Accumulate one weight, dropping exact zeros (OpenSubdiv masks carry
/// explicit zero entries; CSR rows need not).
fn push(row: &mut Sparse, index: u32, weight: f64) {
    if weight != 0.0 {
        merge(row, &[(index, weight as f32)], 1.0);
    }
}

/// `assignCornerLimitMask` + `assignCornerLimitTangentMasks`: the limit
/// interpolates the vertex; tangents run along the first two ring edges.
fn corner_masks(vi: u32, ring: &Ring) -> LimitMasks {
    let mut tangent1 = Sparse::new();
    push(&mut tangent1, vi, -1.0);
    push(&mut tangent1, ring.neighbors[0], 1.0);

    let mut tangent2 = Sparse::new();
    push(&mut tangent2, vi, -1.0);
    push(&mut tangent2, ring.neighbors[1], 1.0);

    LimitMasks {
        position: vec![(vi, 1.0)],
        tangent1,
        tangent2,
    }
}

/// `assignCreaseLimitMask` + `assignCreaseLimitTangentMasks` with
/// `creaseEnds = ends`: the B-spline crease-curve limit along the two
/// sharp edges, and the cross tangent over the counterclockwise fan
/// between them (regular B-spline, Biermann et al. irregular, or the
/// single-face average).
fn crease_masks(vi: u32, ring: &Ring, ends: [usize; 2]) -> LimitMasks {
    let [lead, trail] = ends;

    let mut position = Sparse::new();
    push(&mut position, vi, 2.0 / 3.0);
    push(&mut position, ring.neighbors[lead], 1.0 / 6.0);
    push(&mut position, ring.neighbors[trail], 1.0 / 6.0);

    // Tangent along the crease, oriented with the leading edge.
    let mut tangent1 = Sparse::new();
    push(&mut tangent1, ring.neighbors[lead], 0.5);
    push(&mut tangent1, ring.neighbors[trail], -0.5);

    // Cross tangent over the span; face slot f sits between ring edges
    // f and f + 1, so the span's faces are slots lead .. trail - 1.
    let mut tangent2 = Sparse::new();
    let interior_edge_count = trail - lead - 1;
    if interior_edge_count == 1 {
        // The regular case: uniform B-spline cross-tangent.
        push(&mut tangent2, vi, -4.0 / 6.0);
        push(&mut tangent2, ring.neighbors[lead], -1.0 / 6.0);
        push(&mut tangent2, ring.neighbors[lead + 1], 4.0 / 6.0);
        push(&mut tangent2, ring.neighbors[trail], -1.0 / 6.0);
        push(&mut tangent2, ring.diagonals[lead], 1.0 / 6.0);
        push(&mut tangent2, ring.diagonals[lead + 1], 1.0 / 6.0);
    } else if interior_edge_count > 1 {
        // The irregular case: formulae from Biermann et al.
        let k = (interior_edge_count + 1) as f64;
        let theta = PI / k;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let common_denom = 1.0 / (k * (3.0 + cos_theta));
        let r = (cos_theta + 1.0) / sin_theta;

        push(
            &mut tangent2,
            vi,
            4.0 * r * (cos_theta - 1.0) * common_denom,
        );
        let crease_weight = -r * (1.0 + 2.0 * cos_theta) * common_denom;
        push(&mut tangent2, ring.neighbors[lead], crease_weight);
        push(&mut tangent2, ring.neighbors[trail], crease_weight);
        push(
            &mut tangent2,
            ring.diagonals[lead],
            sin_theta * common_denom,
        );
        for i in 1..interior_edge_count + 1 {
            let sin_theta_i = (i as f64 * theta).sin();
            let sin_theta_i_plus_1 = ((i + 1) as f64 * theta).sin();
            push(
                &mut tangent2,
                ring.neighbors[lead + i],
                4.0 * sin_theta_i * common_denom,
            );
            push(
                &mut tangent2,
                ring.diagonals[lead + i],
                (sin_theta_i + sin_theta_i_plus_1) * common_denom,
            );
        }
    } else {
        // The single-face special case: simple average of the boundary
        // edges.
        push(&mut tangent2, vi, -6.0);
        push(&mut tangent2, ring.neighbors[lead], 3.0);
        push(&mut tangent2, ring.neighbors[trail], 3.0);
    }

    LimitMasks {
        position,
        tangent1,
        tangent2,
    }
}

/// `assignSmoothLimitMask` + `assignSmoothLimitTangentMasks` for an
/// interior vertex of face valence `n`; valence 2 falls back to the
/// corner masks, valence 4 uses the regular specializations.
fn smooth_masks(vi: u32, ring: &Ring) -> LimitMasks {
    let valence = ring.diagonals.len();
    if valence == 2 {
        return corner_masks(vi, ring);
    }

    let mut position = Sparse::new();
    let mut tangent1 = Sparse::new();
    let mut tangent2 = Sparse::new();

    if valence == 4 {
        push(&mut position, vi, 4.0 / 9.0);
        let tan1_edge = [4.0, 0.0, -4.0, 0.0];
        let tan1_face = [1.0, -1.0, -1.0, 1.0];
        let tan2_edge = [0.0, 4.0, 0.0, -4.0];
        let tan2_face = [1.0, 1.0, -1.0, -1.0];
        for i in 0..4 {
            push(&mut position, ring.neighbors[i], 1.0 / 9.0);
            push(&mut position, ring.diagonals[i], 1.0 / 36.0);
            push(&mut tangent1, ring.neighbors[i], tan1_edge[i]);
            push(&mut tangent1, ring.diagonals[i], tan1_face[i]);
            push(&mut tangent2, ring.neighbors[i], tan2_edge[i]);
            push(&mut tangent2, ring.diagonals[i], tan2_face[i]);
        }
    } else {
        // Position weights in f32 like the C++ general case.
        let n = valence as f32;
        let face_weight = 1.0 / (n * (n + 5.0));
        let edge_weight = 4.0 * face_weight;
        push(
            &mut position,
            vi,
            (1.0 - n * (edge_weight + face_weight)) as f64,
        );

        // Tangent weights in f64 like the C++ (double trig, float cast).
        let theta = 2.0 * PI / valence as f64;
        let cos_theta = theta.cos();
        let cos_half_theta = (theta * 0.5).cos();
        let lambda = (5.0 / 16.0)
            + (1.0 / 16.0) * (cos_theta + cos_half_theta * (2.0 * (9.0 + cos_theta)).sqrt());
        let face_weight_scale = 1.0 / (4.0 * lambda - 1.0);

        // tangent2 is tangent1 rotated by one ring slot.
        let rotated = |i: usize| (i + valence - 1) % valence;
        for i in 0..valence {
            push(&mut position, ring.neighbors[i], edge_weight as f64);
            push(&mut position, ring.diagonals[i], face_weight as f64);

            let cos_theta_i = (i as f64 * theta).cos();
            let cos_theta_i_plus_1 = ((i + 1) as f64 * theta).cos();
            push(&mut tangent1, ring.neighbors[i], 4.0 * cos_theta_i);
            push(
                &mut tangent1,
                ring.diagonals[i],
                face_weight_scale * (cos_theta_i + cos_theta_i_plus_1),
            );

            let j = rotated(i);
            let cos_theta_j = (j as f64 * theta).cos();
            let cos_theta_j_plus_1 = ((j + 1) as f64 * theta).cos();
            push(&mut tangent2, ring.neighbors[i], 4.0 * cos_theta_j);
            push(
                &mut tangent2,
                ring.diagonals[i],
                face_weight_scale * (cos_theta_j + cos_theta_j_plus_1),
            );
        }
    }

    LimitMasks {
        position,
        tangent1,
        tangent2,
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU8;

    use crate::{KernelError, Mesh, Refiner, Scheme, SchemeOptions, UniformRefine};

    /// A 2x2 quad grid (the mask gates live in `tests/limit_surface.rs`
    /// and `tests/limit_osd_oracle.rs`; these unit tests cover the
    /// error paths only).
    fn grid() -> Mesh {
        Mesh {
            vertex_count: 9,
            face_vertex_counts: vec![4; 4],
            face_vertex_indices: vec![0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            edge_vertices: Vec::new(),
            edge_creases: Vec::new(),
            vertex_corners: vec![0.0; 9],
        }
    }

    fn refine(
        scheme: Scheme,
        options: SchemeOptions,
        req: &UniformRefine,
    ) -> crate::RefinementResult {
        let refiner = Refiner::new(grid(), scheme, options).expect("refiner");
        refiner.refine_uniform(req).expect("refinement")
    }

    #[test]
    fn non_catmull_clark_scheme_is_rejected() {
        let result = refine(
            Scheme::DooSabin,
            SchemeOptions::default(),
            &UniformRefine::default(),
        );
        assert!(matches!(
            result.limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
        assert!(matches!(
            result.sectored_limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
    }

    #[test]
    fn partial_face_selection_is_rejected() {
        let req = UniformRefine {
            // SAFETY: 1 is non-zero.
            levels: NonZeroU8::new(1).unwrap(),
            selected_faces: Some(vec![true, true, true, false]),
            ..Default::default()
        };
        let result = refine(Scheme::CatmullClark, SchemeOptions::default(), &req);
        assert!(matches!(
            result.limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
        assert!(matches!(
            result.sectored_limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
    }

    #[test]
    fn natural_boundary_on_open_mesh_is_rejected() {
        let options = SchemeOptions {
            boundary_interpolation: crate::BoundaryInterpolation::Natural,
            ..Default::default()
        };
        let result = refine(Scheme::CatmullClark, options, &UniformRefine::default());
        assert!(matches!(
            result.limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
        assert!(matches!(
            result.sectored_limit_stencils(),
            Err(KernelError::NotImplemented(_)),
        ));
    }
}
