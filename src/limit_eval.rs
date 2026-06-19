//! Arbitrary-`(u, v)` limit evaluation over *any* refined
//! Catmull-Clark quad (limit-surface SDF design s2, route a2).
//!
//! [`RefinementResult::limit_evaluator`] wraps the s1
//! [`PatchTable`] in a uniform per-quad interface: regular quads
//! evaluate their bicubic B-spline patch directly; feature quads
//! (EV/crease/corner/boundary contact) evaluate by *recursive local
//! isolation* -- no eigenbasis machinery, exact to the depth the
//! query needs, reusing the ordinary [`Refiner`].
//!
//! # Feature isolation
//!
//! One isolation step extracts the feature quad's *support submesh* --
//! the quad plus every face sharing a vertex with it, carrying the
//! refined-level crease/corner sharpness of the support faces' edges
//! and vertices -- as a standalone [`Mesh`], refines it one
//! level with the same [`SchemeOptions`] (so semi-sharp decay matches
//! the original refinement exactly; design §10-3 v1 semantics), and
//! descends into the child quad containing the query (the quadrant of
//! `(u, v)`, with the parameter map recovered from the children's
//! vertex lineage). The recursion repeats until the containing child
//! is [`QuadClass::Regular`] on its submesh, then evaluates the s1
//! patch.
//!
//! The support contract makes each step exact for the central quad:
//! every control point of a central child's 4x4 patch neighborhood is
//! the vertex/edge/face point of a simplex *incident to a central
//! corner*, and the support carries those simplices' positions,
//! sharpness, and complete fans -- so the children's neighborhoods are
//! bit-faithful to a full-mesh refinement, and the central children's
//! own corners come out fully ringed (the extraction can recurse).
//! Submesh-*border* quads are artificially Feature (truncated outer
//! rings), but the recursion only ever evaluates central children.
//!
//! # Corner snapping and the persistent-feature predicate
//!
//! A query exactly at a quad corner whose vertex is a *persistent*
//! feature -- boundary, extraordinary valence, or never-decaying
//! sharpness under the scheme options -- would recurse forever, so it
//! snaps to the analytic per-sector limit masks of `limit.rs` at that
//! vertex instead (the same masks as
//! [`SectoredLimitStencils`](crate::SectoredLimitStencils), applied to
//! the current isolation level's positions). *Decaying* semi-sharp
//! corners keep descending until the sharpness hits zero and the
//! regular patch takes over. Because each descent doubles `(u, v)`
//! exactly, any query that sits on a feature line (a crease edge, a
//! boundary edge) becomes an exact corner once its dyadic bits
//! exhaust, so f32 queries on feature lines terminate too -- unless
//! the bits outlast the depth cap below.
//!
//! # Depth cap
//!
//! Isolation is capped at [`MAX_ISOLATION_DEPTH`] (20) levels below
//! the evaluated refinement. The cap is a backstop for queries the
//! snap cannot catch -- e.g. points on a crease line whose dyadic
//! expansion exceeds the cap -- and falls back to the sector masks at
//! the *nearest* corner of the deepest central child, at most
//! `2^-21` of the root quad away in parameter, so positions stay well
//! inside the oracle tolerances while derivatives degrade (f32
//! position differences at that depth are noise-dominated).
//!
//! # Weight rows
//!
//! [`LimitEvaluator::weights_at`] runs the same descent in the weight
//! domain: the terminal patch-basis (or corner-mask) row composes
//! through each isolation level's refinement stencils back to the
//! evaluated level's vertices, exposing the limit position's sparse
//! linear dependence on the refined positions -- the per-grab-point
//! `W` row of the amendment limit-surface oracle (its design §3).
//!
//! # Derivative conventions
//!
//! Derivatives are with respect to the *evaluated quad's* own
//! `[0, 1]^2`, exactly as in [`PatchTable`] (an extra `2^level`
//! against root-face/ptex parameterization); the recursion
//! chain-rules each level's 2x2 quadrant map (a rotation times 2) on
//! the way back up. Snapped corners return parametric one-sided
//! derivatives when the sector machinery can supply them
//! (crease-rule corners whose quad edges are sector bounds or the
//! regular cross direction; see
//! [`SectorDerivatives`](crate::limit::SectorDerivatives)) and
//! otherwise the raw in-sector tangent pair -- correct tangent plane
//! and winding-oriented normal (`du x dv`), but without parametric
//! alignment or scale. The latter is unavoidable at cone points and
//! smooth extraordinary vertices, where the parametric derivative
//! itself does not converge (`2 lambda != 1`).
//!
//! # Caching
//!
//! Isolation submeshes are cached per feature quad, and each level's
//! children lazily within its node, so repeated nearby queries
//! (the s3 Newton iterations) refine each support at most once.
//! Corner sector masks are memoized per `(face, corner)` at every
//! level (the s4 perf lever: feature-line feet re-snap the same
//! corners per sample; the masks depend only on topology + sharpness).
//! The caches sit behind `RefCell`s -- the evaluator is cheap to build
//! per thread but not `Sync`.

use std::cell::{OnceCell, RefCell};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::catmull_clark::stencils::Sparse;
use crate::closest_point::SearchIndex;
use crate::limit::{SectorDerivatives, corner_limit_sector};
use crate::patch::QuadClass;
use crate::{
    Adjacency, CornerRule, CreaseComputationMethod, KernelError, Mesh, PatchTable,
    RefinementResult, Refiner, Scheme, SchemeOptions, UniformRefine, VertexOrigin,
};

/// Isolation backstop below the evaluated level; see the module docs.
pub const MAX_ISOLATION_DEPTH: u32 = 20;

/// One limit sample: `(position, dp/du, dp/dv)`, the
/// [`PatchTable::eval_with_derivatives`] shape.
pub type LimitSample = ([f32; 3], [f32; 3], [f32; 3]);

/// Internal sample with f64 derivatives, chain-ruled up the recursion.
type IsolatedSample = ([f32; 3], [f64; 3], [f64; 3]);

/// Sparse f64 weight row over the current vertex set -- the
/// weight-domain counterpart of [`IsolatedSample`].
type WeightRow = Vec<(u32, f64)>;

/// Memoized [`corner_limit_sector`] output of one `(face, corner)`:
/// the position mask and the derivative payload. Feature-line queries
/// (crease feet, the s3 slide) re-snap the same corners per sample;
/// the masks depend only on topology + sharpness, so they are computed
/// once per evaluation context.
type CornerSector = (Sparse, SectorDerivatives);

/// Parent `(u, v)` of CSR corner `k`, the [`PatchTable`] convention.
const CORNER_UV: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Uniform per-quad limit evaluation over a refined level; built by
/// [`RefinementResult::limit_evaluator`]. See the module docs.
pub struct LimitEvaluator<'a> {
    pub(crate) result: &'a RefinementResult,
    pub(crate) positions: &'a [[f32; 3]],
    pub(crate) table: PatchTable,
    isolations: RefCell<HashMap<u32, IsolationNode>>,
    /// Memoized evaluated-level corner sector masks (see
    /// [`CornerSector`]).
    corner_sectors: RefCell<HashMap<(u32, usize), CornerSector>>,
    /// Closest-point acceleration index (s3), built on the first
    /// [`closest_point`](Self::closest_point) query; `closest_point.rs`.
    pub(crate) search: OnceCell<SearchIndex>,
}

impl RefinementResult {
    /// Build a [`LimitEvaluator`] over this refined level.
    ///
    /// `positions` is one position per refined vertex (the
    /// [`PatchTable`] evaluation input -- CPU-interpolated or read
    /// back from the GPU stencil path). Same gating as
    /// [`patch_table`](Self::patch_table): Catmull-Clark, at least one
    /// full (unselected) refinement.
    pub fn limit_evaluator<'a>(
        &'a self,
        positions: &'a [[f32; 3]],
    ) -> Result<LimitEvaluator<'a>, KernelError> {
        let table = self.patch_table()?;
        if positions.len() != self.topology.vertex_count as usize {
            return Err(KernelError::InvalidTopology(
                "positions length does not match the refined vertex count",
            ));
        }
        Ok(LimitEvaluator {
            result: self,
            positions,
            table,
            isolations: RefCell::new(HashMap::new()),
            corner_sectors: RefCell::new(HashMap::new()),
            search: OnceCell::new(),
        })
    }
}

impl LimitEvaluator<'_> {
    /// Classification of refined face `face` (the s1 table's).
    pub fn quad_class(&self, face: u32) -> QuadClass {
        self.table.quad_class(face)
    }

    /// Limit position of refined quad `face` at in-quad `(u, v)`
    /// (clamped to `[0, 1]^2`).
    pub fn eval(&self, face: u32, uv: [f32; 2]) -> Result<[f32; 3], KernelError> {
        match self.table.face_patch(face) {
            Some(patch) => Ok(self.table.eval(
                patch as usize,
                [clamp01(uv[0] as f64) as f32, clamp01(uv[1] as f64) as f32],
                self.positions,
            )),
            None => self.eval_with_derivatives(face, uv).map(|(p, _, _)| p),
        }
    }

    /// Limit position and first derivatives `(p, dp/du, dp/dv)` of
    /// refined quad `face` at in-quad `(u, v)` (clamped to
    /// `[0, 1]^2`); see the module docs for the derivative
    /// conventions. `du x dv` is the winding-oriented surface normal
    /// wherever it does not degenerate.
    pub fn eval_with_derivatives(
        &self,
        face: u32,
        uv: [f32; 2],
    ) -> Result<LimitSample, KernelError> {
        let uv = [clamp01(uv[0] as f64), clamp01(uv[1] as f64)];
        let (p, du, dv) = self.evaluate(face, uv)?;
        Ok((
            p,
            [du[0] as f32, du[1] as f32, du[2] as f32],
            [dv[0] as f32, dv[1] as f32, dv[2] as f32],
        ))
    }

    /// The sparse position stencil of [`eval`](Self::eval) at
    /// `(face, uv)`, over the refined level's vertices (amendment
    /// limit-oracle design §3): unique `(vertex, weight)` pairs
    /// partitioning unity, with `eval(face, uv) == sum_i w_i *
    /// positions[i]` up to f32 rounding. The limit point is linear in
    /// the refined positions, so the row is exactly
    /// `d eval(face, uv) / d positions`; folding it through
    /// [`RefinementResult::compose_stencils`] yields the cage stencil
    /// (the host-side split of
    /// [`RefinementResult::compose_limit_stencils`]).
    ///
    /// `(u, v)` is clamped like `eval` and the row follows the same
    /// branch structure -- regular patch basis, persistent-feature
    /// corner masks, recursive isolation with the identical depth-cap
    /// fallback -- composed in f64 and rounded once on return, so a
    /// row exists wherever `eval` succeeds and shares its isolation
    /// cache.
    pub fn weights_at(&self, face: u32, uv: [f32; 2]) -> Result<Vec<(u32, f32)>, KernelError> {
        let mesh = &self.result.topology;
        let adjacency = &self.result.adjacency;
        let options = &self.result.options;
        let uv = [clamp01(uv[0] as f64), clamp01(uv[1] as f64)];
        let row = if let Some(patch) = self.table.face_patch(face) {
            self.table
                .position_weights(patch as usize, [uv[0] as f32, uv[1] as f32])
                .to_vec()
        } else if let Some(corner) = snap_target(uv, face, mesh, adjacency, options) {
            let mut sectors = self.corner_sectors.borrow_mut();
            let sector =
                corner_sector_cached(&mut sectors, mesh, adjacency, options, face, corner)?;
            corner_row(sector)
        } else {
            let mut isolations = self.isolations.borrow_mut();
            let node = match isolations.entry(face) {
                Entry::Occupied(occupied) => occupied.into_mut(),
                Entry::Vacant(vacant) => vacant.insert(isolation_node(
                    mesh,
                    adjacency,
                    self.positions,
                    face,
                    options,
                )?),
            };
            let inner = weights_isolated(node, uv, 0, options)?;
            lift_row(&inner, node)
        };
        Ok(merge_row(row))
    }

    fn evaluate(&self, face: u32, uv: [f64; 2]) -> Result<IsolatedSample, KernelError> {
        let mesh = &self.result.topology;
        let adjacency = &self.result.adjacency;
        let options = &self.result.options;
        if let Some(patch) = self.table.face_patch(face) {
            let (p, du, dv) = self.table.eval_with_derivatives(
                patch as usize,
                [uv[0] as f32, uv[1] as f32],
                self.positions,
            );
            Ok((p, v3(du), v3(dv)))
        } else if let Some(corner) = snap_target(uv, face, mesh, adjacency, options) {
            let mut sectors = self.corner_sectors.borrow_mut();
            let sector =
                corner_sector_cached(&mut sectors, mesh, adjacency, options, face, corner)?;
            Ok(snap_corner(sector, self.positions, corner))
        } else {
            let mut isolations = self.isolations.borrow_mut();
            let node = match isolations.entry(face) {
                Entry::Occupied(occupied) => occupied.into_mut(),
                Entry::Vacant(vacant) => vacant.insert(isolation_node(
                    mesh,
                    adjacency,
                    self.positions,
                    face,
                    options,
                )?),
            };
            eval_isolated(node, uv, 0, options)
        }
    }
}

/// One isolation level: the once-refined support submesh of a quad.
struct IsolationNode {
    /// One-level refinement of the support submesh.
    refined: RefinementResult,
    /// Refined submesh vertex positions.
    positions: Vec<[f32; 3]>,
    /// s1 patches over the refined submesh.
    table: PatchTable,
    /// Submesh cage vertex -> parent-level vertex (the support
    /// extraction's selection; [`lift_row`] maps rows through it).
    sub_vertices: Vec<u32>,
    /// Memoized corner sector masks at this isolation level (see
    /// [`CornerSector`]).
    corner_sectors: HashMap<(u32, usize), CornerSector>,
    /// The central quad's child per quadrant (parent corner slot).
    children: [QuadrantChild; 4],
}

/// One quadrant child of an isolation node's central quad, with the
/// affine in-parent -> in-child parameter map and its lazily isolated
/// own node.
struct QuadrantChild {
    /// Refined-submesh face index.
    face: u32,
    /// Parent `(u, v)` of the child's CSR corner 0.
    origin: [f64; 2],
    /// `child_i = sum_j jacobian[i][j] * (parent - origin)_j`; entries
    /// in `{0, +-2}`, so the map is exact on dyadic parameters.
    jacobian: [[f64; 2]; 2],
    node: Option<Box<IsolationNode>>,
}

impl QuadrantChild {
    fn child_uv(&self, uv: [f64; 2]) -> [f64; 2] {
        let d = [uv[0] - self.origin[0], uv[1] - self.origin[1]];
        [
            self.jacobian[0][0] * d[0] + self.jacobian[0][1] * d[1],
            self.jacobian[1][0] * d[0] + self.jacobian[1][1] * d[1],
        ]
    }

    /// Chain-rule child-parameter derivatives back to the parent's.
    fn parent_derivatives(&self, du: [f64; 3], dv: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let j = &self.jacobian;
        let combine = |a: f64, b: f64| {
            [
                a * du[0] + b * dv[0],
                a * du[1] + b * dv[1],
                a * du[2] + b * dv[2],
            ]
        };
        (combine(j[0][0], j[1][0]), combine(j[0][1], j[1][1]))
    }
}

fn eval_isolated(
    node: &mut IsolationNode,
    uv: [f64; 2],
    depth: u32,
    options: &SchemeOptions,
) -> Result<IsolatedSample, KernelError> {
    let k = quadrant(uv);
    let child_uv = node.children[k].child_uv(uv);
    let face = node.children[k].face;
    debug_assert!(
        (-1e-12..=1.0 + 1e-12).contains(&child_uv[0])
            && (-1e-12..=1.0 + 1e-12).contains(&child_uv[1]),
        "quadrant map left the child quad: {child_uv:?}",
    );
    let mesh = &node.refined.topology;
    let adjacency = &node.refined.adjacency;
    let snap = |sectors: &mut HashMap<(u32, usize), CornerSector>,
                positions: &[[f32; 3]],
                corner: usize| {
        let sector = corner_sector_cached(sectors, mesh, adjacency, options, face, corner)?;
        Ok(snap_corner(sector, positions, corner))
    };
    let (p, du, dv) = if let Some(patch) = node.table.face_patch(face) {
        let (p, du, dv) = node.table.eval_with_derivatives(
            patch as usize,
            [child_uv[0] as f32, child_uv[1] as f32],
            &node.positions,
        );
        (p, v3(du), v3(dv))
    } else if let Some(corner) = snap_target(child_uv, face, mesh, adjacency, options) {
        snap(&mut node.corner_sectors, &node.positions, corner)?
    } else if depth + 1 >= MAX_ISOLATION_DEPTH {
        snap(
            &mut node.corner_sectors,
            &node.positions,
            nearest_corner(child_uv),
        )?
    } else {
        let child = match &mut node.children[k].node {
            Some(child) => child,
            vacant @ None => vacant.insert(Box::new(isolation_node(
                &node.refined.topology,
                &node.refined.adjacency,
                &node.positions,
                face,
                options,
            )?)),
        };
        eval_isolated(child, child_uv, depth + 1, options)?
    };
    let (du, dv) = node.children[k].parent_derivatives(du, dv);
    Ok((p, du, dv))
}

/// The weight-domain [`eval_isolated`]: the position row of `(u, v)`
/// over `node`'s once-refined submesh vertices, by the identical
/// descent (terminal patch basis, persistent-corner masks, depth-cap
/// fallback). Recursive rows come back over the child's refined
/// vertices and are lifted one level by [`lift_row`].
fn weights_isolated(
    node: &mut IsolationNode,
    uv: [f64; 2],
    depth: u32,
    options: &SchemeOptions,
) -> Result<WeightRow, KernelError> {
    let k = quadrant(uv);
    let child_uv = node.children[k].child_uv(uv);
    let face = node.children[k].face;
    let mesh = &node.refined.topology;
    let adjacency = &node.refined.adjacency;
    if let Some(patch) = node.table.face_patch(face) {
        Ok(node
            .table
            .position_weights(patch as usize, [child_uv[0] as f32, child_uv[1] as f32])
            .to_vec())
    } else if let Some(corner) = snap_target(child_uv, face, mesh, adjacency, options) {
        corner_sector_cached(
            &mut node.corner_sectors,
            mesh,
            adjacency,
            options,
            face,
            corner,
        )
        .map(corner_row)
    } else if depth + 1 >= MAX_ISOLATION_DEPTH {
        corner_sector_cached(
            &mut node.corner_sectors,
            mesh,
            adjacency,
            options,
            face,
            nearest_corner(child_uv),
        )
        .map(corner_row)
    } else {
        let child = match &mut node.children[k].node {
            Some(child) => child,
            vacant @ None => vacant.insert(Box::new(isolation_node(
                &node.refined.topology,
                &node.refined.adjacency,
                &node.positions,
                face,
                options,
            )?)),
        };
        let inner = weights_isolated(child, child_uv, depth + 1, options)?;
        Ok(lift_row(&inner, child))
    }
}

/// The position row of one snapped corner (the weight-domain
/// [`snap_corner`], position mask only).
fn corner_row(sector: &CornerSector) -> WeightRow {
    sector.0.iter().map(|&(i, w)| (i, w as f64)).collect()
}

/// Lift a row over `node`'s once-refined submesh vertices to a row
/// over the vertex set `node` was extracted from: expand each entry
/// through the submesh's one-level refinement stencils, then map the
/// submesh cage indices back through the extraction selection.
fn lift_row(row: &WeightRow, node: &IsolationNode) -> WeightRow {
    debug_assert_eq!(
        node.refined.level_stencils.len(),
        1,
        "isolation refines exactly one level",
    );
    let stencils = &node.refined.level_stencils[0];
    row.iter()
        .fold(BTreeMap::new(), |mut acc: BTreeMap<u32, f64>, &(j, w)| {
            let start = stencils.offsets[j as usize] as usize;
            let end = stencils.offsets[j as usize + 1] as usize;
            stencils.indices[start..end]
                .iter()
                .zip(&stencils.weights[start..end])
                .for_each(|(&cage, &cw)| {
                    *acc.entry(node.sub_vertices[cage as usize]).or_insert(0.0) += w * cw as f64;
                });
            acc
        })
        .into_iter()
        .collect()
}

/// Merge duplicate indices, drop exact zeros (corner/edge basis rows
/// carry structurally zero columns), and round once to f32.
fn merge_row(row: WeightRow) -> Vec<(u32, f32)> {
    row.iter()
        .fold(BTreeMap::new(), |mut acc: BTreeMap<u32, f64>, &(i, w)| {
            *acc.entry(i).or_insert(0.0) += w;
            acc
        })
        .into_iter()
        .filter(|&(_, w)| w != 0.0)
        .map(|(i, w)| (i, w as f32))
        .collect()
}

/// Build one isolation node: extract the support submesh of `face`,
/// refine it one level, and locate the central quad's children.
fn isolation_node(
    mesh: &Mesh,
    adjacency: &Adjacency,
    positions: &[[f32; 3]],
    face: u32,
    options: &SchemeOptions,
) -> Result<IsolationNode, KernelError> {
    let off = (face * 4) as usize;
    let corners = &mesh.face_vertex_indices[off..off + 4];

    // The support: every face sharing a vertex with the central quad.
    let support_faces: Vec<u32> = corners
        .iter()
        .flat_map(|&corner| {
            let start = adjacency.vertex_face_offsets[corner as usize] as usize;
            let end = adjacency.vertex_face_offsets[corner as usize + 1] as usize;
            adjacency.vertex_faces[start..end].iter().copied()
        })
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .collect();
    let central =
        support_faces
            .iter()
            .position(|&f| f == face)
            .ok_or(KernelError::InvalidTopology(
                "central quad is not incident to its own corners",
            ))? as u32;

    // Dense submesh vertex order: first appearance across the support
    // faces' CSR corners.
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut sub_vertices: Vec<u32> = Vec::new();
    let face_vertex_indices: Vec<u32> = support_faces
        .iter()
        .flat_map(|&f| mesh.face_vertex_indices[(f * 4) as usize..(f * 4) as usize + 4].iter())
        .map(|&v| {
            *vertex_map.entry(v).or_insert_with(|| {
                sub_vertices.push(v);
                sub_vertices.len() as u32 - 1
            })
        })
        .collect();

    // Sharpness carried verbatim: creased edges of the support faces
    // (every edge incident to a central corner is one) and the
    // vertices' corner values.
    let crease_edges: BTreeSet<u32> = support_faces
        .iter()
        .flat_map(|&f| adjacency.face_edges[(f * 4) as usize..(f * 4) as usize + 4].iter())
        .copied()
        .filter(|&e| mesh.edge_creases[e as usize] > 0.0)
        .collect();
    let (edge_vertices, edge_creases): (Vec<[u32; 2]>, Vec<f32>) = crease_edges
        .iter()
        .map(|&e| {
            let [a, b] = mesh.edge_vertices[e as usize];
            (
                [vertex_map[&a], vertex_map[&b]],
                mesh.edge_creases[e as usize],
            )
        })
        .unzip();

    let submesh = Mesh {
        vertex_count: sub_vertices.len() as u32,
        face_vertex_counts: vec![4; support_faces.len()],
        face_vertex_indices,
        edge_vertices,
        edge_creases,
        vertex_corners: sub_vertices
            .iter()
            .map(|&v| mesh.vertex_corners[v as usize])
            .collect(),
    };
    let sub_positions: Vec<[f32; 3]> = sub_vertices
        .iter()
        .map(|&v| positions[v as usize])
        .collect();
    let central_corners: Vec<u32> = corners.iter().map(|&c| vertex_map[&c]).collect();

    let refined = Refiner::new(submesh, Scheme::CatmullClark, *options)?
        .refine_uniform(&UniformRefine::default())?;
    let positions = refined.interpolate(&sub_positions);
    let table = refined.patch_table()?;
    let children = quadrant_children(&refined, central, &central_corners)?;
    Ok(IsolationNode {
        refined,
        positions,
        table,
        sub_vertices,
        corner_sectors: HashMap::new(),
        children,
    })
}

/// Locate the central quad's four children in the refined submesh and
/// recover each child's parent-parameter frame from its vertex
/// lineage: the child holding the vertex point of central corner `k`
/// covers quadrant `k`, its CSR corners sit (winding-preserved) at the
/// corner, the two adjacent edge midpoints, and the face center.
fn quadrant_children(
    refined: &RefinementResult,
    central: u32,
    central_corners: &[u32],
) -> Result<[QuadrantChild; 4], KernelError> {
    let mut children: [Option<QuadrantChild>; 4] = [None, None, None, None];
    for (child_face, &parent) in refined.lineage.face_parent.iter().enumerate() {
        if parent != central {
            continue;
        }
        let child_corners =
            &refined.topology.face_vertex_indices[child_face * 4..child_face * 4 + 4];
        let (r, k) = child_corners
            .iter()
            .enumerate()
            .find_map(|(r, &c)| match refined.lineage.vertex_origin[c as usize] {
                VertexOrigin::Vertex(pv) => central_corners
                    .iter()
                    .position(|&cc| cc == pv)
                    .map(|k| (r, k)),
                _ => None,
            })
            .ok_or(KernelError::InvalidTopology(
                "central child quad has no central-corner vertex point",
            ))?;
        debug_assert!(
            matches!(
                refined.lineage.vertex_origin[child_corners[(r + 2) % 4] as usize],
                VertexOrigin::Face(f) if f == central,
            ),
            "central child quad's diagonal is not the central face point",
        );
        debug_assert!(
            matches!(
                refined.lineage.vertex_origin[child_corners[(r + 1) % 4] as usize],
                VertexOrigin::Edge(_),
            ) && matches!(
                refined.lineage.vertex_origin[child_corners[(r + 3) % 4] as usize],
                VertexOrigin::Edge(_),
            ),
            "central child quad's off-corners are not edge points",
        );

        let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let mut parent_uv = [[0.0f64; 2]; 4];
        parent_uv[r] = CORNER_UV[k];
        parent_uv[(r + 1) % 4] = mid(CORNER_UV[k], CORNER_UV[(k + 1) % 4]);
        parent_uv[(r + 2) % 4] = [0.5, 0.5];
        parent_uv[(r + 3) % 4] = mid(CORNER_UV[k], CORNER_UV[(k + 3) % 4]);

        let e_u = [
            parent_uv[1][0] - parent_uv[0][0],
            parent_uv[1][1] - parent_uv[0][1],
        ];
        let e_v = [
            parent_uv[3][0] - parent_uv[0][0],
            parent_uv[3][1] - parent_uv[0][1],
        ];
        let det = e_u[0] * e_v[1] - e_v[0] * e_u[1];
        debug_assert!(det.abs() > 1e-12, "degenerate child parameter frame");
        children[k] = Some(QuadrantChild {
            face: child_face as u32,
            origin: parent_uv[0],
            jacobian: [[e_v[1] / det, -e_v[0] / det], [-e_u[1] / det, e_u[0] / det]],
            node: None,
        });
    }
    // Winding sanity: consecutive quadrant children share the edge
    // point on the parent edge between them.
    debug_assert!(
        (0..4).all(|k| {
            children[k]
                .as_ref()
                .zip(children[(k + 1) % 4].as_ref())
                .is_none_or(|(a, b)| {
                    let corners = |f: u32| {
                        &refined.topology.face_vertex_indices
                            [(f * 4) as usize..(f * 4) as usize + 4]
                    };
                    corners(a.face)
                        .iter()
                        .filter(|c| corners(b.face).contains(c))
                        .count()
                        == 2
                })
        }),
        "quadrant children do not share their lead edge points",
    );
    let [a, b, c, d] = children;
    a.zip(b)
        .zip(c.zip(d))
        .map(|((a, b), (c, d))| [a, b, c, d])
        .ok_or(KernelError::InvalidTopology(
            "central quad did not refine into four quadrant children",
        ))
}

/// Memoized lookup of one `(face, corner)`'s sector masks; computes
/// [`corner_limit_sector`] on the first request only.
fn corner_sector_cached<'c>(
    cache: &'c mut HashMap<(u32, usize), CornerSector>,
    mesh: &Mesh,
    adjacency: &Adjacency,
    options: &SchemeOptions,
    face: u32,
    corner: usize,
) -> Result<&'c CornerSector, KernelError> {
    match cache.entry((face, corner)) {
        Entry::Occupied(occupied) => Ok(occupied.into_mut()),
        Entry::Vacant(vacant) => {
            Ok(vacant.insert(corner_limit_sector(face, corner, mesh, adjacency, options)?))
        }
    }
}

/// Sector-mask evaluation at one quad corner from its memoized masks:
/// position plus either parametric one-sided `(du, dv)` (mapped from
/// the corner's out/in edge derivatives by the corner's orientation in
/// the quad frame) or the raw in-sector tangent plane. See
/// [`SectorDerivatives`].
fn snap_corner(sector: &CornerSector, positions: &[[f32; 3]], corner: usize) -> IsolatedSample {
    let (position, derivatives) = sector;
    let p = apply(position, positions);
    let (du, dv) = match derivatives {
        SectorDerivatives::Parametric { d_out, d_in } => {
            let (d_out, d_in) = (apply(d_out, positions), apply(d_in, positions));
            let neg = |d: [f64; 3]| [-d[0], -d[1], -d[2]];
            match corner {
                0 => (d_out, d_in),
                1 => (neg(d_in), d_out),
                2 => (neg(d_out), neg(d_in)),
                _ => (d_in, neg(d_out)),
            }
        }
        SectorDerivatives::Plane { tangent1, tangent2 } => {
            (apply(tangent1, positions), apply(tangent2, positions))
        }
    };
    ([p[0] as f32, p[1] as f32, p[2] as f32], du, dv)
}

/// Whether the vertex stays a feature at every deeper isolation level:
/// boundary, extraordinary valence, or sharpness that never decays
/// under `options` (infinite always; the OpenSubdiv `10.0` sentinel
/// where the decay rule preserves it; any positive value under the
/// normalize flags). Decaying semi-sharpness returns `false` -- deeper
/// isolation resolves it.
fn persistent_feature_vertex(
    vi: usize,
    mesh: &Mesh,
    adjacency: &Adjacency,
    options: &SchemeOptions,
) -> bool {
    let start = adjacency.vertex_edge_offsets[vi] as usize;
    let end = adjacency.vertex_edge_offsets[vi + 1] as usize;
    let persistent_crease = |s: f32| persistent_sharp_edge(s, options);
    let persistent_corner = |s: f32| {
        s > 0.0
            && (options.corner_normalize
                || s.is_infinite()
                || (options.corner_rule == CornerRule::OpenSubdivDeRose && s >= 10.0))
    };
    adjacency.vertex_is_boundary[vi]
        || end - start != 4
        || persistent_corner(mesh.vertex_corners[vi])
        || adjacency.vertex_edges[start..end]
            .iter()
            .any(|&e| persistent_crease(mesh.edge_creases[e as usize]))
}

/// Whether an edge's stored sharpness never decays under `options`
/// (the crease half of [`persistent_feature_vertex`]; the s3
/// closest-point walk uses it to recognize feature lines whose
/// on-line derivatives are depth-cap-degraded).
pub(crate) fn persistent_sharp_edge(sharpness: f32, options: &SchemeOptions) -> bool {
    sharpness > 0.0
        && (options.crease_normalize
            || sharpness.is_infinite()
            || (options.corner_rule == CornerRule::OpenSubdivDeRose
                && options.crease_computation == CreaseComputationMethod::Uniform
                && sharpness >= 10.0))
}

/// The persistent-feature corner `(u, v)` snaps to, if any -- the
/// shared snap branch of the value and weight evaluations.
fn snap_target(
    uv: [f64; 2],
    face: u32,
    mesh: &Mesh,
    adjacency: &Adjacency,
    options: &SchemeOptions,
) -> Option<usize> {
    exact_corner(uv).filter(|&corner| {
        persistent_feature_vertex(
            mesh.face_vertex_indices[(face * 4) as usize + corner] as usize,
            mesh,
            adjacency,
            options,
        )
    })
}

/// The CSR corner at `(u, v)` when both parameters are exactly 0 or 1.
fn exact_corner(uv: [f64; 2]) -> Option<usize> {
    let bit = |t: f64| (t == 0.0).then_some(false).or((t == 1.0).then_some(true));
    bit(uv[0]).zip(bit(uv[1])).map(|bits| match bits {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    })
}

/// Quadrant (= parent corner slot) containing `(u, v)`; ties toward
/// corner 0.
fn quadrant(uv: [f64; 2]) -> usize {
    match (uv[0] > 0.5, uv[1] > 0.5) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// CSR corner nearest to `(u, v)` (the depth-cap fallback target).
fn nearest_corner(uv: [f64; 2]) -> usize {
    match (uv[0] >= 0.5, uv[1] >= 0.5) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

fn clamp01(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}

fn v3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// Apply one sparse mask row to a positions buffer, accumulating f64.
fn apply(row: &Sparse, positions: &[[f32; 3]]) -> [f64; 3] {
    row.iter().fold([0.0f64; 3], |acc, &(i, w)| {
        let p = positions[i as usize];
        [
            acc[0] + w as f64 * p[0] as f64,
            acc[1] + w as f64 * p[1] as f64,
            acc[2] + w as f64 * p[2] as f64,
        ]
    })
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU8;

    use crate::{KernelError, Mesh, Refiner, Scheme, SchemeOptions, UniformRefine};

    /// A 2x2 quad grid (the geometry gates live in
    /// `tests/limit_eval.rs` and `tests/limit_eval_osd_oracle.rs`;
    /// these unit tests cover the error paths only).
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

    #[test]
    fn non_catmull_clark_scheme_is_rejected() {
        let refiner =
            Refiner::new(grid(), Scheme::DooSabin, SchemeOptions::default()).expect("refiner");
        let result = refiner
            .refine_uniform(&UniformRefine::default())
            .expect("refinement");
        let positions = vec![[0.0f32; 3]; result.topology.vertex_count as usize];
        assert!(matches!(
            result.limit_evaluator(&positions).err(),
            Some(KernelError::NotImplemented(_)),
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
        let refiner =
            Refiner::new(grid(), Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
        let result = refiner.refine_uniform(&req).expect("refinement");
        let positions = vec![[0.0f32; 3]; result.topology.vertex_count as usize];
        assert!(matches!(
            result.limit_evaluator(&positions).err(),
            Some(KernelError::NotImplemented(_)),
        ));
    }

    #[test]
    fn mismatched_positions_length_is_rejected() {
        let refiner =
            Refiner::new(grid(), Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
        let result = refiner
            .refine_uniform(&UniformRefine::default())
            .expect("refinement");
        let positions = vec![[0.0f32; 3]; 3];
        assert!(matches!(
            result.limit_evaluator(&positions).err(),
            Some(KernelError::InvalidTopology(_)),
        ));
    }
}
