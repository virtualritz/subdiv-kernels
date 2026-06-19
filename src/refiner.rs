//! Topology refiner with cached analysis.
//!
//! The refiner is two-phase for all four schemes:
//! [`Refiner::refine_topology`] builds and caches per-level topology
//! in a [`Refinement`]; stencils are extracted on demand via
//! [`Refinement::vertex_stencils`] or
//! [`Refinement::face_varying_stencils`] without rebuilding
//! topology.
//!
//! Per-level cached data is kept in a scheme-specific struct
//! (`CcLevelData`, `LoopLevelData`, `Sqrt3LevelData`, or
//! `DooSabinLevelData`) behind a single owning
//! [`LevelData`] enum. A [`LevelDataCommon`] trait exposes the
//! always-present fields (`mesh`, `lineage`, `face_selected`,
//! `adjacency`) via enum-dispatch-generated impls, so accessors on
//! [`Refinement`] can read them without manual match arms while
//! scheme-specific stencil extraction still matches on the enum
//! variant to pick the right per-scheme function.

use enum_dispatch::enum_dispatch;

use crate::catmull_clark::stencils::{
    CcLevelData, base_level_data as cc_base_level_data,
    refine_topology_once as cc_refine_topology_once,
    vertex_stencils_from_level as cc_vertex_stencils_from_level,
};
use crate::doo_sabin::stencils::{
    DooSabinLevelData, base_level_data as doo_sabin_base_level_data,
    refine_topology_once as doo_sabin_refine_topology_once,
    vertex_stencils_from_level as doo_sabin_vertex_stencils_from_level,
};
use crate::loop_subdivision::stencils::{
    LoopLevelData, base_level_data as loop_base_level_data,
    refine_topology_once as loop_refine_topology_once,
    vertex_stencils_from_level as loop_vertex_stencils_from_level,
};
use crate::sqrt3::stencils::{
    Sqrt3LevelData, base_level_data as sqrt3_base_level_data,
    refine_topology_once as sqrt3_refine_topology_once,
    vertex_stencils_from_level as sqrt3_vertex_stencils_from_level,
};
use crate::{
    Adjacency, FaceVaryingChannel, FaceVaryingInterpolation, KernelError, LineageMaps, Mesh,
    RefinementResult, Scheme, SchemeOptions, StencilTable, UniformRefine,
};

/// Per-scheme level data: shared accessor interface for the fields
/// that every scheme's level cache holds. Scheme-specific fields
/// (notably the internal `topo` analysis struct and crease/corner
/// scratch) are read directly from the concrete variant in places
/// that need them.
#[enum_dispatch]
pub(crate) trait LevelDataCommon {
    fn mesh(&self) -> &Mesh;
    fn lineage(&self) -> &LineageMaps;
    fn face_selected(&self) -> &[bool];
    fn adjacency(&self) -> &Adjacency;
}

/// Owning enum over per-scheme level caches.
///
/// Variants are type-distinct so the compiler can enforce "this
/// `LevelData::Cc(_)` holds a `CcLevelData`, not a `LoopLevelData`".
/// Per-element overhead is one discriminant byte plus alignment
/// padding — negligible compared to the kilobytes/megabytes of cached
/// topology inside each variant.
#[enum_dispatch(LevelDataCommon)]
pub(crate) enum LevelData {
    Cc(CcLevelData),
    Loop(LoopLevelData),
    Sqrt3(Sqrt3LevelData),
    DooSabin(DooSabinLevelData),
}

/// Subdivision refiner with validated topology and cached analysis.
///
/// Create a `Refiner` once for a given topology + scheme, then call
/// [`refine_topology`](Self::refine_topology) for the cached two-phase
/// API or [`refine_uniform`](Self::refine_uniform) for the one-shot
/// `RefinementResult`-returning API.
///
/// # Example
///
/// ```ignore
/// use core::num::NonZeroU8;
/// let refiner = Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default())?;
/// let req = UniformRefine { levels: NonZeroU8::new(2).unwrap(), ..Default::default() };
/// let result = refiner.refine_uniform(&req)?;
/// let positions = result.interpolate(&my_positions);
/// ```
pub struct Refiner {
    topology: Mesh,
    scheme: Scheme,
    options: SchemeOptions,
}

/// Cached multi-level refinement topology.
///
/// Created by [`Refiner::refine_topology`]. Use
/// [`vertex_stencils`](Self::vertex_stencils) and
/// [`face_varying_stencils`](Self::face_varying_stencils) to compute
/// stencils from the cached topology without redundant edge discovery,
/// and [`level_lineage`](Self::level_lineage) /
/// [`refinement_steps`](Self::refinement_steps) to walk multi-level
/// ancestry by reference.
///
/// The shape mirrors OpenSubdiv's `Far::TopologyRefiner` — all per-level
/// queries return borrowed slices into cached storage so adapters can
/// fold across levels without cloning per-level state.
#[must_use]
pub struct Refinement {
    /// Base level + all refined levels. `levels[0]` is the base mesh
    /// (lineage is [`LineageMaps::default`]); `levels[1..=N]` hold
    /// each refinement step's cached state.
    levels: Vec<LevelData>,
    scheme: Scheme,
    options: SchemeOptions,
    /// Polyline tracking (populated when the request asked for it).
    edge_polylines: Option<Vec<Vec<u32>>>,
}

/// Owned outputs of [`Refinement::into_final_parts`].
///
/// Mirrors the public bits of [`RefinementResult`] without the per-level
/// stencil tables — those must be computed via
/// [`vertex_stencils`](Refinement::vertex_stencils) and
/// [`face_varying_stencils`](Refinement::face_varying_stencils)
/// *before* consuming the handle.
///
/// For multi-level ancestry, walk
/// [`Refinement::level_lineage`] before calling
/// [`into_final_parts`](Refinement::into_final_parts). Those
/// accessors return borrowed slices and cost nothing; `into_final_parts`
/// then consumes the handle and moves out the final-level owned state
/// with zero clones.
#[non_exhaustive]
pub struct RefinedFinalParts {
    /// Final-level control mesh (faces, edges, creases).
    pub topology: Mesh,
    /// Per-level vertex/edge/face lineage back to the input.
    pub lineage: LineageMaps,
    /// Final-level adjacency (edges, vertex rings, boundary flags).
    pub adjacency: Adjacency,
    /// Final-level face-selection mask, if face-selective refinement was used.
    pub selected_faces: Option<Vec<bool>>,
    /// Per input edge, the refined vertices lying on it, if requested.
    pub edge_polylines: Option<Vec<Vec<u32>>>,
}

impl Refinement {
    /// The subdivision scheme this refinement was produced with.
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// Compute per-level vertex stencils from cached topology.
    ///
    /// Returns one `StencilTable` per refinement step (length equals
    /// [`refinement_steps`](Self::refinement_steps)). Apply with the
    /// chaining pattern from
    /// [`RefinementResult::interpolate`](crate::RefinementResult::interpolate):
    ///
    /// ```ignore
    /// let tables = refined.vertex_stencils();
    /// let final_positions = tables
    ///     .iter()
    ///     .fold(input_positions, |data, t| t.interpolate(&data));
    /// ```
    pub fn vertex_stencils(&self) -> Vec<StencilTable> {
        // The parent of level k is `levels[k]` (level 0 = base).
        // Skip the last element because it has no "next level" output.
        let parent_count = self.levels.len().saturating_sub(1);
        (0..parent_count)
            .map(|k| match &self.levels[k] {
                LevelData::Cc(parent) => cc_vertex_stencils_from_level(parent, &self.options),
                LevelData::Loop(parent) => loop_vertex_stencils_from_level(parent, &self.options),
                LevelData::Sqrt3(parent) => sqrt3_vertex_stencils_from_level(parent, &self.options),
                LevelData::DooSabin(parent) => {
                    doo_sabin_vertex_stencils_from_level(parent, &self.options)
                }
            })
            .collect()
    }

    /// Compute per-level face-varying stencils from cached topology.
    ///
    /// All four [`FaceVaryingInterpolation`] modes are implemented for CC,
    /// Loop and √3; Doo-Sabin's face-local smooth rule makes its three smooth
    /// modes coincide.
    pub fn face_varying_stencils(
        &self,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        let level_count = self.levels.len().saturating_sub(1);
        let mut tables = Vec::with_capacity(level_count);
        let mut current_fvar = channel.clone();

        for i in 0..level_count {
            let table = match (&self.levels[i], &self.levels[i + 1]) {
                (LevelData::Cc(parent), LevelData::Cc(child)) => {
                    crate::catmull_clark::face_varying::fvar_stencils_once(
                        parent,
                        child,
                        &current_fvar,
                        mode,
                        &self.options,
                    )?
                }
                (LevelData::Loop(parent), LevelData::Loop(child)) => {
                    crate::loop_subdivision::face_varying::fvar_stencils_once(
                        parent,
                        child,
                        &current_fvar,
                        mode,
                        &self.options,
                    )?
                }
                (LevelData::Sqrt3(parent), LevelData::Sqrt3(child)) => {
                    crate::sqrt3::face_varying::fvar_stencils_once(
                        parent,
                        child,
                        &current_fvar,
                        mode,
                        &self.options,
                    )?
                }
                (LevelData::DooSabin(parent), LevelData::DooSabin(child)) => {
                    crate::doo_sabin::face_varying::fvar_stencils_once(
                        parent,
                        child,
                        &current_fvar,
                        mode,
                        &self.options,
                    )?
                }
                _ => unreachable!("a refinement chain holds one scheme's level data throughout"),
            };
            tables.push(table);

            current_fvar = identity_channel(self.levels[i + 1].mesh());
        }

        Ok(tables)
    }

    /// Final refined topology.
    pub fn final_topology(&self) -> &Mesh {
        self.levels.last().expect("at least one level").mesh()
    }

    /// Lineage maps (last level, relative to level N-1).
    pub fn lineage(&self) -> &LineageMaps {
        self.levels.last().expect("at least one level").lineage()
    }

    /// Adjacency (last level).
    pub fn adjacency(&self) -> &Adjacency {
        self.levels.last().expect("at least one level").adjacency()
    }

    /// Edge polylines (original parent-edge slots → refined vertex
    /// index sequences), if requested at refinement time.
    pub fn edge_polylines(&self) -> Option<&[Vec<u32>]> {
        self.edge_polylines.as_deref()
    }

    /// Selected faces at final level.
    pub fn selected_faces(&self) -> Option<&[bool]> {
        // Only meaningful when at least one refinement level ran AND
        // the request carried a selection mask; the base level stores
        // `face_selected = vec![true; face_count]` when no selection
        // was supplied, so we need to distinguish those cases. For
        // the CC cached path we mirror the earlier accessor's check
        // and return `None` when `levels.len() <= 1` (base-only =
        // refinement did not run).
        if self.levels.len() <= 1 {
            return None;
        }
        Some(
            self.levels
                .last()
                .expect("at least one level")
                .face_selected(),
        )
    }

    /// Number of refinement steps (levels beyond the base mesh).
    pub fn refinement_steps(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }

    /// Per-level lineage view by reference, zero allocation.
    ///
    /// `level_lineage(step)` returns the lineage of the refinement step
    /// indexed from 0 (first refinement level relative to the base
    /// mesh). Valid range is `0..refinement_steps()`.
    ///
    /// Adapters fold over this to chain ancestry from the final level
    /// back to the original input mesh, mirroring OpenSubdiv's
    /// `TopologyLevel::face_parent_face` / per-level walk pattern.
    pub fn level_lineage(&self, step: usize) -> Option<&LineageMaps> {
        // levels[0] is the base (empty lineage); refinement steps
        // start at levels[1].
        self.levels.get(step + 1).map(|level| level.lineage())
    }

    /// Consume the handle and yield its owned final-level outputs.
    ///
    /// Call this *after* any stencil or per-level-lineage borrows have
    /// ended ([`vertex_stencils`](Self::vertex_stencils),
    /// [`face_varying_stencils`](Self::face_varying_stencils),
    /// [`level_lineage`](Self::level_lineage)), since those read from
    /// cached level data that this method moves out.
    ///
    /// Zero clones: topology / lineage / adjacency / polylines are
    /// moved out of the final level's scheme-specific struct.
    pub fn into_final_parts(mut self) -> RefinedFinalParts {
        let last = self.levels.pop().expect("at least one level");
        // After the pop, `levels.len() >= 1` means refinement ran
        // (base still present). Preserve the selection mask only when
        // refinement actually happened.
        let refinement_ran = !self.levels.is_empty();
        match last {
            LevelData::Cc(level) => RefinedFinalParts {
                topology: level.mesh,
                lineage: level.lineage,
                adjacency: level.adjacency,
                selected_faces: refinement_ran.then_some(level.face_selected),
                edge_polylines: self.edge_polylines,
            },
            LevelData::Loop(level) => RefinedFinalParts {
                topology: level.mesh,
                lineage: level.lineage,
                adjacency: level.adjacency,
                selected_faces: refinement_ran.then_some(level.face_selected),
                edge_polylines: self.edge_polylines,
            },
            LevelData::Sqrt3(level) => RefinedFinalParts {
                topology: level.mesh,
                lineage: level.lineage,
                adjacency: level.adjacency,
                selected_faces: refinement_ran.then_some(level.face_selected),
                edge_polylines: self.edge_polylines,
            },
            LevelData::DooSabin(level) => RefinedFinalParts {
                topology: level.mesh,
                lineage: level.lineage,
                adjacency: level.adjacency,
                selected_faces: refinement_ran.then_some(level.face_selected),
                edge_polylines: self.edge_polylines,
            },
        }
    }
}

impl Refiner {
    /// Create a new refiner, validating the input topology.
    pub fn new(
        topology: Mesh,
        scheme: Scheme,
        options: SchemeOptions,
    ) -> Result<Self, KernelError> {
        topology.validate()?;
        Ok(Self {
            topology,
            scheme,
            options,
        })
    }

    /// Access the input topology.
    pub fn topology(&self) -> &Mesh {
        &self.topology
    }

    /// Access the scheme.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// Access the scheme options.
    pub fn options(&self) -> &SchemeOptions {
        &self.options
    }

    /// Build and cache per-level topology for all refinement levels.
    ///
    /// This is the expensive phase (edge discovery, adjacency
    /// construction). Call [`Refinement::vertex_stencils`] and
    /// [`Refinement::face_varying_stencils`] to compute stencils
    /// from the cached topology without redundant edge rebuilds.
    pub fn refine_topology(&self, req: &UniformRefine) -> Result<Refinement, KernelError> {
        match self.scheme {
            Scheme::CatmullClark => self.refine_topology_cc(req),
            Scheme::Loop => self.refine_topology_loop(req),
            Scheme::Sqrt3 => self.refine_topology_sqrt3(req),
            Scheme::DooSabin => self.refine_topology_doo_sabin(req),
        }
    }

    fn active_selection(&self, req: &UniformRefine) -> Result<Vec<bool>, KernelError> {
        let initial_face_count = self.topology.face_vertex_counts.len();
        req.selected_faces
            .as_ref()
            .map(|m| {
                (m.len() == initial_face_count).then(|| m.clone()).ok_or(
                    KernelError::InvalidTopology(
                        "selected-face mask length does not match face count",
                    ),
                )
            })
            .transpose()
            .map(|opt| opt.unwrap_or_else(|| vec![true; initial_face_count]))
    }

    fn refine_topology_cc(&self, req: &UniformRefine) -> Result<Refinement, KernelError> {
        let active_sel = self.active_selection(req)?;
        let base = cc_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;

        let mut levels: Vec<LevelData> = Vec::with_capacity(req.levels.get() as usize + 1);
        levels.push(LevelData::Cc(base));

        let mut polylines = req.edge_polylines.then(|| {
            self.topology
                .edge_vertices
                .iter()
                .map(|&[v0, v1]| vec![v0, v1])
                .collect::<Vec<_>>()
        });

        for _ in 0..req.levels.get() {
            let parent = match levels.last().unwrap() {
                LevelData::Cc(p) => p,
                _ => unreachable!("CC refine push-chain only touches CC variants"),
            };
            let child =
                cc_refine_topology_once(parent, &self.options, req.selection_boundary_crease)?;

            if let Some(ref mut polys) = polylines {
                Self::refine_polylines(polys, &child.lineage, &parent.mesh);
            }

            levels.push(LevelData::Cc(child));
        }

        Ok(Refinement {
            levels,
            scheme: self.scheme,
            options: self.options,
            edge_polylines: polylines,
        })
    }

    fn refine_topology_loop(&self, req: &UniformRefine) -> Result<Refinement, KernelError> {
        let active_sel = self.active_selection(req)?;
        let base = loop_base_level_data(
            &self.topology,
            active_sel,
            &self.options,
            req.selection_boundary_crease,
        )?;

        let mut levels: Vec<LevelData> = Vec::with_capacity(req.levels.get() as usize + 1);
        levels.push(LevelData::Loop(base));

        let mut polylines = req.edge_polylines.then(|| {
            self.topology
                .edge_vertices
                .iter()
                .map(|&[v0, v1]| vec![v0, v1])
                .collect::<Vec<_>>()
        });

        for _ in 0..req.levels.get() {
            let parent = match levels.last().unwrap() {
                LevelData::Loop(p) => p,
                _ => unreachable!("Loop refine push-chain only touches Loop variants"),
            };
            let child =
                loop_refine_topology_once(parent, &self.options, req.selection_boundary_crease)?;

            if let Some(ref mut polys) = polylines {
                Self::refine_polylines(polys, &child.lineage, &parent.mesh);
            }

            levels.push(LevelData::Loop(child));
        }

        Ok(Refinement {
            levels,
            scheme: self.scheme,
            options: self.options,
            edge_polylines: polylines,
        })
    }

    fn refine_topology_sqrt3(&self, req: &UniformRefine) -> Result<Refinement, KernelError> {
        let active_sel = self.active_selection(req)?;
        let base =
            sqrt3_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;

        let mut levels: Vec<LevelData> = Vec::with_capacity(req.levels.get() as usize + 1);
        levels.push(LevelData::Sqrt3(base));

        let mut polylines = req.edge_polylines.then(|| {
            self.topology
                .edge_vertices
                .iter()
                .map(|&[v0, v1]| vec![v0, v1])
                .collect::<Vec<_>>()
        });

        for _ in 0..req.levels.get() {
            let parent = match levels.last().unwrap() {
                LevelData::Sqrt3(p) => p,
                _ => unreachable!("Sqrt3 refine push-chain only touches Sqrt3 variants"),
            };
            let child =
                sqrt3_refine_topology_once(parent, &self.options, req.selection_boundary_crease)?;

            if let Some(ref mut polys) = polylines {
                Self::refine_polylines(polys, &child.lineage, &parent.mesh);
            }

            levels.push(LevelData::Sqrt3(child));
        }

        Ok(Refinement {
            levels,
            scheme: self.scheme,
            options: self.options,
            edge_polylines: polylines,
        })
    }

    fn refine_topology_doo_sabin(&self, req: &UniformRefine) -> Result<Refinement, KernelError> {
        let active_sel = self.active_selection(req)?;
        let base =
            doo_sabin_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;

        let mut levels: Vec<LevelData> = Vec::with_capacity(req.levels.get() as usize + 1);
        levels.push(LevelData::DooSabin(base));

        // Doo-Sabin is a dual scheme: parent vertices do not survive as
        // `VertexOrigin::Vertex`, and parent edges do not survive as edges
        // (they become faces). `Self::refine_polylines` relies on
        // vertex-origin / edge-origin lineage to advance polylines, so its
        // output is meaningless for Doo-Sabin. Always return `None`, even if
        // the caller set `edge_polylines: true`.
        let polylines: Option<Vec<Vec<u32>>> = None;

        for _ in 0..req.levels.get() {
            let parent = match levels.last().unwrap() {
                LevelData::DooSabin(p) => p,
                _ => unreachable!("DooSabin refine push-chain only touches DooSabin variants"),
            };
            let child = doo_sabin_refine_topology_once(
                parent,
                &self.options,
                req.selection_boundary_crease,
            )?;

            levels.push(LevelData::DooSabin(child));
        }

        Ok(Refinement {
            levels,
            scheme: self.scheme,
            options: self.options,
            edge_polylines: polylines,
        })
    }

    /// Perform uniform refinement, producing refined topology + vertex stencils.
    ///
    /// The returned [`StencilTable`]s map input vertex data to refined
    /// vertex data. Apply them with [`StencilTable::interpolate`] to
    /// any buffer, or use [`RefinementResult::interpolate`] to chain
    /// through all levels in one call.
    pub fn refine_uniform(&self, req: &UniformRefine) -> Result<RefinementResult, KernelError> {
        let refined = self.refine_topology(req)?;
        let level_stencils = refined.vertex_stencils();
        // Pre-fold the per-level face lineage to the base mesh, so adapters
        // get refined-face -> input-face directly instead of re-walking the
        // levels (which `RefinementResult` does not carry).
        let face_root = {
            let base_faces = self.topology.face_vertex_counts.len() as u32;
            let mut root: Vec<u32> = (0..base_faces).collect();
            for step in 0..refined.refinement_steps() {
                let lineage = refined
                    .level_lineage(step)
                    .expect("refinement_steps bounds level_lineage");
                root = lineage
                    .face_parent
                    .iter()
                    .map(|&parent| root[parent as usize])
                    .collect();
            }
            root
        };
        Ok(RefinementResult {
            topology: refined.final_topology().clone(),
            level_stencils,
            lineage: refined.lineage().clone(),
            face_root,
            selected_faces: refined.selected_faces().map(|s| s.to_vec()),
            edge_polylines: refined.edge_polylines().map(|p| p.to_vec()),
            adjacency: refined.adjacency().clone(),
            scheme: self.scheme,
            options: self.options,
        })
    }

    /// For each parent-edge polyline, insert the edge-point vertex at
    /// each split position in the polyline. After refinement, a polyline
    /// segment `[A, B]` where edge A-B was split becomes `[A, edge_pt, B]`.
    fn refine_polylines(polylines: &mut [Vec<u32>], lineage: &LineageMaps, parent_topo: &Mesh) {
        use crate::output::VertexOrigin;

        // Build a map: parent edge index → refined vertex index (the edge-point).
        let edge_point_for_parent: Vec<Option<u32>> = {
            let edge_count = parent_topo.edge_vertices.len();
            let mut map = vec![None; edge_count];
            lineage
                .vertex_origin
                .iter()
                .enumerate()
                .for_each(|(vi, origin)| {
                    if let VertexOrigin::Edge(parent_ei) = *origin {
                        map[parent_ei as usize] = Some(vi as u32);
                    }
                });
            map
        };

        // Build a map: (v0, v1) canonical pair → parent edge index.
        let edge_key = |a: u32, b: u32| if a <= b { (a, b) } else { (b, a) };
        let edge_key_to_idx: rustc_hash::FxHashMap<(u32, u32), usize> = parent_topo
            .edge_vertices
            .iter()
            .enumerate()
            .map(|(ei, &[v0, v1])| (edge_key(v0, v1), ei))
            .collect();

        // For vertex-point vertices, find the mapping from parent vertex → refined vertex.
        let vertex_point_for_parent: Vec<Option<u32>> = {
            let vert_count = parent_topo.vertex_count as usize;
            let mut map = vec![None; vert_count];
            lineage
                .vertex_origin
                .iter()
                .enumerate()
                .for_each(|(vi, origin)| {
                    if let VertexOrigin::Vertex(parent_vi) = *origin {
                        map[parent_vi as usize] = Some(vi as u32);
                    }
                });
            map
        };

        polylines.iter_mut().for_each(|poly| {
            let mut new_poly = Vec::with_capacity(poly.len() * 2);

            poly.windows(2).for_each(|pair| {
                let a = pair[0];
                let b = pair[1];

                let ra = vertex_point_for_parent
                    .get(a as usize)
                    .copied()
                    .flatten()
                    .unwrap_or(a);

                new_poly.push(ra);

                let key = edge_key(a, b);
                if let Some(&ei) = edge_key_to_idx.get(&key) {
                    if let Some(ep) = edge_point_for_parent[ei] {
                        new_poly.push(ep);
                    }
                }
            });

            if let Some(&last) = poly.last() {
                let rl = vertex_point_for_parent
                    .get(last as usize)
                    .copied()
                    .flatten()
                    .unwrap_or(last);
                new_poly.push(rl);
            }

            *poly = new_poly;
        });
    }

    /// Compute per-level face-varying stencil tables for a channel.
    ///
    /// Returns one [`StencilTable`] per refinement level. Use the same
    /// chaining pattern as [`RefinementResult::interpolate`]:
    ///
    /// ```ignore
    /// let fvar_tables = refiner.face_varying_stencils(&req, &channel, mode)?;
    /// let mut uvs = my_uvs.to_vec();
    /// for table in &fvar_tables {
    ///     uvs = table.interpolate(&uvs);
    /// }
    /// ```
    pub fn face_varying_stencils(
        &self,
        req: &UniformRefine,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        match self.scheme {
            Scheme::CatmullClark => self.face_varying_stencils_cc(req, channel, mode),
            Scheme::Loop => self.face_varying_stencils_loop(req, channel, mode),
            Scheme::Sqrt3 => self.face_varying_stencils_sqrt3(req, channel, mode),
            Scheme::DooSabin => self.face_varying_stencils_doo_sabin(req, channel, mode),
        }
    }

    fn face_varying_stencils_loop(
        &self,
        req: &UniformRefine,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        use crate::loop_subdivision::face_varying::fvar_stencils_once;

        let active_sel = self.active_selection(req)?;
        let mut parent = loop_base_level_data(
            &self.topology,
            active_sel,
            &self.options,
            req.selection_boundary_crease,
        )?;
        let mut current_fvar = channel.clone();
        let mut tables = Vec::with_capacity(req.levels.get() as usize);

        for _ in 0..req.levels.get() {
            let child =
                loop_refine_topology_once(&parent, &self.options, req.selection_boundary_crease)?;
            tables.push(fvar_stencils_once(
                &parent,
                &child,
                &current_fvar,
                mode,
                &self.options,
            )?);
            current_fvar = identity_channel(&child.mesh);
            parent = child;
        }

        Ok(tables)
    }

    fn face_varying_stencils_sqrt3(
        &self,
        req: &UniformRefine,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        use crate::sqrt3::face_varying::fvar_stencils_once;

        let active_sel = self.active_selection(req)?;
        let mut parent =
            sqrt3_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;
        let mut current_fvar = channel.clone();
        let mut tables = Vec::with_capacity(req.levels.get() as usize);

        for _ in 0..req.levels.get() {
            let child =
                sqrt3_refine_topology_once(&parent, &self.options, req.selection_boundary_crease)?;
            tables.push(fvar_stencils_once(
                &parent,
                &child,
                &current_fvar,
                mode,
                &self.options,
            )?);
            current_fvar = identity_channel(&child.mesh);
            parent = child;
        }

        Ok(tables)
    }

    fn face_varying_stencils_doo_sabin(
        &self,
        req: &UniformRefine,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        use crate::doo_sabin::face_varying::fvar_stencils_once;

        let active_sel = self.active_selection(req)?;
        let mut parent =
            doo_sabin_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;
        let mut current_fvar = channel.clone();
        let mut tables = Vec::with_capacity(req.levels.get() as usize);

        for _ in 0..req.levels.get() {
            let child = doo_sabin_refine_topology_once(
                &parent,
                &self.options,
                req.selection_boundary_crease,
            )?;
            tables.push(fvar_stencils_once(
                &parent,
                &child,
                &current_fvar,
                mode,
                &self.options,
            )?);
            current_fvar = identity_channel(&child.mesh);
            parent = child;
        }

        Ok(tables)
    }

    fn face_varying_stencils_cc(
        &self,
        req: &UniformRefine,
        channel: &FaceVaryingChannel,
        mode: FaceVaryingInterpolation,
    ) -> Result<Vec<StencilTable>, KernelError> {
        use crate::catmull_clark::face_varying::fvar_stencils_once;

        let active_sel = self.active_selection(req)?;
        let mut current_level =
            cc_base_level_data(&self.topology, active_sel, req.selection_boundary_crease)?;
        let mut current_fvar = channel.clone();
        let mut tables = Vec::with_capacity(req.levels.get() as usize);

        for _ in 0..req.levels.get() {
            let child = cc_refine_topology_once(
                &current_level,
                &self.options,
                req.selection_boundary_crease,
            )?;

            tables.push(fvar_stencils_once(
                &current_level,
                &child,
                &current_fvar,
                mode,
                &self.options,
            )?);

            current_fvar = identity_channel(&child.mesh);
            current_level = child;
        }

        Ok(tables)
    }
}

/// Identity face-varying channel for a refined mesh: every refined corner is
/// its own distinct value. Used to reset the channel between per-level
/// face-varying stencil tables, mirroring the per-vertex stencil chaining.
fn identity_channel(mesh: &Mesh) -> FaceVaryingChannel {
    let n: u32 = mesh.face_vertex_counts.iter().sum();
    FaceVaryingChannel {
        indices: (0..n).collect(),
        value_count: n,
    }
}
