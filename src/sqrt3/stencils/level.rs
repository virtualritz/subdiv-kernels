//! Sqrt3 per-level cached topology state and helpers.
//!
//! For Sqrt3 the `topo` held in [`Sqrt3LevelData`] is the *parent*
//! topology (pre-refinement), used by
//! [`super::refine::vertex_stencils_from_level`] to compute the
//! vertex-point smoothing stencils. After
//! [`super::refine::refine_topology_once`] produces a child, the
//! child's `topo` is the rebuilt **post-flip** topology — the input
//! the next refinement step will read from.

use crate::refiner::LevelDataCommon;
use crate::{Adjacency, KernelError, LineageMaps, Mesh};

use super::super::topology::{Topology, build_topology};

/// Per-level cached topology state for Sqrt3 refinement.
pub(crate) struct Sqrt3LevelData {
    pub(crate) mesh: Mesh,
    pub(crate) topo: Topology,
    pub(crate) face_selected: Vec<bool>,
    pub(crate) effective_edge_creases: Vec<f32>,
    pub(crate) lineage: LineageMaps,
    pub(crate) adjacency: Adjacency,
}

impl LevelDataCommon for Sqrt3LevelData {
    fn mesh(&self) -> &Mesh {
        &self.mesh
    }
    fn lineage(&self) -> &LineageMaps {
        &self.lineage
    }
    fn face_selected(&self) -> &[bool] {
        &self.face_selected
    }
    fn adjacency(&self) -> &Adjacency {
        &self.adjacency
    }
}

/// Compute effective edge creases by merging the selection-boundary
/// crease into any edge straddling selected/unselected faces.
pub(super) fn compute_effective_edge_creases(
    topology: &Topology,
    face_selected: &[bool],
    selection_boundary_crease: f32,
) -> Vec<f32> {
    let edge_count = topology.edge_vertices.len();

    let all_selected = face_selected.iter().all(|&s| s);
    let edge_is_sel_boundary: Vec<bool> = if all_selected {
        vec![false; edge_count]
    } else {
        (0..edge_count)
            .map(|ei| {
                let ef = topology.edge_faces.row(ei);
                let s0 = ef
                    .first()
                    .map(|&fi| face_selected[fi as usize])
                    .unwrap_or(false);
                let s1 = ef
                    .get(1)
                    .map(|&fi| face_selected[fi as usize])
                    .unwrap_or(false);
                s0 != s1
            })
            .collect()
    };

    topology
        .edge_creases
        .iter()
        .copied()
        .enumerate()
        .map(|(ei, base)| {
            if edge_is_sel_boundary[ei] && selection_boundary_crease > 0.0 {
                base.max(selection_boundary_crease)
            } else {
                base
            }
        })
        .collect()
}

/// Create base-level data from an input `Mesh`.
pub(crate) fn base_level_data(
    topo: &Mesh,
    face_selected: Vec<bool>,
    selection_boundary_crease: f32,
) -> Result<Sqrt3LevelData, KernelError> {
    let topology = build_topology(topo)?;
    let face_count = topology.faces.len();

    if face_selected.len() != face_count {
        return Err(KernelError::InvalidTopology(
            "selected-face mask length does not match face count",
        ));
    }

    // Validate triangles for selected faces.
    (0..face_count)
        .filter(|&fi| face_selected[fi])
        .try_for_each(|fi| {
            (topology.faces.row_len(fi) == 3)
                .then_some(())
                .ok_or(KernelError::InvalidTopology(
                    "sqrt3 requires selected faces to be triangles",
                ))
        })?;

    let effective_edge_creases =
        compute_effective_edge_creases(&topology, &face_selected, selection_boundary_crease);

    Ok(Sqrt3LevelData {
        mesh: topo.clone(),
        topo: topology,
        face_selected,
        effective_edge_creases,
        lineage: LineageMaps::default(),
        adjacency: Adjacency {
            face_edges: Vec::new(),
            edge_faces: Vec::new(),
            vert_edge_offsets: Vec::new(),
            vert_edges: Vec::new(),
            vert_face_offsets: Vec::new(),
            vert_faces: Vec::new(),
            edge_is_boundary: Vec::new(),
            vertex_is_boundary: Vec::new(),
        },
    })
}
