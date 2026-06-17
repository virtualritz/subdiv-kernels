//! Catmull-Clark per-level cached topology state and helpers.
//!
//! [`CcLevelData`] is the per-level record stored by the refiner. It
//! holds the topology, selection mask, and crease/corner evaluation
//! scratch needed to compute stencils without re-walking the mesh.
//! Level 0 is built by [`base_level_data`]; subsequent levels are
//! produced by [`super::refine::refine_topology_once`].

use crate::refiner::LevelDataCommon;
use crate::{Adjacency, KernelError, LineageMaps, Mesh};

use super::super::topology::{Topology, build_topology_from};

/// Per-level cached topology state for CC refinement.
///
/// Created by [`base_level_data`] (level 0) or
/// [`super::refine::refine_topology_once`] (subsequent levels).
/// Contains everything needed to compute stencils and FVar
/// interpolation without rebuilding topology.
pub(crate) struct CcLevelData {
    pub(crate) mesh: Mesh,
    pub(crate) topo: Topology,
    pub(crate) face_selected: Vec<bool>,
    pub(crate) effective_edge_creases: Vec<f32>,
    pub(crate) edge_has_selected_face: Vec<bool>,
    pub(crate) lineage: LineageMaps,
    pub(crate) adjacency: Adjacency,
}

impl LevelDataCommon for CcLevelData {
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

/// Compute derived edge selection and crease data from a topology + selection.
pub(super) fn compute_edge_data(
    topo: &Topology,
    face_selected: &[bool],
    selection_boundary_crease: f32,
) -> (Vec<bool>, Vec<f32>) {
    let edge_count = topo.edge_vertices.len();

    let edge_has_selected_face: Vec<bool> = (0..edge_count)
        .map(|ei| {
            topo.edge_faces
                .row(ei)
                .iter()
                .any(|&fi| face_selected[fi as usize])
        })
        .collect();

    let all_selected = face_selected.iter().all(|&s| s);
    let edge_is_sel_boundary: Vec<bool> = if all_selected {
        vec![false; edge_count]
    } else {
        (0..edge_count)
            .map(|ei| {
                let row = topo.edge_faces.row(ei);
                let s0 = row
                    .first()
                    .map(|&fi| face_selected[fi as usize])
                    .unwrap_or(false);
                let s1 = row
                    .get(1)
                    .map(|&fi| face_selected[fi as usize])
                    .unwrap_or(false);
                s0 != s1
            })
            .collect()
    };

    let effective_edge_creases: Vec<f32> = topo
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
        .collect();

    (edge_has_selected_face, effective_edge_creases)
}

/// Build adjacency export from a refined `Topology`.
pub(super) fn adjacency_from_topology(refined_topo: &Topology) -> Adjacency {
    let (_, face_edges_flat) = refined_topo.face_edges.clone().into_parts();

    let refined_edge_count = refined_topo.edge_vertices.len();
    let edge_faces_pairs: Vec<[u32; 2]> = (0..refined_edge_count)
        .map(|ei| {
            let row = refined_topo.edge_faces.row(ei);
            match row.len() {
                0 => [u32::MAX, u32::MAX],
                1 => [row[0], u32::MAX],
                _ => [row[0], row[1]],
            }
        })
        .collect();

    let edge_is_boundary: Vec<bool> = edge_faces_pairs
        .iter()
        .map(|ef| ef[1] == u32::MAX)
        .collect();

    let refined_vert_count = refined_topo.vertex_edges.len();
    let mut vertex_is_boundary = vec![false; refined_vert_count];
    for vi in 0..refined_vert_count {
        if refined_topo
            .vertex_edges
            .row(vi)
            .iter()
            .any(|&ei| edge_is_boundary[ei as usize])
        {
            vertex_is_boundary[vi] = true;
        }
    }

    let (vert_edge_offsets, vert_edges) = refined_topo.vertex_edges.clone().into_parts();
    let (vert_face_offsets, vert_faces) = refined_topo.vertex_faces.clone().into_parts();

    Adjacency {
        face_edges: face_edges_flat,
        edge_faces: edge_faces_pairs,
        vert_edge_offsets,
        vert_edges,
        vert_face_offsets,
        vert_faces,
        edge_is_boundary,
        vertex_is_boundary,
    }
}

/// Create base-level data from input topology.
pub(crate) fn base_level_data(
    topo: &Mesh,
    face_selected: Vec<bool>,
    selection_boundary_crease: f32,
) -> Result<CcLevelData, KernelError> {
    let topology = build_topology_from(topo)?;
    let face_count = topology.faces.len();

    if face_selected.len() != face_count {
        return Err(KernelError::InvalidTopology(
            "selected-face mask length does not match current face count",
        ));
    }

    if topo.vertex_count == 0 || face_count == 0 {
        return Err(KernelError::InvalidTopology(
            "empty topology is not refinable",
        ));
    }

    let (edge_has_selected_face, effective_edge_creases) =
        compute_edge_data(&topology, &face_selected, selection_boundary_crease);

    Ok(CcLevelData {
        mesh: topo.clone(),
        topo: topology,
        face_selected,
        effective_edge_creases,
        edge_has_selected_face,
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
