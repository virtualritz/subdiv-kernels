//! Loop per-level cached topology state and helpers.
//!
//! [`LoopLevelData`] holds the per-level topology, selection mask, and
//! crease/corner scratch used by [`super::refine`]. Level 0 is built
//! by [`base_level_data`]; subsequent levels are produced by
//! [`super::refine::refine_topology_once`].

use crate::refiner::LevelDataCommon;
use crate::{Adjacency, KernelError, LineageMaps, Mesh, SchemeOptions};

use super::super::sharpness::auto_corner_sharpness_from_creases;
use super::super::topology::{Topology, build_topology_from};

/// Per-level cached topology state for Loop refinement.
pub(crate) struct LoopLevelData {
    pub(crate) mesh: Mesh,
    pub(crate) topo: Topology,
    pub(crate) face_selected: Vec<bool>,
    pub(crate) effective_edge_creases: Vec<f32>,
    pub(crate) edge_has_selected_face: Vec<bool>,
    pub(crate) effective_vertex_corners: Vec<f32>,
    pub(crate) lineage: LineageMaps,
    pub(crate) adjacency: Adjacency,
}

impl LevelDataCommon for LoopLevelData {
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

/// Compute derived edge selection and crease data from a topology +
/// selection. Loop's edge_faces is a flat Vec<[u32; 2]> where
/// `[_, u32::MAX]` marks a boundary edge.
pub(super) fn compute_edge_data(
    topology: &Topology,
    face_selected: &[bool],
    selection_boundary_crease: f32,
) -> (Vec<bool>, Vec<f32>) {
    let edge_count = topology.edge_vertices.len();

    let edge_has_selected_face: Vec<bool> = (0..edge_count)
        .map(|ei| {
            let [f0, f1] = topology.edge_faces[ei];
            face_selected[f0 as usize] || (f1 != u32::MAX && face_selected[f1 as usize])
        })
        .collect();

    let all_selected = face_selected.iter().all(|&s| s);
    let edge_is_sel_boundary: Vec<bool> = if all_selected {
        vec![false; edge_count]
    } else {
        topology
            .edge_faces
            .iter()
            .map(|faces| {
                let s0 = face_selected[faces[0] as usize];
                let s1 = if faces[1] != u32::MAX {
                    face_selected[faces[1] as usize]
                } else {
                    false
                };
                s0 != s1
            })
            .collect()
    };

    let effective_edge_creases: Vec<f32> = topology
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

/// Compute effective vertex corners (explicit + auto-corner promotion
/// from incident creases).
pub(super) fn compute_effective_vertex_corners(
    mesh: &Mesh,
    topology: &Topology,
    effective_edge_creases: &[f32],
    options: &SchemeOptions,
) -> Vec<f32> {
    let vertex_count = mesh.vertex_count as usize;
    (0..vertex_count)
        .map(|vi| {
            let explicit = mesh.vertex_corners[vi];
            if !options.auto_corner {
                explicit
            } else {
                let auto = auto_corner_sharpness_from_creases(
                    topology
                        .vertex_edges
                        .row(vi)
                        .iter()
                        .map(|&ei| effective_edge_creases[ei as usize]),
                );
                explicit.max(auto)
            }
        })
        .collect()
}

/// Create base-level data from an input `Mesh`.
pub(crate) fn base_level_data(
    topo: &Mesh,
    face_selected: Vec<bool>,
    options: &SchemeOptions,
    selection_boundary_crease: f32,
) -> Result<LoopLevelData, KernelError> {
    let topology = build_topology_from(topo)?;
    let face_count = topology.faces.len();

    if face_selected.len() != face_count {
        return Err(KernelError::InvalidTopology(
            "selected-face mask length does not match current face count",
        ));
    }

    // Validate triangles for selected faces.
    (0..face_count)
        .filter(|&fi| face_selected[fi])
        .try_for_each(|fi| {
            (topology.faces.row_len(fi) == 3)
                .then_some(())
                .ok_or(KernelError::InvalidTopology(
                    "loop subdivision requires selected faces to be triangles",
                ))
        })?;

    let (edge_has_selected_face, effective_edge_creases) =
        compute_edge_data(&topology, &face_selected, selection_boundary_crease);
    let effective_vertex_corners =
        compute_effective_vertex_corners(topo, &topology, &effective_edge_creases, options);

    Ok(LoopLevelData {
        mesh: topo.clone(),
        topo: topology,
        face_selected,
        effective_edge_creases,
        edge_has_selected_face,
        effective_vertex_corners,
        lineage: LineageMaps::default(),
        adjacency: Adjacency {
            face_edges: Vec::new(),
            edge_faces: Vec::new(),
            vertex_edge_offsets: Vec::new(),
            vertex_edges: Vec::new(),
            vertex_face_offsets: Vec::new(),
            vertex_faces: Vec::new(),
            edge_is_boundary: Vec::new(),
            vertex_is_boundary: Vec::new(),
        },
    })
}
