//! Doo-Sabin topology refinement and stencil extraction.
//!
//! Doo-Sabin inserts one face-vertex point per `(face, corner)` pair
//! and rebuilds topology as:
//!
//! - F-faces: one per parent face, connecting its new face-vertex points.
//! - E-faces: one quad per non-boundary parent edge.
//! - V-faces: one fan per non-boundary parent vertex, connecting the
//!   face-vertex points around that vertex in order.
//!
//! [`refine_topology_once`] builds all three and rebuilds topology.
//! [`vertex_stencils_from_level`] reads the parent topology and
//! produces the per-face-vertex stencil table without touching the
//! refined mesh.

use crate::sharpness::decay_sharpness;
use crate::{Adjacency, KernelError, LineageMaps, Mesh, SchemeOptions, StencilTable, VertexOrigin};

use super::super::topology::{Topology, build_topology, edge_key};
use super::level::{DooSabinLevelData, compute_effective_edge_creases};
use super::points::face_vertex_stencils;
use super::sparse::pack;

/// Build one refinement level's topology from the parent level.
///
/// Does NOT compute vertex stencils — use [`vertex_stencils_from_level`].
pub(crate) fn refine_topology_once(
    parent: &DooSabinLevelData,
    options: &SchemeOptions,
    selection_boundary_crease: f32,
) -> Result<DooSabinLevelData, KernelError> {
    let topology = &parent.topo;
    let face_selected = &parent.face_selected;
    let effective_edge_creases = &parent.effective_edge_creases;
    let topo_mesh = &parent.mesh;
    let vertex_count = topo_mesh.vertex_count as usize;
    let face_count = topology.faces.len();
    let edge_count = topology.edge_vertices.len();

    // ── Face-vertex index mapping ─────────────────────────────────────
    let face_vertex_offset: Vec<u32> = {
        let mut offsets = Vec::with_capacity(face_count + 1);
        let mut acc = 0u32;
        for fi in 0..face_count {
            offsets.push(acc);
            acc += topology.faces.row_len(fi) as u32;
        }
        offsets.push(acc);
        offsets
    };
    let total_verts = *face_vertex_offset.last().unwrap_or(&0);
    let fv_idx = |fi: usize, corner: usize| -> u32 { face_vertex_offset[fi] + corner as u32 };

    // ── Step 2: F-face topology ───────────────────────────────────────
    let mut new_faces: Vec<Vec<u32>> = Vec::new();
    let mut face_parent: Vec<u32> = Vec::new();
    let mut new_face_selected: Vec<bool> = Vec::new();
    let mut crease_edge_parent: Vec<((u32, u32), usize)> = Vec::new();

    for fi in 0..face_count {
        let n = topology.faces.row_len(fi);
        let f_face: Vec<u32> = (0..n).map(|c| fv_idx(fi, c)).collect();
        new_faces.push(f_face);
        face_parent.push(fi as u32);
        new_face_selected.push(face_selected[fi]);
    }

    // ── Step 3: E-face topology ───────────────────────────────────────
    for ei in 0..edge_count {
        if topology.edge_is_boundary[ei] {
            continue;
        }

        let ef = topology.edge_faces.row(ei);
        if ef.len() != 2 {
            continue;
        }

        let f0 = ef[0] as usize;
        let f1 = ef[1] as usize;
        let [v0, v1] = topology.edge_vertices[ei];

        let c_v0_f0 = corner_of(topology.faces.row(f0), v0);
        let c_v1_f0 = corner_of(topology.faces.row(f0), v1);
        let c_v0_f1 = corner_of(topology.faces.row(f1), v0);
        let c_v1_f1 = corner_of(topology.faces.row(f1), v1);

        if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (c_v0_f0, c_v1_f0, c_v0_f1, c_v1_f1) {
            let e_face = vec![
                fv_idx(f0, c10),
                fv_idx(f0, c00),
                fv_idx(f1, c01),
                fv_idx(f1, c11),
            ];

            new_faces.push(e_face);
            face_parent.push(f0 as u32);
            new_face_selected.push(face_selected[f0] || face_selected[f1]);

            if effective_edge_creases[ei] > 0.0 {
                crease_edge_parent.push((edge_key(fv_idx(f0, c00), fv_idx(f0, c10)), ei));
                crease_edge_parent.push((edge_key(fv_idx(f1, c01), fv_idx(f1, c11)), ei));
            }
        }
    }

    // ── Step 4: V-face topology ───────────────────────────────────────
    for vi in 0..vertex_count {
        if topology.vertex_is_boundary[vi] {
            continue;
        }

        let incident_faces = topology.vertex_faces.row(vi);
        if incident_faces.len() < 3 {
            continue;
        }

        let ordered = order_faces_around_vertex(vi as u32, incident_faces, topology);

        let v_face: Vec<u32> = ordered
            .iter()
            .filter_map(|&fi| corner_of(topology.faces.row(fi), vi as u32).map(|c| fv_idx(fi, c)))
            .collect();

        if v_face.len() >= 3 {
            new_faces.push(v_face);
            face_parent.push(ordered[0] as u32);
            new_face_selected.push(ordered.iter().any(|&fi| face_selected[fi]));
        }
    }

    // ── Step 5: Build refined Mesh + rebuild topology ─────────
    let mut fvc = Vec::with_capacity(new_faces.len());
    let mut fvi = Vec::new();
    new_faces.iter().for_each(|f| {
        fvc.push(f.len() as u32);
        fvi.extend(f.iter().copied());
    });

    let refined_topology = Mesh {
        vertex_count: total_verts,
        face_vertex_counts: fvc,
        face_vertex_indices: fvi,
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; total_verts as usize],
    };

    let refined_analysis = build_topology(&refined_topology)?;

    // Propagate edge creases to E-face edges.
    let mut refined_edge_creases = vec![0.0f32; refined_analysis.edge_vertices.len()];
    let mut edge_parent_map = vec![u32::MAX; refined_analysis.edge_vertices.len()];

    crease_edge_parent.iter().for_each(|&(key, parent_ei)| {
        if let Some(&nei) = refined_analysis.edge_key_to_index.get(&key) {
            let crease = effective_edge_creases[parent_ei];
            let child_crease = if options.crease_normalize {
                crease
            } else {
                decay_sharpness(crease)
            };
            if child_crease > 0.0 {
                refined_edge_creases[nei] = child_crease;
                edge_parent_map[nei] = parent_ei as u32;
            }
        }
    });

    // Propagate corner sharpness to all face-vertex points of each vertex.
    let mut refined_corners = vec![0.0f32; total_verts as usize];
    for vi in 0..vertex_count {
        let corner = topo_mesh.vertex_corners[vi];
        if corner <= 0.0 {
            continue;
        }
        let child_corner = if options.corner_normalize {
            corner
        } else {
            decay_sharpness(corner)
        };
        if child_corner > 0.0 {
            topology.vertex_faces.row(vi).iter().for_each(|&fi| {
                let fi = fi as usize;
                if let Some(c) = corner_of(topology.faces.row(fi), vi as u32) {
                    refined_corners[fv_idx(fi, c) as usize] = child_corner;
                }
            });
        }
    }

    let refined_mesh = Mesh {
        vertex_count: total_verts,
        face_vertex_counts: refined_topology.face_vertex_counts,
        face_vertex_indices: refined_topology.face_vertex_indices,
        edge_vertices: refined_analysis.edge_vertices.clone(),
        edge_creases: refined_edge_creases,
        vertex_corners: refined_corners,
    };

    // ── Lineage ───────────────────────────────────────────────────────
    let vertex_origin: Vec<VertexOrigin> = (0..face_count)
        .flat_map(|fi| {
            std::iter::repeat_n(VertexOrigin::Face(fi as u32), topology.faces.row_len(fi))
        })
        .collect();

    let lineage = LineageMaps {
        vertex_origin,
        face_parent,
        edge_parent: edge_parent_map,
    };

    // ── Adjacency export ──────────────────────────────────────────────
    // DooSabin Topology doesn't store face_edges; compute flat from
    // faces + edge map.
    let edge_index = &refined_analysis.edge_key_to_index;
    let refined_face_count = refined_analysis.faces.len();
    let face_edges_flat: Vec<u32> = (0..refined_face_count)
        .flat_map(|fi| {
            let face = refined_analysis.faces.row(fi);
            let n = face.len();
            (0..n).map(move |i| {
                let key = edge_key(face[i], face[(i + 1) % n]);
                edge_index[&key] as u32
            })
        })
        .collect();

    let refined_edge_count = refined_analysis.edge_faces.len();
    let adj_edge_faces: Vec<[u32; 2]> = (0..refined_edge_count)
        .map(|ei| {
            let ef = refined_analysis.edge_faces.row(ei);
            match ef.len() {
                0 => [u32::MAX, u32::MAX],
                1 => [ef[0], u32::MAX],
                _ => [ef[0], ef[1]],
            }
        })
        .collect();

    let adj_edge_is_boundary: Vec<bool> =
        adj_edge_faces.iter().map(|ef| ef[1] == u32::MAX).collect();

    let (vert_edge_offsets, vert_edges) = refined_analysis.vertex_edges.clone().into_parts();

    let vert_count = vert_edge_offsets.len().saturating_sub(1);
    let mut adj_vertex_is_boundary = vec![false; vert_count];
    for vi in 0..vert_count {
        let s = vert_edge_offsets[vi] as usize;
        let e = vert_edge_offsets[vi + 1] as usize;
        if vert_edges[s..e]
            .iter()
            .any(|&ei| adj_edge_is_boundary[ei as usize])
        {
            adj_vertex_is_boundary[vi] = true;
        }
    }

    let (vert_face_offsets, vert_faces) = refined_analysis.vertex_faces.clone().into_parts();

    let adjacency = Adjacency {
        face_edges: face_edges_flat,
        edge_faces: adj_edge_faces,
        vert_edge_offsets,
        vert_edges,
        vert_face_offsets,
        vert_faces,
        edge_is_boundary: adj_edge_is_boundary,
        vertex_is_boundary: adj_vertex_is_boundary,
    };

    // Precompute the child's effective edge creases so the next level
    // can run refine_topology_once without re-deriving them.
    let child_effective_edge_creases = compute_effective_edge_creases(
        &refined_analysis,
        &new_face_selected,
        selection_boundary_crease,
    );

    Ok(DooSabinLevelData {
        mesh: refined_mesh,
        topo: refined_analysis,
        face_selected: new_face_selected,
        effective_edge_creases: child_effective_edge_creases,
        lineage,
        adjacency,
    })
}

/// Compute per-level vertex stencils from a cached parent level.
pub(crate) fn vertex_stencils_from_level(
    parent: &DooSabinLevelData,
    options: &SchemeOptions,
) -> StencilTable {
    let all_stencils = face_vertex_stencils(
        &parent.topo,
        &parent.mesh,
        &parent.face_selected,
        &parent.effective_edge_creases,
        options,
    );
    pack(&all_stencils)
}

/// Find the corner index of vertex `v` in face vertex list.
fn corner_of(face: &[u32], v: u32) -> Option<usize> {
    face.iter().position(|&fv| fv == v)
}

/// Order incident faces around a vertex by walking edge-face adjacency.
///
/// At each face, vertex `vi` sits between two edges. We enter the face
/// from one edge and exit via the other, so the full fan is traversed
/// even when multiple faces share the same edge orientation.
fn order_faces_around_vertex(vi: u32, incident_faces: &[u32], topology: &Topology) -> Vec<usize> {
    if incident_faces.is_empty() {
        return Vec::new();
    }

    let mut ordered = Vec::with_capacity(incident_faces.len());
    let first = incident_faces[0] as usize;
    ordered.push(first);

    let fv = topology.faces.row(first);
    let vi_pos = match corner_of(fv, vi) {
        Some(p) => p,
        None => return ordered,
    };
    let n = fv.len();
    let prev_v = fv[(vi_pos + n - 1) % n];
    let mut entry_edge = edge_key(vi, prev_v);

    let mut current = first;

    for _ in 1..incident_faces.len() {
        let fv = topology.faces.row(current);
        let vi_pos = match corner_of(fv, vi) {
            Some(p) => p,
            None => break,
        };
        let n = fv.len();
        let prev_v = fv[(vi_pos + n - 1) % n];
        let next_v = fv[(vi_pos + 1) % n];

        let prev_key = edge_key(vi, prev_v);
        let next_key = edge_key(vi, next_v);

        let exit_key = if prev_key == entry_edge {
            next_key
        } else {
            prev_key
        };

        let next = topology.edge_key_to_index.get(&exit_key).and_then(|&ei| {
            topology
                .edge_faces
                .row(ei)
                .iter()
                .find(|&&fi| fi as usize != current)
                .map(|&fi| fi as usize)
        });

        match next {
            Some(fi) => {
                ordered.push(fi);
                entry_edge = exit_key;
                current = fi;
            }
            None => break,
        }
    }

    ordered
}
