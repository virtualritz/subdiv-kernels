//! Loop topology refinement and stencil extraction.

use crate::{Adjacency, KernelError, LineageMaps, Mesh, SchemeOptions, StencilTable, VertexOrigin};

use super::super::sharpness::decay_sharpness;
use super::super::topology::{build_loop_refined_uniform, build_topology_from, edge_key};
use super::level::{LoopLevelData, compute_edge_data, compute_effective_vertex_corners};
use super::points::{edge_point_stencil, vertex_point_stencil};
use super::sparse::{Sparse, pack};

/// Build one refinement level's topology from the parent level.
///
/// Does NOT compute vertex stencils — use [`vertex_stencils_from_level`].
pub(crate) fn refine_topology_once(
    parent: &LoopLevelData,
    options: &SchemeOptions,
    selection_boundary_crease: f32,
) -> Result<LoopLevelData, KernelError> {
    let topology = &parent.topo;
    let face_selected = &parent.face_selected;
    let effective_edge_creases = &parent.effective_edge_creases;
    let vertex_count = parent.mesh.vertex_count as usize;
    let edge_count = topology.edge_vertices.len();
    let face_count = topology.faces.len();

    // ── Build refined faces + child topology ──────────────────────────
    // Loop child indices are fixed: ep(ei)=ei, vp(vi)=E+vi.
    let ep_idx = |ei: usize| ei as u32;
    let vp_idx = |vi: usize| (edge_count + vi) as u32;

    let total_verts = (edge_count + vertex_count) as u32;

    // Uniform (all faces selected) levels build the regular child topology
    // analytically (no per-corner hash). Mixed selection (adaptive) keeps the
    // generic builder, which handles the vertex/vertex child edges unselected
    // faces introduce. Either way `refined_topo` carries edge_creases == 0.0;
    // the loop below overwrites it, and it is reused directly as the child
    // topology (no second `build_topology_from`).
    let (face_vertex_counts, face_vertex_indices, face_parent, new_face_selected, mut refined_topo) =
        if face_selected.iter().all(|&s| s) {
            let r = build_loop_refined_uniform(topology, vertex_count);
            let child_face_count = r.face_vertex_counts.len();
            (
                r.face_vertex_counts,
                r.face_vertex_indices,
                r.face_parent,
                vec![true; child_face_count],
                r.topology,
            )
        } else {
            let mut new_faces: Vec<Vec<u32>> = Vec::new();
            let mut face_parent = Vec::new();
            let mut new_face_selected = Vec::new();

            (0..face_count).for_each(|fi| {
                let face_verts = topology.faces.row(fi);
                if face_selected[fi] {
                    let v0 = face_verts[0];
                    let v1 = face_verts[1];
                    let v2 = face_verts[2];

                    let e01 = topology.face_edges.get(fi, 0);
                    let e12 = topology.face_edges.get(fi, 1);
                    let e20 = topology.face_edges.get(fi, 2);

                    let nv0 = vp_idx(v0 as usize);
                    let nv1 = vp_idx(v1 as usize);
                    let nv2 = vp_idx(v2 as usize);

                    [
                        [nv0, ep_idx(e01 as usize), ep_idx(e20 as usize)],
                        [nv1, ep_idx(e12 as usize), ep_idx(e01 as usize)],
                        [nv2, ep_idx(e20 as usize), ep_idx(e12 as usize)],
                        [
                            ep_idx(e01 as usize),
                            ep_idx(e12 as usize),
                            ep_idx(e20 as usize),
                        ],
                    ]
                    .into_iter()
                    .for_each(|tri| {
                        new_faces.push(tri.to_vec());
                        face_parent.push(fi as u32);
                        new_face_selected.push(true);
                    });
                } else {
                    new_faces.push(face_verts.iter().map(|&v| vp_idx(v as usize)).collect());
                    face_parent.push(fi as u32);
                    new_face_selected.push(false);
                }
            });

            let mut fvc = Vec::with_capacity(new_faces.len());
            let mut fvi = Vec::new();
            new_faces.iter().for_each(|f| {
                fvc.push(f.len() as u32);
                fvi.extend(f.iter().copied());
            });

            let provisional = Mesh {
                vertex_count: total_verts,
                face_vertex_counts: fvc,
                face_vertex_indices: fvi,
                edge_vertices: Vec::new(),
                edge_creases: Vec::new(),
                vertex_corners: vec![0.0; total_verts as usize],
            };
            let topo = build_topology_from(&provisional)?;
            (
                provisional.face_vertex_counts,
                provisional.face_vertex_indices,
                face_parent,
                new_face_selected,
                topo,
            )
        };

    // Propagate edge creases directly into refined_topo.edge_creases (it starts
    // at 0.0), so refined_topo can serve as the child topology unchanged.
    let mut edge_parent_map = vec![u32::MAX; refined_topo.edge_vertices.len()];

    for (ei, [v0, v1]) in topology.edge_vertices.iter().copied().enumerate() {
        if effective_edge_creases[ei] <= 0.0 {
            continue;
        }
        let child_crease = decay_sharpness(effective_edge_creases[ei]);
        if child_crease <= 0.0 {
            continue;
        }

        let edge_point = ep_idx(ei);
        let key_a = edge_key(vp_idx(v0 as usize), edge_point);
        let key_b = edge_key(vp_idx(v1 as usize), edge_point);

        if let Some(&nei) = refined_topo.edge_key_to_index.get(&key_a) {
            refined_topo.edge_creases[nei] = child_crease;
            edge_parent_map[nei] = ei as u32;
        }
        if let Some(&nei) = refined_topo.edge_key_to_index.get(&key_b) {
            refined_topo.edge_creases[nei] = child_crease;
            edge_parent_map[nei] = ei as u32;
        }
    }

    // Propagate vertex corners: edge points get 0.0, vertex points keep theirs.
    let refined_corners: Vec<f32> = std::iter::repeat(0.0)
        .take(edge_count)
        .chain(parent.mesh.vertex_corners.iter().copied())
        .collect();

    let refined_mesh = Mesh {
        vertex_count: total_verts,
        face_vertex_counts,
        face_vertex_indices,
        edge_vertices: refined_topo.edge_vertices.clone(),
        edge_creases: refined_topo.edge_creases.clone(),
        vertex_corners: refined_corners,
    };

    // Lineage
    let mut vertex_origin = Vec::with_capacity(total_verts as usize);
    vertex_origin.extend((0..edge_count).map(|i| VertexOrigin::Edge(i as u32)));
    vertex_origin.extend((0..vertex_count).map(|i| VertexOrigin::Vertex(i as u32)));

    let lineage = LineageMaps {
        vertex_origin,
        face_parent,
        edge_parent: edge_parent_map,
    };

    // Adjacency export for the refined mesh (consumed by adapters).
    let (face_edge_offsets, face_edges_flat) = refined_topo.face_edges.clone().into_parts();
    let (vertex_edge_offsets, vertex_edges) = refined_topo.vertex_edges.clone().into_parts();
    let (vertex_face_offsets, vertex_faces) = refined_topo.vertex_faces.clone().into_parts();
    let _ = face_edge_offsets;
    let adjacency = Adjacency {
        face_edges: face_edges_flat,
        edge_faces: refined_topo.edge_faces.clone(),
        vertex_edge_offsets,
        vertex_edges,
        vertex_face_offsets,
        vertex_faces,
        edge_is_boundary: refined_topo.edge_is_boundary.clone(),
        vertex_is_boundary: refined_topo.vertex_is_boundary.clone(),
    };

    // `refined_topo` already carries the propagated creases, so it serves as
    // the child topology directly — no second `build_topology_from`.
    let (child_edge_has_selected_face, child_effective_edge_creases) =
        compute_edge_data(&refined_topo, &new_face_selected, selection_boundary_crease);
    let child_effective_vertex_corners = compute_effective_vertex_corners(
        &refined_mesh,
        &refined_topo,
        &child_effective_edge_creases,
        options,
    );

    Ok(LoopLevelData {
        mesh: refined_mesh,
        topo: refined_topo,
        face_selected: new_face_selected,
        effective_edge_creases: child_effective_edge_creases,
        edge_has_selected_face: child_edge_has_selected_face,
        effective_vertex_corners: child_effective_vertex_corners,
        lineage,
        adjacency,
    })
}

/// Compute per-level vertex stencils from a cached parent level.
pub(crate) fn vertex_stencils_from_level(
    parent: &LoopLevelData,
    _options: &SchemeOptions,
) -> StencilTable {
    let topology = &parent.topo;
    let vertex_count = parent.mesh.vertex_count as usize;
    let edge_count = topology.edge_vertices.len();

    // Output order: [edge_points(0..E), vertex_points(E..E+V)]
    let ep_stencils: Vec<Sparse> = (0..edge_count)
        .map(|ei| {
            if parent.edge_has_selected_face[ei] {
                edge_point_stencil(ei, topology, &parent.effective_edge_creases)
            } else {
                // Unselected: identity-ish (not reachable in practice since
                // these edges won't appear in refined faces, but we need a
                // placeholder stencil for the output index).
                let [v0, v1] = topology.edge_vertices[ei];
                vec![(v0, 0.5), (v1, 0.5)]
            }
        })
        .collect();

    let vp_stencils: Vec<Sparse> = (0..vertex_count)
        .map(|vi| {
            vertex_point_stencil(
                vi,
                topology,
                &parent.face_selected,
                &parent.effective_edge_creases,
                &parent.effective_vertex_corners,
            )
        })
        .collect();

    let mut all_stencils = Vec::with_capacity(edge_count + vertex_count);
    all_stencils.extend(ep_stencils);
    all_stencils.extend(vp_stencils);

    pack(&all_stencils)
}
