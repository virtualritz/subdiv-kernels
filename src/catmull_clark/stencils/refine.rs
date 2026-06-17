//! Catmull-Clark topology refinement and stencil extraction.
//!
//! [`refine_topology_once`] produces a child [`CcLevelData`] from a
//! parent, running the full per-level topology rebuild + selection and
//! crease propagation. [`vertex_stencils_from_level`] consumes a
//! [`CcLevelData`] and emits a [`StencilTable`] for that level's
//! vertex-data interpolation without touching the topology — this is
//! how the two-phase cache lets callers avoid rebuilding topology
//! once per data channel.

use crate::{LineageMaps, Mesh, SchemeOptions, StencilTable, VertexOrigin};

use super::super::sharpness::{child_edge_creases, decay_corner_sharpness};
use super::super::topology::{build_cc_refined_uniform, build_topology_from, edge_key};
use super::level::{CcLevelData, adjacency_from_topology, compute_edge_data};
use super::points::{edge_point_stencils, face_point_stencils, vertex_point_stencil};
use super::sparse::{Sparse, pack};

/// Build one refinement level's topology from the parent level.
///
/// Does NOT compute vertex stencils — use [`vertex_stencils_from_level`].
pub(crate) fn refine_topology_once(
    parent: &CcLevelData,
    options: &SchemeOptions,
    selection_boundary_crease: f32,
) -> Result<CcLevelData, crate::KernelError> {
    let topology = &parent.topo;
    let face_selected = &parent.face_selected;
    let effective_edge_creases = &parent.effective_edge_creases;

    let vertex_count = parent.mesh.vertex_count as usize;
    let edge_count = topology.edge_vertices.len();
    let face_count = topology.faces.len();

    // ── Build child faces + child topology ────────────────────────────
    // CC child element indices are fixed: fp(fi)=fi, ep(ei)=F+ei,
    // vp(vi)=F+E+vi. `ep_idx`/`vp_idx` are also used in crease propagation.
    let ep_idx = |ei: usize| (face_count + ei) as u32;
    let vp_idx = |vi: usize| (face_count + edge_count + vi) as u32;

    let total_verts = (face_count + edge_count + vertex_count) as u32;

    // Uniform (all faces selected) levels build the regular child topology
    // analytically: faces authored straight into CSR and child edges
    // enumerated from the parent with direct-index dedup — no per-corner hash.
    // Mixed selection (adaptive) keeps the generic builder, which also handles
    // the vertex/vertex child edges unselected faces introduce. Either way
    // `refined_topo` carries edge_creases == 0.0; the loop below overwrites it
    // with the propagated values.
    let (face_vertex_counts, face_vertex_indices, face_parent, new_face_selected, mut refined_topo) =
        if face_selected.iter().all(|&s| s) {
            let r = build_cc_refined_uniform(topology, vertex_count);
            let child_face_count = r.face_vertex_counts.len();
            (
                r.face_vertex_counts,
                r.face_vertex_indices,
                r.face_parent,
                vec![true; child_face_count],
                r.topology,
            )
        } else {
            let fp_idx = |fi: usize| fi as u32;
            let mut new_faces: Vec<Vec<u32>> = Vec::new();
            let mut face_parent = Vec::new();
            let mut new_face_selected = Vec::new();

            (0..face_count).for_each(|fi| {
                let face_verts = topology.faces.row(fi);
                if face_selected[fi] {
                    let cc = face_verts.len();
                    (0..cc).for_each(|corner| {
                        let cur_v = face_verts[corner] as usize;
                        let cur_e = topology.face_edges.get(fi, corner) as usize;
                        let prev_e = topology.face_edges.get(fi, (corner + cc - 1) % cc) as usize;
                        new_faces.push(vec![
                            fp_idx(fi),
                            ep_idx(prev_e),
                            vp_idx(cur_v),
                            ep_idx(cur_e),
                        ]);
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

    // Propagate edge creases directly into refined_topo.edge_creases.
    let refined_edge_count = refined_topo.edge_vertices.len();
    let mut edge_parent_map = vec![u32::MAX; refined_edge_count];

    topology
        .edge_vertices
        .iter()
        .copied()
        .enumerate()
        .filter(|(ei, _)| effective_edge_creases[*ei] > 0.0)
        .for_each(|(ei, [v0, v1])| {
            let (c0, c1) = child_edge_creases(
                ei,
                topology,
                effective_edge_creases,
                options.crease_normalize,
                options.crease_computation,
                options.corner_rule,
            );
            let key0 = edge_key(vp_idx(v0 as usize), ep_idx(ei));
            let key1 = edge_key(vp_idx(v1 as usize), ep_idx(ei));
            if let Some(&nei) = refined_topo.edge_key_to_index.get(&key0) {
                if c0 > 0.0 {
                    refined_topo.edge_creases[nei] = c0;
                }
                edge_parent_map[nei] = ei as u32;
            }
            if let Some(&nei) = refined_topo.edge_key_to_index.get(&key1) {
                if c1 > 0.0 {
                    refined_topo.edge_creases[nei] = c1;
                }
                edge_parent_map[nei] = ei as u32;
            }
        });

    // Propagate vertex corners
    let refined_corners: Vec<f32> = std::iter::repeat(0.0)
        .take(face_count + edge_count)
        .chain(parent.mesh.vertex_corners.iter().map(|&c| {
            if options.corner_normalize {
                c
            } else {
                decay_corner_sharpness(c, options.corner_rule)
            }
        }))
        .collect();

    // `refined_mesh` is the Mesh returned in `CcLevelData.mesh`
    // for downstream consumers (stencil interpolation, FinalParts).
    // `edge_vertices` and `edge_creases` are cloned from `refined_topo`
    // — a few KB per level. Sharing via Arc would require changing the
    // public `Mesh` field type, which is deferred.
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
    vertex_origin.extend((0..face_count).map(|i| VertexOrigin::Face(i as u32)));
    vertex_origin.extend((0..edge_count).map(|i| VertexOrigin::Edge(i as u32)));
    vertex_origin.extend((0..vertex_count).map(|i| VertexOrigin::Vertex(i as u32)));

    let lineage = LineageMaps {
        vertex_origin,
        face_parent,
        edge_parent: edge_parent_map,
    };

    // Adjacency + derived edge data both come from the single
    // `refined_topo` — no second `build_topology_from` call.
    let adjacency = adjacency_from_topology(&refined_topo);
    let (child_edge_has_selected_face, child_effective_edge_creases) =
        compute_edge_data(&refined_topo, &new_face_selected, selection_boundary_crease);

    Ok(CcLevelData {
        mesh: refined_mesh,
        topo: refined_topo,
        face_selected: new_face_selected,
        effective_edge_creases: child_effective_edge_creases,
        edge_has_selected_face: child_edge_has_selected_face,
        lineage,
        adjacency,
    })
}

/// Compute per-level vertex stencils from cached topology.
pub(crate) fn vertex_stencils_from_level(
    parent: &CcLevelData,
    options: &SchemeOptions,
) -> StencilTable {
    let topology = &parent.topo;
    let vertex_count = parent.mesh.vertex_count as usize;
    let edge_count = topology.edge_vertices.len();
    let face_count = topology.faces.len();

    let fp_stencils = face_point_stencils(&topology.faces);

    let ep_stencils = edge_point_stencils(
        topology,
        &fp_stencils,
        options,
        &parent.face_selected,
        &parent.edge_has_selected_face,
        &parent.effective_edge_creases,
    );

    let vp_stencils: Vec<Sparse> = (0..vertex_count)
        .map(|vi| {
            vertex_point_stencil(
                vi,
                topology,
                &parent.mesh.vertex_corners,
                options,
                &parent.face_selected,
                &parent.edge_has_selected_face,
                &parent.effective_edge_creases,
                &fp_stencils,
            )
        })
        .collect();

    let mut all_stencils = Vec::with_capacity(face_count + edge_count + vertex_count);
    all_stencils.extend(fp_stencils);
    all_stencils.extend(ep_stencils);
    all_stencils.extend(vp_stencils);

    pack(&all_stencils)
}
