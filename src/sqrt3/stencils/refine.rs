//! Sqrt3 topology refinement and stencil extraction.
//!
//! Sqrt3 levels are built in four steps:
//!
//! 1. Insert a centroid at every selected face.
//! 2. Split each selected triangle into three child triangles joining
//!    the centroid to each original edge.
//! 3. Flip every interior parent edge whose crease is below 1 and
//!    whose adjacent faces are both selected.
//! 4. Rebuild topology on the post-flip mesh and compute vertex
//!    smoothing stencils from the *parent* topology.
//!
//! [`refine_topology_once`] runs steps 1–3 and the post-flip topology
//! rebuild (step 4's setup), producing a child [`Sqrt3LevelData`]
//! whose `topo` field is the rebuilt post-flip topology ready for the
//! next refinement step. [`vertex_stencils_from_level`] reads the
//! parent [`Sqrt3LevelData`]'s `topo` (which is the *input* to that
//! refinement step, i.e. its own parent topology) and produces the
//! centroid + vertex-point stencil table.

use crate::{Adjacency, KernelError, LineageMaps, Mesh, SchemeOptions, StencilTable, VertexOrigin};

use crate::sharpness::decay_sharpness;

use super::super::topology::{build_topology, edge_key};
use super::level::{Sqrt3LevelData, compute_effective_edge_creases};
use super::points::vertex_point_stencil;
use super::sparse::{Sparse, pack};

/// Build one refinement level's topology from the parent level.
///
/// Does NOT compute vertex stencils — use [`vertex_stencils_from_level`].
pub(crate) fn refine_topology_once(
    parent: &Sqrt3LevelData,
    options: &SchemeOptions,
    selection_boundary_crease: f32,
) -> Result<Sqrt3LevelData, KernelError> {
    let topology = &parent.topo;
    let face_selected = &parent.face_selected;
    let effective_edge_creases = &parent.effective_edge_creases;
    let topo_mesh = &parent.mesh;
    let vertex_count = topo_mesh.vertex_count as usize;
    let edge_count = topology.edge_vertices.len();
    let face_count = topology.faces.len();

    // ── Step 2: Triangle splitting ─────────────────────────────────────
    let cp_idx = |fi: usize| fi as u32;
    let vp_idx = |vi: usize| (face_count + vi) as u32;

    let mut new_faces: Vec<Vec<u32>> = Vec::new();
    let mut face_parent: Vec<u32> = Vec::new();
    let mut new_face_selected: Vec<bool> = Vec::new();

    // Track which parent edges appear in child faces (for flipping later).
    let mut parent_edge_children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); edge_count];

    (0..face_count).for_each(|fi| {
        let fv = topology.faces.row(fi);
        if face_selected[fi] {
            (0..3).for_each(|i| {
                let child_fi = new_faces.len();
                let v_cur = fv[i];
                let v_next = fv[(i + 1) % 3];
                new_faces.push(vec![
                    cp_idx(fi),
                    vp_idx(v_cur as usize),
                    vp_idx(v_next as usize),
                ]);
                face_parent.push(fi as u32);
                new_face_selected.push(true);

                let fe_i = topology.face_edges.get(fi, i) as usize;
                parent_edge_children[fe_i].push((child_fi, 1));
            });
        } else {
            new_faces.push(fv.iter().map(|&v| vp_idx(v as usize)).collect());
            face_parent.push(fi as u32);
            new_face_selected.push(false);
        }
    });

    // ── Step 3: Edge flipping ──────────────────────────────────────────
    (0..edge_count).for_each(|ei| {
        if topology.edge_is_boundary[ei] || effective_edge_creases[ei] >= 1.0 {
            return;
        }
        let ef = topology.edge_faces.row(ei);
        if ef.len() < 2 || !face_selected[ef[0] as usize] || !face_selected[ef[1] as usize] {
            return;
        }

        let children = &parent_edge_children[ei];
        if children.len() != 2 {
            return;
        }

        let (cf0, _edge_pos_0) = children[0];
        let (cf1, _edge_pos_1) = children[1];

        let [v0, v1] = topology.edge_vertices[ei];
        let vp0 = vp_idx(v0 as usize);
        let vp1 = vp_idx(v1 as usize);

        let centroid_0 = new_faces[cf0]
            .iter()
            .find(|&&v| v != vp0 && v != vp1)
            .copied();
        let centroid_1 = new_faces[cf1]
            .iter()
            .find(|&&v| v != vp0 && v != vp1)
            .copied();

        if let (Some(c0), Some(c1)) = (centroid_0, centroid_1) {
            new_faces[cf0] = vec![c0, c1, vp0];
            new_faces[cf1] = vec![c0, vp1, c1];
        }
    });

    // ── Step 4: Build post-flip topology ───────────────────────────────
    let total_verts = (face_count + vertex_count) as u32;
    let mut fvc = Vec::with_capacity(new_faces.len());
    let mut fvi = Vec::new();
    new_faces.iter().for_each(|f| {
        fvc.push(f.len() as u32);
        fvi.extend(f.iter().copied());
    });

    let post_flip_topo = Mesh {
        vertex_count: total_verts,
        face_vertex_counts: fvc.clone(),
        face_vertex_indices: fvi.clone(),
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; total_verts as usize],
    };

    let post_flip_analysis = build_topology(&post_flip_topo)?;

    // ── Propagate creases and corners ──────────────────────────────────
    let mut refined_creases = vec![0.0f32; post_flip_analysis.edge_vertices.len()];
    let mut edge_parent_map = vec![u32::MAX; post_flip_analysis.edge_vertices.len()];

    topology
        .edge_vertices
        .iter()
        .copied()
        .enumerate()
        .filter(|(ei, _)| effective_edge_creases[*ei] > 0.0)
        .for_each(|(ei, [v0, v1])| {
            let child_crease = if options.crease_normalize {
                effective_edge_creases[ei]
            } else {
                decay_sharpness(effective_edge_creases[ei])
            };
            if child_crease <= 0.0 {
                return;
            }

            // After edge flip, the parent edge (v0,v1) might not exist
            // as (vp0,vp1) anymore. Look for it — if it was flipped, it's gone.
            let key = edge_key(vp_idx(v0 as usize), vp_idx(v1 as usize));
            if let Some(&nei) = post_flip_analysis.edge_key_to_index.get(&key) {
                refined_creases[nei] = child_crease;
                edge_parent_map[nei] = ei as u32;
            }
        });

    let refined_corners: Vec<f32> = std::iter::repeat(0.0)
        .take(face_count)
        .chain(topo_mesh.vertex_corners.iter().map(|&c| {
            if options.corner_normalize || c <= 0.0 {
                c
            } else {
                decay_sharpness(c)
            }
        }))
        .collect();

    // ── Assemble refined mesh + lineage + adjacency ────────────────────
    let refined_mesh = Mesh {
        vertex_count: total_verts,
        face_vertex_counts: fvc,
        face_vertex_indices: fvi,
        edge_vertices: post_flip_analysis.edge_vertices.clone(),
        edge_creases: refined_creases,
        vertex_corners: refined_corners,
    };

    let mut vertex_origin = Vec::with_capacity(total_verts as usize);
    vertex_origin.extend((0..face_count).map(|i| VertexOrigin::Face(i as u32)));
    vertex_origin.extend((0..vertex_count).map(|i| VertexOrigin::Vertex(i as u32)));

    let lineage = LineageMaps {
        vertex_origin,
        face_parent,
        edge_parent: edge_parent_map,
    };

    let adjacency = {
        let (_, face_edges) = post_flip_analysis.face_edges.clone().into_parts();

        let edge_faces: Vec<[u32; 2]> = (0..post_flip_analysis.edge_vertices.len())
            .map(|ei| {
                let ef = post_flip_analysis.edge_faces.row(ei);
                match ef.len() {
                    0 => [u32::MAX, u32::MAX],
                    1 => [ef[0], u32::MAX],
                    _ => [ef[0], ef[1]],
                }
            })
            .collect();

        let edge_is_boundary: Vec<bool> = edge_faces.iter().map(|ef| ef[1] == u32::MAX).collect();

        let vertex_count_post = post_flip_analysis.vertex_edges.len();
        let mut vertex_is_boundary = vec![false; vertex_count_post];
        for vi in 0..vertex_count_post {
            if post_flip_analysis
                .vertex_edges
                .row(vi)
                .iter()
                .any(|&ei| edge_is_boundary[ei as usize])
            {
                vertex_is_boundary[vi] = true;
            }
        }

        let (vertex_edge_offsets, vertex_edges) = post_flip_analysis.vertex_edges.clone().into_parts();
        let (vertex_face_offsets, vertex_faces) = post_flip_analysis.vertex_faces.clone().into_parts();

        Adjacency {
            face_edges,
            edge_faces,
            vertex_edge_offsets,
            vertex_edges,
            vertex_face_offsets,
            vertex_faces,
            edge_is_boundary,
            vertex_is_boundary,
        }
    };

    // Precompute the child's effective edge creases for the next level's
    // refinement, matching the pattern CC uses.
    let child_effective_edge_creases = compute_effective_edge_creases(
        &post_flip_analysis,
        &new_face_selected,
        selection_boundary_crease,
    );

    Ok(Sqrt3LevelData {
        mesh: refined_mesh,
        topo: post_flip_analysis,
        face_selected: new_face_selected,
        effective_edge_creases: child_effective_edge_creases,
        lineage,
        adjacency,
    })
}

/// Compute per-level vertex stencils from a cached parent level.
///
/// Reads the parent topology (pre-refinement) and produces stencils
/// for `[centroids(0..F), vertex_points(F..F+V)]` in the child mesh.
pub(crate) fn vertex_stencils_from_level(
    parent: &Sqrt3LevelData,
    options: &SchemeOptions,
) -> StencilTable {
    let topology = &parent.topo;
    let vertex_count = parent.mesh.vertex_count as usize;
    let face_count = topology.faces.len();

    // ── Centroid stencils ──────────────────────────────────────────────
    let centroid_stencils: Vec<Sparse> = (0..face_count)
        .map(|fi| {
            let fv = topology.faces.row(fi);
            let w = 1.0 / fv.len() as f32;
            fv.iter().map(|&v| (v, w)).collect()
        })
        .collect();

    // ── Vertex smoothing stencils ──────────────────────────────────────
    let vp_stencils: Vec<Sparse> = (0..vertex_count)
        .map(|vi| {
            vertex_point_stencil(
                vi,
                topology,
                &parent.mesh.vertex_corners,
                options,
                &parent.face_selected,
                &parent.effective_edge_creases,
                &centroid_stencils,
            )
        })
        .collect();

    let mut all_stencils = Vec::with_capacity(face_count + vertex_count);
    all_stencils.extend(centroid_stencils);
    all_stencils.extend(vp_stencils);

    pack(&all_stencils)
}
