use crate::csr::CsrVec;
use crate::{KernelError, Mesh};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Topology {
    pub(crate) faces: CsrVec,
    pub(crate) face_edges: CsrVec,
    pub(crate) edge_vertices: Vec<[u32; 2]>,
    pub(crate) edge_faces: CsrVec,
    pub(crate) vertex_edges: CsrVec,
    pub(crate) vertex_faces: CsrVec,
    pub(crate) edge_key_to_index: FxHashMap<(u32, u32), usize>,
    pub(crate) edge_creases: Vec<f32>,
}

/// Build topology analysis from a [`Mesh`].
pub(crate) fn build_topology_from(topo: &Mesh) -> Result<Topology, KernelError> {
    build_topology_raw(
        topo.vertex_count as usize,
        &topo.face_vertex_counts,
        &topo.face_vertex_indices,
        &topo.edge_vertices,
        &topo.edge_creases,
    )
}

fn build_topology_raw(
    vertex_count: usize,
    face_vertex_counts: &[u32],
    face_vertex_indices: &[u32],
    source_edge_vertices: &[[u32; 2]],
    source_edge_creases: &[f32],
) -> Result<Topology, KernelError> {
    if vertex_count == 0 {
        return Err(KernelError::InvalidTopology("mesh has no vertices"));
    }

    let faces = decode_faces(face_vertex_counts, face_vertex_indices, vertex_count)?;

    let mut source_creases = FxHashMap::<(u32, u32), f32>::default();
    source_edge_vertices
        .iter()
        .zip(source_edge_creases.iter())
        .for_each(|(edge, &crease)| {
            let key = edge_key(edge[0], edge[1]);
            source_creases
                .entry(key)
                .and_modify(|existing| {
                    if crease.partial_cmp(existing).unwrap_or(Ordering::Less) == Ordering::Greater {
                        *existing = crease;
                    }
                })
                .or_insert(crease);
        });

    let mut edge_vertices = Vec::<[u32; 2]>::new();
    let mut edge_faces = Vec::<SmallVec<[usize; 2]>>::new();
    let mut edge_key_to_index = FxHashMap::<(u32, u32), usize>::default();
    let mut face_edges = vec![SmallVec::<[usize; 4]>::new(); faces.len()];

    for (face_index, face_vertices) in faces.iter().enumerate() {
        let corner_count = face_vertices.len();
        for corner in 0..corner_count {
            let v0 = face_vertices[corner];
            let v1 = face_vertices[(corner + 1) % corner_count];

            if v0 == v1 {
                return Err(KernelError::InvalidTopology(
                    "degenerate face edge uses duplicate vertices",
                ));
            }

            let key = edge_key(v0, v1);
            let edge_index = if let Some(&existing) = edge_key_to_index.get(&key) {
                existing
            } else {
                let created = edge_vertices.len();
                edge_vertices.push([key.0, key.1]);
                edge_faces.push(SmallVec::new());
                edge_key_to_index.insert(key, created);
                created
            };

            let faces_for_edge = &mut edge_faces[edge_index];
            if !faces_for_edge.contains(&face_index) {
                if faces_for_edge.len() >= 2 {
                    return Err(KernelError::InvalidTopology(
                        "non-manifold edge has more than two incident faces",
                    ));
                }
                faces_for_edge.push(face_index);
            }

            face_edges[face_index].push(edge_index);
        }
    }

    let mut vertex_edges = vec![SmallVec::<[usize; 4]>::new(); vertex_count];
    edge_vertices
        .iter()
        .enumerate()
        .for_each(|(edge_index, edge)| {
            vertex_edges[edge[0] as usize].push(edge_index);
            vertex_edges[edge[1] as usize].push(edge_index);
        });

    let mut vertex_faces = vec![SmallVec::<[usize; 4]>::new(); vertex_count];
    faces
        .iter()
        .enumerate()
        .for_each(|(face_index, face_vertices)| {
            face_vertices.iter().for_each(|&vertex| {
                vertex_faces[vertex as usize].push(face_index);
            });
        });

    let edge_creases = edge_vertices
        .iter()
        .map(|edge| {
            source_creases
                .get(&edge_key(edge[0], edge[1]))
                .copied()
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();

    Ok(Topology {
        faces: CsrVec::from_jagged_u32(&faces),
        face_edges: CsrVec::from_jagged(&face_edges),
        edge_vertices,
        edge_faces: CsrVec::from_jagged(&edge_faces),
        vertex_edges: CsrVec::from_jagged(&vertex_edges),
        vertex_faces: CsrVec::from_jagged(&vertex_faces),
        edge_key_to_index,
        edge_creases,
    })
}

/// Result of analytically refining a fully-selected CC level: the child faces
/// (CSR counts/indices), their parent-face lineage, and the child [`Topology`].
pub(crate) struct RefinedUniform {
    pub(crate) face_vertex_counts: Vec<u32>,
    pub(crate) face_vertex_indices: Vec<u32>,
    pub(crate) face_parent: Vec<u32>,
    pub(crate) topology: Topology,
}

/// Side (0/1) of vertex `v` within sorted parent edge `e`.
#[inline]
fn edge_side(parent: &Topology, e: usize, v: u32) -> usize {
    if v == parent.edge_vertices[e][0] {
        0
    } else {
        1
    }
}

#[inline]
fn prefix_sum(degrees: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(degrees.len() + 1);
    offsets.push(0u32);
    let mut acc = 0u32;
    for &d in degrees {
        acc += d;
        offsets.push(acc);
    }
    offsets
}

/// Analytically refine a **fully-selected** Catmull-Clark level.
///
/// CC refinement is regular: child element indices are fixed by construction —
/// face-point `fp(fi)=fi`, edge-point `ep(ei)=F+ei`, vertex-point
/// `vp(vi)=F+E+vi` — and every child adjacency is enumerable from the parent.
/// This iterates parent faces/corners (authoring the child corner-quads
/// directly into CSR) and discovers child edges in the **same order**
/// [`build_topology_raw`] would while scanning those child faces, deduping via
/// direct-index arrays (a spoke edge keyed by its parent corner, a split edge
/// by `2·parent_edge+side`) instead of a per-corner hash map. The result is
/// bit-identical to `build_topology_from` on the same child mesh, with
/// `edge_creases` left at `0.0` (the caller propagates creases afterward).
///
/// Only valid when **all** parent faces are selected (uniform refinement);
/// mixed selection introduces vertex-point/vertex-point edges and must use the
/// generic builder.
pub(crate) fn build_cc_refined_uniform(
    parent: &Topology,
    parent_vertex_count: usize,
) -> RefinedUniform {
    let face_count = parent.faces.len();
    let edge_count = parent.edge_vertices.len();
    let fp = |fi: usize| fi as u32;
    let ep = |ei: usize| (face_count + ei) as u32;
    let vp = |vi: usize| (face_count + edge_count + vi) as u32;
    let total_verts = face_count + edge_count + parent_vertex_count;

    // Parent corner offsets (spoke dedup is keyed by global parent corner).
    let mut p_corner_off = Vec::with_capacity(face_count + 1);
    p_corner_off.push(0usize);
    for fi in 0..face_count {
        p_corner_off.push(p_corner_off[fi] + parent.faces.row_len(fi));
    }
    let child_face_count = p_corner_off[face_count];

    let mut face_vertex_counts = Vec::with_capacity(child_face_count);
    let mut face_vertex_indices = Vec::with_capacity(child_face_count * 4);
    let mut face_parent = Vec::with_capacity(child_face_count);
    let mut fe_values = Vec::with_capacity(child_face_count * 4);

    let mut edge_vertices = Vec::<[u32; 2]>::new();
    let mut edge_faces_jagged = Vec::<SmallVec<[u32; 2]>>::new();
    let mut edge_key_to_index = FxHashMap::<(u32, u32), usize>::default();

    let mut spoke_idx = vec![u32::MAX; child_face_count];
    let mut split_idx = vec![u32::MAX; 2 * edge_count];

    let mut child_face = 0u32;
    for fi in 0..face_count {
        let val = parent.faces.row_len(fi);
        for c in 0..val {
            let cur_v = parent.faces.get(fi, c) as usize;
            let cur_e = parent.face_edges.get(fi, c) as usize;
            let prev_local = (c + val - 1) % val;
            let prev_e = parent.face_edges.get(fi, prev_local) as usize;

            // Child quad: [fp(fi), ep(prev_e), vp(cur_v), ep(cur_e)].
            let quad = [fp(fi), ep(prev_e), vp(cur_v), ep(cur_e)];
            face_vertex_counts.push(4);
            face_vertex_indices.extend_from_slice(&quad);
            face_parent.push(fi as u32);

            // Four edges, in corner order — the exact order build_topology_raw
            // encounters them. 0,3 are spokes; 1,2 are splits.
            for k in 0..4usize {
                let a = quad[k];
                let b = quad[(k + 1) % 4];
                let slot = match k {
                    0 => &mut spoke_idx[p_corner_off[fi] + prev_local],
                    3 => &mut spoke_idx[p_corner_off[fi] + c],
                    1 => &mut split_idx[2 * prev_e + edge_side(parent, prev_e, cur_v as u32)],
                    _ => &mut split_idx[2 * cur_e + edge_side(parent, cur_e, cur_v as u32)],
                };
                let edge_index = if *slot != u32::MAX {
                    *slot as usize
                } else {
                    let created = edge_vertices.len();
                    let key = edge_key(a, b);
                    edge_vertices.push([key.0, key.1]);
                    edge_faces_jagged.push(SmallVec::new());
                    edge_key_to_index.insert(key, created);
                    *slot = created as u32;
                    created
                };
                let incident = &mut edge_faces_jagged[edge_index];
                if !incident.contains(&child_face) {
                    incident.push(child_face);
                }
                fe_values.push(edge_index as u32);
            }
            child_face += 1;
        }
    }

    let new_edge_count = edge_vertices.len();

    // faces / face_edges: every child face is a quad ⇒ stride 4.
    let quad_offsets: Vec<u32> = (0..=child_face_count).map(|i| (i * 4) as u32).collect();
    let faces = CsrVec::from_parts(quad_offsets.clone(), face_vertex_indices.clone());
    let face_edges = CsrVec::from_parts(quad_offsets, fe_values);
    let edge_faces = CsrVec::from_jagged_u32(&edge_faces_jagged);

    // vertex_edges: edges in edge-index order, pushed to both endpoints.
    let mut ve_deg = vec![0u32; total_verts];
    for &[a, b] in &edge_vertices {
        ve_deg[a as usize] += 1;
        ve_deg[b as usize] += 1;
    }
    let ve_offsets = prefix_sum(&ve_deg);
    let mut ve_values = vec![0u32; *ve_offsets.last().unwrap() as usize];
    let mut ve_cursor = ve_offsets[..total_verts].to_vec();
    for (ei, &[a, b]) in edge_vertices.iter().enumerate() {
        ve_values[ve_cursor[a as usize] as usize] = ei as u32;
        ve_cursor[a as usize] += 1;
        ve_values[ve_cursor[b as usize] as usize] = ei as u32;
        ve_cursor[b as usize] += 1;
    }
    let vertex_edges = CsrVec::from_parts(ve_offsets, ve_values);

    // vertex_faces: faces in child-face order.
    let mut vf_deg = vec![0u32; total_verts];
    for &v in &face_vertex_indices {
        vf_deg[v as usize] += 1;
    }
    let vf_offsets = prefix_sum(&vf_deg);
    let mut vf_values = vec![0u32; *vf_offsets.last().unwrap() as usize];
    let mut vf_cursor = vf_offsets[..total_verts].to_vec();
    for cf in 0..child_face_count {
        for k in 0..4 {
            let v = face_vertex_indices[cf * 4 + k] as usize;
            vf_values[vf_cursor[v] as usize] = cf as u32;
            vf_cursor[v] += 1;
        }
    }
    let vertex_faces = CsrVec::from_parts(vf_offsets, vf_values);

    let topology = Topology {
        faces,
        face_edges,
        edge_vertices,
        edge_faces,
        vertex_edges,
        vertex_faces,
        edge_key_to_index,
        edge_creases: vec![0.0; new_edge_count],
    };

    RefinedUniform {
        face_vertex_counts,
        face_vertex_indices,
        face_parent,
        topology,
    }
}

fn decode_faces(
    face_vertex_counts: &[u32],
    face_vertex_indices: &[u32],
    vertex_count: usize,
) -> Result<Vec<SmallVec<[u32; 4]>>, KernelError> {
    let mut offset = 0usize;
    let mut faces = Vec::with_capacity(face_vertex_counts.len());

    for &count_u32 in face_vertex_counts {
        let count = count_u32 as usize;
        if count < 3 {
            return Err(KernelError::InvalidTopology(
                "face has fewer than three corners",
            ));
        }

        if offset + count > face_vertex_indices.len() {
            return Err(KernelError::InvalidTopology(
                "face index buffer underflow while decoding faces",
            ));
        }

        let face: SmallVec<[u32; 4]> =
            SmallVec::from_slice(&face_vertex_indices[offset..offset + count]);

        if face.iter().any(|&vertex| vertex as usize >= vertex_count) {
            return Err(KernelError::InvalidTopology(
                "face index references missing vertex",
            ));
        }

        faces.push(face);
        offset += count;
    }

    if offset != face_vertex_indices.len() {
        return Err(KernelError::InvalidTopology(
            "face index buffer has trailing data",
        ));
    }

    Ok(faces)
}

pub(crate) fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}
