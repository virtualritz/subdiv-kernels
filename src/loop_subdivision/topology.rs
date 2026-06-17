use crate::csr::CsrVec;
use crate::{KernelError, Mesh};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Topology {
    pub(crate) faces: CsrVec,
    pub(crate) face_edges: CsrVec,
    pub(crate) edge_vertices: Vec<[u32; 2]>,
    pub(crate) edge_faces: Vec<[u32; 2]>,
    pub(crate) edge_key_to_index: FxHashMap<(u32, u32), usize>,
    pub(crate) edge_creases: Vec<f32>,
    pub(crate) edge_is_boundary: Vec<bool>,
    pub(crate) vertex_edges: CsrVec,
    pub(crate) vertex_faces: CsrVec,
    pub(crate) vertex_is_boundary: Vec<bool>,
}

/// Build topology analysis from a [`Mesh`].
pub(crate) fn build_topology_from(topo: &Mesh) -> Result<Topology, KernelError> {
    topo.validate()?;
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
    let mut edge_faces = Vec::<[u32; 2]>::new();
    let mut edge_key_to_index = FxHashMap::<(u32, u32), usize>::default();
    let mut face_edges = vec![Vec::<usize>::new(); faces.len()];

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
                edge_faces.push([face_index as u32, u32::MAX]);
                edge_key_to_index.insert(key, created);
                created
            };

            let edge_face_pair = &mut edge_faces[edge_index];
            if edge_face_pair[0] != face_index as u32 && edge_face_pair[1] != face_index as u32 {
                if edge_face_pair[1] != u32::MAX {
                    return Err(KernelError::InvalidTopology(
                        "non-manifold edge has more than two incident faces",
                    ));
                }
                edge_face_pair[1] = face_index as u32;
            }

            face_edges[face_index].push(edge_index);
        }
    }

    let edge_is_boundary = edge_faces
        .iter()
        .map(|faces_for_edge| faces_for_edge[1] == u32::MAX)
        .collect::<Vec<_>>();

    let mut vertex_edges = vec![Vec::<usize>::new(); vertex_count];
    edge_vertices
        .iter()
        .enumerate()
        .for_each(|(edge_index, edge)| {
            vertex_edges[edge[0] as usize].push(edge_index);
            vertex_edges[edge[1] as usize].push(edge_index);
        });

    let mut vertex_faces = vec![Vec::<usize>::new(); vertex_count];
    faces
        .iter()
        .enumerate()
        .for_each(|(face_index, face_vertices)| {
            face_vertices.iter().for_each(|&vertex| {
                vertex_faces[vertex as usize].push(face_index);
            });
        });

    let mut vertex_is_boundary = vec![false; vertex_count];
    edge_is_boundary
        .iter()
        .copied()
        .enumerate()
        .for_each(|(edge_index, is_boundary)| {
            if is_boundary {
                let [v0, v1] = edge_vertices[edge_index];
                vertex_is_boundary[v0 as usize] = true;
                vertex_is_boundary[v1 as usize] = true;
            }
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
        edge_faces,
        edge_key_to_index,
        edge_creases,
        edge_is_boundary,
        vertex_edges: CsrVec::from_jagged(&vertex_edges),
        vertex_faces: CsrVec::from_jagged(&vertex_faces),
        vertex_is_boundary,
    })
}

/// Result of analytically refining a fully-selected Loop level: the child
/// triangles (CSR counts/indices), their parent-face lineage, and the child
/// [`Topology`] (with `edge_creases` left at `0.0` for the caller to fill).
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

/// Edge accumulator: dedups child edges via caller-supplied direct-index slots
/// while recording `edge_vertices` / `edge_faces` / `face_edges` in exactly the
/// order [`build_topology_raw`] discovers them.
struct EdgeAccum {
    edge_vertices: Vec<[u32; 2]>,
    edge_faces: Vec<[u32; 2]>,
    edge_key_to_index: FxHashMap<(u32, u32), usize>,
    fe_values: Vec<u32>,
}

impl EdgeAccum {
    fn edge(&mut self, slot: &mut u32, a: u32, b: u32, child_face: u32) {
        let edge_index = if *slot != u32::MAX {
            let ei = *slot as usize;
            let pair = &mut self.edge_faces[ei];
            if pair[0] != child_face && pair[1] != child_face {
                pair[1] = child_face;
            }
            ei
        } else {
            let created = self.edge_vertices.len();
            let key = edge_key(a, b);
            self.edge_vertices.push([key.0, key.1]);
            self.edge_faces.push([child_face, u32::MAX]);
            self.edge_key_to_index.insert(key, created);
            *slot = created as u32;
            created
        };
        self.fe_values.push(edge_index as u32);
    }
}

/// Analytically refine a **fully-selected** Loop level.
///
/// Loop refinement is regular: child indices are fixed — edge-point `ep(ei)=ei`,
/// vertex-point `vp(vi)=E+vi` — and each parent triangle splits into four child
/// triangles `[vp0,ep01,ep20] [vp1,ep12,ep01] [vp2,ep20,ep12] [ep01,ep12,ep20]`.
/// Child edges are two families: *split* edges (vp↔ep, two per parent edge,
/// shared across the faces adjacent to that edge) and *mid-mid* edges (ep↔ep,
/// three per parent triangle, interior to it). Both are deduped via direct-index
/// arrays (split by `2·edge+side`, mid-mid by parent corner) while scanning the
/// four child triangles in the same order `build_topology_raw` would — so the
/// result is bit-identical to `build_topology_from` on the child mesh, with no
/// per-corner hash. `edge_creases` is left at `0.0`; the caller propagates.
///
/// Only valid when all parent faces are selected (uniform refinement); mixed
/// selection introduces vertex/vertex child edges and must use the generic
/// builder.
pub(crate) fn build_loop_refined_uniform(
    parent: &Topology,
    parent_vertex_count: usize,
) -> RefinedUniform {
    let face_count = parent.faces.len();
    let edge_count = parent.edge_vertices.len();
    let ep = |ei: u32| ei; // edge-point index
    let vp = |vi: u32| edge_count as u32 + vi; // vertex-point index
    let total_verts = edge_count + parent_vertex_count;

    // Parent corner offsets (mid-mid edges are keyed by parent corner).
    let mut p_corner_off = Vec::with_capacity(face_count + 1);
    p_corner_off.push(0usize);
    for fi in 0..face_count {
        p_corner_off.push(p_corner_off[fi] + parent.faces.row_len(fi));
    }
    let child_face_count = 4 * face_count;

    let mut face_vertex_indices = Vec::with_capacity(child_face_count * 3);
    let mut face_parent = Vec::with_capacity(child_face_count);

    let mut accum = EdgeAccum {
        edge_vertices: Vec::new(),
        edge_faces: Vec::new(),
        edge_key_to_index: FxHashMap::default(),
        fe_values: Vec::with_capacity(child_face_count * 3),
    };

    let mut split_idx = vec![u32::MAX; 2 * edge_count];
    let mut mid_idx = vec![u32::MAX; p_corner_off[face_count]];

    let mut child_face = 0u32;
    for fi in 0..face_count {
        let v0 = parent.faces.get(fi, 0);
        let v1 = parent.faces.get(fi, 1);
        let v2 = parent.faces.get(fi, 2);
        let e01 = parent.face_edges.get(fi, 0) as usize;
        let e12 = parent.face_edges.get(fi, 1) as usize;
        let e20 = parent.face_edges.get(fi, 2) as usize;

        let (vp0, vp1, vp2) = (vp(v0), vp(v1), vp(v2));
        let (a01, a12, a20) = (ep(e01 as u32), ep(e12 as u32), ep(e20 as u32));
        let cb = p_corner_off[fi];

        let s = |e: usize, v: u32| edge_side(parent, e, v);

        // T0 = [vp0, a01, a20]
        face_vertex_indices.extend_from_slice(&[vp0, a01, a20]);
        face_parent.push(fi as u32);
        accum.edge(&mut split_idx[2 * e01 + s(e01, v0)], vp0, a01, child_face);
        accum.edge(&mut mid_idx[cb], a01, a20, child_face);
        accum.edge(&mut split_idx[2 * e20 + s(e20, v0)], a20, vp0, child_face);
        child_face += 1;

        // T1 = [vp1, a12, a01]
        face_vertex_indices.extend_from_slice(&[vp1, a12, a01]);
        face_parent.push(fi as u32);
        accum.edge(&mut split_idx[2 * e12 + s(e12, v1)], vp1, a12, child_face);
        accum.edge(&mut mid_idx[cb + 1], a12, a01, child_face);
        accum.edge(&mut split_idx[2 * e01 + s(e01, v1)], a01, vp1, child_face);
        child_face += 1;

        // T2 = [vp2, a20, a12]
        face_vertex_indices.extend_from_slice(&[vp2, a20, a12]);
        face_parent.push(fi as u32);
        accum.edge(&mut split_idx[2 * e20 + s(e20, v2)], vp2, a20, child_face);
        accum.edge(&mut mid_idx[cb + 2], a20, a12, child_face);
        accum.edge(&mut split_idx[2 * e12 + s(e12, v2)], a12, vp2, child_face);
        child_face += 1;

        // T3 = [a01, a12, a20] (all mid-mid, already created above)
        face_vertex_indices.extend_from_slice(&[a01, a12, a20]);
        face_parent.push(fi as u32);
        accum.edge(&mut mid_idx[cb + 1], a01, a12, child_face);
        accum.edge(&mut mid_idx[cb + 2], a12, a20, child_face);
        accum.edge(&mut mid_idx[cb], a20, a01, child_face);
        child_face += 1;
    }

    let EdgeAccum {
        edge_vertices,
        edge_faces,
        edge_key_to_index,
        fe_values,
    } = accum;
    let new_edge_count = edge_vertices.len();

    let tri_offsets: Vec<u32> = (0..=child_face_count).map(|i| (i * 3) as u32).collect();
    let faces = CsrVec::from_parts(tri_offsets.clone(), face_vertex_indices.clone());
    let face_edges = CsrVec::from_parts(tri_offsets, fe_values);

    let edge_is_boundary: Vec<bool> = edge_faces.iter().map(|p| p[1] == u32::MAX).collect();

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
        for k in 0..3 {
            let v = face_vertex_indices[cf * 3 + k] as usize;
            vf_values[vf_cursor[v] as usize] = cf as u32;
            vf_cursor[v] += 1;
        }
    }
    let vertex_faces = CsrVec::from_parts(vf_offsets, vf_values);

    let mut vertex_is_boundary = vec![false; total_verts];
    for (ei, &boundary) in edge_is_boundary.iter().enumerate() {
        if boundary {
            let [a, b] = edge_vertices[ei];
            vertex_is_boundary[a as usize] = true;
            vertex_is_boundary[b as usize] = true;
        }
    }

    let topology = Topology {
        faces,
        face_edges,
        edge_vertices,
        edge_faces,
        edge_key_to_index,
        edge_creases: vec![0.0; new_edge_count],
        edge_is_boundary,
        vertex_edges,
        vertex_faces,
        vertex_is_boundary,
    };

    RefinedUniform {
        face_vertex_counts: vec![3; child_face_count],
        face_vertex_indices,
        face_parent,
        topology,
    }
}

fn decode_faces(
    face_vertex_counts: &[u32],
    face_vertex_indices: &[u32],
    vertex_count: usize,
) -> Result<Vec<Vec<u32>>, KernelError> {
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

        let face = face_vertex_indices[offset..offset + count].to_vec();

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

#[inline]
pub(crate) fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}
