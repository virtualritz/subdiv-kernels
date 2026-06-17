//! Sqrt3 topology analysis and edge flipping.

use crate::csr::CsrVec;
use crate::{KernelError, Mesh};
use rustc_hash::FxHashMap;

/// Adjacency data for Sqrt3 refinement.
pub(crate) struct Topology {
    /// Face corner lists (CSR-packed).
    pub faces: CsrVec,
    /// Per-face edge index lists (parallel to faces, CSR-packed).
    pub face_edges: CsrVec,
    /// Canonical edge endpoints.
    pub edge_vertices: Vec<[u32; 2]>,
    /// Per-edge incident face indices (up to 2, CSR-packed).
    pub edge_faces: CsrVec,
    /// Per-edge crease values.
    pub edge_creases: Vec<f32>,
    /// Per-vertex incident edge indices (CSR-packed).
    pub vertex_edges: CsrVec,
    /// Per-vertex incident face indices (CSR-packed).
    pub vertex_faces: CsrVec,
    /// Edge key → index map.
    pub edge_key_to_index: FxHashMap<(u32, u32), usize>,
    /// Per-edge boundary flag.
    pub edge_is_boundary: Vec<bool>,
    /// Per-vertex boundary flag.
    pub vertex_is_boundary: Vec<bool>,
}

pub(crate) fn build_topology(topo: &Mesh) -> Result<Topology, KernelError> {
    let vertex_count = topo.vertex_count as usize;
    (vertex_count > 0)
        .then_some(())
        .ok_or(KernelError::InvalidTopology("mesh has no vertices"))?;

    let faces = decode_faces(
        &topo.face_vertex_counts,
        &topo.face_vertex_indices,
        vertex_count,
    )?;

    let mut source_creases = FxHashMap::<(u32, u32), f32>::default();
    topo.edge_vertices
        .iter()
        .zip(topo.edge_creases.iter())
        .for_each(|(edge, &crease)| {
            let key = edge_key(edge[0], edge[1]);
            source_creases
                .entry(key)
                .and_modify(|e| *e = e.max(crease))
                .or_insert(crease);
        });

    let mut edge_vertices = Vec::<[u32; 2]>::new();
    let mut edge_faces = Vec::<Vec<usize>>::new();
    let mut edge_key_to_index = FxHashMap::<(u32, u32), usize>::default();
    let mut face_edges = vec![Vec::<usize>::new(); faces.len()];

    for (fi, fv) in faces.iter().enumerate() {
        let n = fv.len();
        for corner in 0..n {
            let v0 = fv[corner];
            let v1 = fv[(corner + 1) % n];
            (v0 != v1)
                .then_some(())
                .ok_or(KernelError::InvalidTopology("degenerate face edge"))?;

            let key = edge_key(v0, v1);
            let ei = *edge_key_to_index.entry(key).or_insert_with(|| {
                let created = edge_vertices.len();
                edge_vertices.push([key.0, key.1]);
                edge_faces.push(Vec::new());
                created
            });

            let ef = &mut edge_faces[ei];
            if !ef.contains(&fi) {
                (ef.len() < 2)
                    .then_some(())
                    .ok_or(KernelError::InvalidTopology("non-manifold edge"))?;
                ef.push(fi);
            }
            face_edges[fi].push(ei);
        }
    }

    let edge_is_boundary: Vec<bool> = edge_faces.iter().map(|ef| ef.len() < 2).collect();

    let mut vertex_edges = vec![Vec::<usize>::new(); vertex_count];
    edge_vertices
        .iter()
        .enumerate()
        .for_each(|(ei, &[v0, v1])| {
            vertex_edges[v0 as usize].push(ei);
            vertex_edges[v1 as usize].push(ei);
        });

    let mut vertex_faces = vec![Vec::<usize>::new(); vertex_count];
    faces.iter().enumerate().for_each(|(fi, fv)| {
        fv.iter().for_each(|&v| {
            vertex_faces[v as usize].push(fi);
        });
    });

    let vertex_is_boundary: Vec<bool> = (0..vertex_count)
        .map(|vi| vertex_edges[vi].iter().any(|&ei| edge_is_boundary[ei]))
        .collect();

    let edge_creases: Vec<f32> = edge_vertices
        .iter()
        .map(|ev| {
            source_creases
                .get(&edge_key(ev[0], ev[1]))
                .copied()
                .unwrap_or(0.0)
        })
        .collect();

    Ok(Topology {
        faces: CsrVec::from_jagged_u32(&faces),
        face_edges: CsrVec::from_jagged(&face_edges),
        edge_vertices,
        edge_faces: CsrVec::from_jagged(&edge_faces),
        edge_creases,
        vertex_edges: CsrVec::from_jagged(&vertex_edges),
        vertex_faces: CsrVec::from_jagged(&vertex_faces),
        edge_key_to_index,
        edge_is_boundary,
        vertex_is_boundary,
    })
}

fn decode_faces(
    counts: &[u32],
    indices: &[u32],
    vertex_count: usize,
) -> Result<Vec<Vec<u32>>, KernelError> {
    let mut offset = 0usize;
    counts
        .iter()
        .map(|&count| {
            let n = count as usize;
            (n >= 3 && offset + n <= indices.len())
                .then_some(())
                .ok_or(KernelError::InvalidTopology("invalid face"))?;

            let face = indices[offset..offset + n].to_vec();
            face.iter()
                .all(|&v| (v as usize) < vertex_count)
                .then_some(())
                .ok_or(KernelError::InvalidTopology("face index out of bounds"))?;

            offset += n;
            Ok(face)
        })
        .collect()
}

pub(crate) fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}
