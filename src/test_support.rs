//! Shared test-only helpers for face-varying parity oracles.
//!
//! The central trick is [`seamed_position_channel`]: it builds a face-varying
//! channel whose **values equal the vertex positions** but whose **indices are
//! split (seamed) along a chosen set of edges**. Refining that channel under
//! the `Smooth` face-varying mode (seams treated as smooth crease curves) must
//! reproduce the *positional* refinement obtained when the same edges are
//! marked as infinite creases — a strong, scheme-independent oracle for the
//! seam crease-curve rule that needs no external reference.

use std::collections::HashMap;

use crate::{FaceVaryingChannel, Mesh};

/// Canonical undirected edge key.
fn ek(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// CSR-style face-corner offsets.
fn face_offsets(mesh: &Mesh) -> Vec<usize> {
    let mut off = vec![0usize; mesh.face_vertex_counts.len() + 1];
    for f in 0..mesh.face_vertex_counts.len() {
        off[f + 1] = off[f] + mesh.face_vertex_counts[f] as usize;
    }
    off
}

/// Per-edge infinite-crease array aligned to `mesh.edge_vertices`, with the
/// edges in `seam` (given as unordered vertex pairs) set to the OSD infinite
/// sentinel (`10.0`) and all others to `0.0`.
pub(crate) fn creases_for(mesh: &Mesh, seam: &[[u32; 2]]) -> Vec<f32> {
    let seam_set: std::collections::HashSet<(u32, u32)> =
        seam.iter().map(|&[a, b]| ek(a, b)).collect();
    mesh.edge_vertices
        .iter()
        .map(|&[a, b]| {
            if seam_set.contains(&ek(a, b)) {
                10.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Build a position-valued face-varying channel that is seamed along `seam`.
///
/// Faces are grouped into islands (connected components under non-seam shared
/// edges). Each `(vertex, island)` pair gets a distinct face-varying value
/// index whose value is the vertex's position, so the channel carries the
/// geometry verbatim while encoding exactly the requested seams. Returns the
/// channel topology and its value array.
pub(crate) fn seamed_position_channel(
    mesh: &Mesh,
    seam: &[[u32; 2]],
    positions: &[[f32; 3]],
) -> (FaceVaryingChannel, Vec<[f32; 3]>) {
    let seam_set: std::collections::HashSet<(u32, u32)> =
        seam.iter().map(|&[a, b]| ek(a, b)).collect();
    let face_count = mesh.face_vertex_counts.len();
    let off = face_offsets(mesh);

    // Undirected edge -> incident faces.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for f in 0..face_count {
        let n = mesh.face_vertex_counts[f] as usize;
        for k in 0..n {
            let a = mesh.face_vertex_indices[off[f] + k];
            let b = mesh.face_vertex_indices[off[f] + (k + 1) % n];
            edge_faces.entry(ek(a, b)).or_default().push(f);
        }
    }

    // Union faces that share a non-seam edge.
    let mut parent: Vec<usize> = (0..face_count).collect();
    for (key, faces) in &edge_faces {
        if faces.len() == 2 && !seam_set.contains(key) {
            uf_union(&mut parent, faces[0], faces[1]);
        }
    }

    // Assign a value index per (vertex, island).
    let mut map: HashMap<(u32, usize), u32> = HashMap::new();
    let mut indices = vec![0u32; off[face_count]];
    let mut values: Vec<[f32; 3]> = Vec::new();
    for f in 0..face_count {
        let island = uf_find(&mut parent, f);
        let n = mesh.face_vertex_counts[f] as usize;
        for k in 0..n {
            let v = mesh.face_vertex_indices[off[f] + k];
            let id = *map.entry((v, island)).or_insert_with(|| {
                let id = values.len() as u32;
                values.push(positions[v as usize]);
                id
            });
            indices[off[f] + k] = id;
        }
    }

    (
        FaceVaryingChannel {
            indices,
            value_count: values.len() as u32,
        },
        values,
    )
}
