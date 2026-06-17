//! Shared face-varying stencil helpers, scheme-agnostic.
//!
//! Each scheme's `face_varying.rs` builds per-face-corner stencils that
//! mirror that scheme's topology refinement; the sparse-stencil plumbing
//! and the CSR face-offset helper live here so they are written once.
//!
//! Catmull-Clark keeps its own copies in
//! [`crate::catmull_clark::face_varying`] (predates this module and is
//! OSD-validated); the Loop / √3 / Doo-Sabin modules use these.

use std::collections::{HashMap, HashSet};

use crate::{
    FaceVaryingChannel, FaceVaryingInterpolation, LineageMaps, Mesh, StencilTable, VertexOrigin,
};

/// Canonical (unordered) edge key.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// A sparse stencil row: `(input value index, weight)` pairs.
pub(crate) type Sparse = Vec<(u32, f32)>;

/// Accumulate `source * w` into `stencil`, merging duplicate indices.
pub(crate) fn merge(stencil: &mut Sparse, source: &[(u32, f32)], w: f32) {
    source.iter().for_each(|&(idx, sw)| {
        if let Some(entry) = stencil.iter_mut().find(|(i, _)| *i == idx) {
            entry.1 += sw * w;
        } else {
            stencil.push((idx, sw * w));
        }
    });
}

/// Pack per-row sparse stencils into a CSR [`StencilTable`].
pub(crate) fn pack(stencils: &[Sparse]) -> StencilTable {
    let mut offsets = Vec::with_capacity(stencils.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();

    offsets.push(0u32);
    stencils.iter().for_each(|s| {
        s.iter().for_each(|&(idx, w)| {
            indices.push(idx);
            weights.push(w);
        });
        offsets.push(indices.len() as u32);
    });

    StencilTable {
        offsets,
        indices,
        weights,
    }
}

/// CSR face-corner offsets from `face_vertex_counts`: `offsets[fi]` is the
/// index of face `fi`'s first corner, `offsets[face_count]` the total.
pub(crate) fn face_offsets(topo: &Mesh) -> Vec<usize> {
    std::iter::once(0)
        .chain(topo.face_vertex_counts.iter().scan(0usize, |acc, &c| {
            *acc += c as usize;
            Some(*acc)
        }))
        .collect()
}

/// The face-varying index of vertex `v` in parent face `f`, if present.
fn fvar_in_face(
    parent_mesh: &Mesh,
    p_off: &[usize],
    channel: &FaceVaryingChannel,
    f: usize,
    v: u32,
) -> Option<u32> {
    let n = parent_mesh.face_vertex_counts[f] as usize;
    let s = p_off[f];
    parent_mesh.face_vertex_indices[s..s + n]
        .iter()
        .position(|&u| u == v)
        .map(|k| channel.indices[s + k])
}

/// AllLinear face-varying stencils for a scheme whose child vertices carry
/// a meaningful [`VertexOrigin`] (Loop, √3, and CC-style): each refined
/// face-corner takes a purely face-local, seam-preserving value.
///
/// - `VertexOrigin::Vertex(v)` → copy of `v`'s face-varying value in the
///   child face's **home** parent face (`face_parent`).
/// - `VertexOrigin::Edge(e)` → midpoint of edge `e`'s two endpoints, read in
///   the home face (so a seam keeps each side distinct).
/// - `VertexOrigin::Face(f)` → centroid of face `f`'s corner values.
///
/// Reading the child connectivity (`child_mesh`) makes this correct even when
/// the scheme rewires child faces after splitting (√3's edge flips): the
/// per-corner value only depends on the corner's origin and its child face's
/// home, both of which the flip preserves.
pub(crate) fn all_linear_via_origin(
    parent_mesh: &Mesh,
    parent_edge_vertices: &[[u32; 2]],
    child_mesh: &Mesh,
    child_lineage: &LineageMaps,
    channel: &FaceVaryingChannel,
) -> Vec<Sparse> {
    let p_off = face_offsets(parent_mesh);
    let c_off = face_offsets(child_mesh);

    let mut out = Vec::with_capacity(child_mesh.face_vertex_indices.len());
    for cf in 0..child_mesh.face_vertex_counts.len() {
        let home = child_lineage.face_parent[cf] as usize;
        let n = child_mesh.face_vertex_counts[cf] as usize;
        let s = c_off[cf];
        for k in 0..n {
            let cv = child_mesh.face_vertex_indices[s + k];
            let stencil = match child_lineage.vertex_origin[cv as usize] {
                VertexOrigin::Vertex(v) => {
                    let fv = fvar_in_face(parent_mesh, &p_off, channel, home, v).unwrap_or(v);
                    vec![(fv, 1.0)]
                }
                VertexOrigin::Edge(e) => {
                    let [a, b] = parent_edge_vertices[e as usize];
                    let fa = fvar_in_face(parent_mesh, &p_off, channel, home, a);
                    let fb = fvar_in_face(parent_mesh, &p_off, channel, home, b);
                    match (fa, fb) {
                        (Some(fa), Some(fb)) => vec![(fa, 0.5), (fb, 0.5)],
                        _ => vec![(fa.or(fb).unwrap_or(a), 1.0)],
                    }
                }
                VertexOrigin::Face(f) => {
                    let f = f as usize;
                    let n2 = parent_mesh.face_vertex_counts[f] as usize;
                    let s2 = p_off[f];
                    let w = 1.0 / n2 as f32;
                    (0..n2).map(|j| (channel.indices[s2 + j], w)).collect()
                }
            };
            out.push(stencil);
        }
    }
    out
}

/// AllLinear face-varying stencils for Doo-Sabin: every child vertex is a
/// face-vertex point `fv(fi, c)` whose index equals the parent corner index,
/// so each refined corner copies its source corner's face-varying value
/// (seam-preserving by construction).
pub(crate) fn all_linear_copy_corners(
    child_mesh: &Mesh,
    channel: &FaceVaryingChannel,
) -> Vec<Sparse> {
    child_mesh
        .face_vertex_indices
        .iter()
        .map(|&cv| vec![(channel.indices[cv as usize], 1.0)])
        .collect()
}

// ── Seam detection (faces-only; base-level adjacency may be empty) ──────

/// Per-edge seam flags + per-vertex boundary flags, computed only from the
/// parent faces, its complete edge list, and the channel — the base level's
/// shared `Adjacency` is empty and its `mesh.edge_vertices` may be a
/// crease-only subset, so neither is used here.
///
/// `parent_edge_vertices` is the scheme topology's complete edge list (the
/// space that child `VertexOrigin::Edge(e)` indexes). An edge is a seam if
/// its endpoints' face-varying indices differ across its two incident faces;
/// boundary edges are always seams. A vertex is a boundary vertex if any
/// incident edge is a seam.
pub(crate) fn detect_seams(
    parent_mesh: &Mesh,
    parent_edge_vertices: &[[u32; 2]],
    channel: &FaceVaryingChannel,
    p_off: &[usize],
) -> (Vec<bool>, Vec<bool>) {
    // Map each undirected face edge to its incident faces.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for fi in 0..parent_mesh.face_vertex_counts.len() {
        let n = parent_mesh.face_vertex_counts[fi] as usize;
        let s = p_off[fi];
        for k in 0..n {
            let a = parent_mesh.face_vertex_indices[s + k];
            let b = parent_mesh.face_vertex_indices[s + (k + 1) % n];
            edge_faces.entry(edge_key(a, b)).or_default().push(fi);
        }
    }

    let edge_seams: Vec<bool> = parent_edge_vertices
        .iter()
        .map(|&[v0, v1]| match edge_faces.get(&edge_key(v0, v1)) {
            Some(fs) if fs.len() >= 2 => {
                let f = |fi: usize, v: u32| fvar_in_face(parent_mesh, p_off, channel, fi, v);
                f(fs[0], v0) != f(fs[1], v0) || f(fs[0], v1) != f(fs[1], v1)
            }
            _ => true, // boundary or missing
        })
        .collect();

    let mut vert_boundary = vec![false; parent_mesh.vertex_count as usize];
    for (ei, &seam) in edge_seams.iter().enumerate() {
        if seam {
            let [a, b] = parent_edge_vertices[ei];
            vert_boundary[a as usize] = true;
            vert_boundary[b as usize] = true;
        }
    }

    (edge_seams, vert_boundary)
}

// ── Face-varying islands (home-side remap) ──────────────────────────────

/// Face-varying islands: faces partitioned into charts by non-seam adjacency,
/// plus the face-varying value each parent vertex carries within each chart.
///
/// A face-varying value at a vertex is multi-valued exactly at seams (one copy
/// per chart). Remapping a positional stencil into face-varying space must
/// therefore pick each contributing parent vertex's value **on the home face's
/// chart** — otherwise an interior point next to a seam silently bleeds the
/// other side's value. `face_island[f]` is `f`'s chart; `value_of(v, chart)`
/// is `v`'s face-varying index in that chart (if `v` borders it).
struct Islands {
    face_island: Vec<u32>,
    vert_chart_value: HashMap<(u32, u32), u32>,
}

impl Islands {
    fn build(
        parent_mesh: &Mesh,
        parent_edge_vertices: &[[u32; 2]],
        channel: &FaceVaryingChannel,
        edge_seams: &[bool],
        p_off: &[usize],
    ) -> Self {
        let face_count = parent_mesh.face_vertex_counts.len();

        // Undirected edge -> incident faces.
        let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for fi in 0..face_count {
            let n = parent_mesh.face_vertex_counts[fi] as usize;
            let s = p_off[fi];
            for k in 0..n {
                let a = parent_mesh.face_vertex_indices[s + k];
                let b = parent_mesh.face_vertex_indices[s + (k + 1) % n];
                edge_faces.entry(edge_key(a, b)).or_default().push(fi);
            }
        }

        // Seam edge keys (boundary edges are seams).
        let seam_keys: HashSet<(u32, u32)> = parent_edge_vertices
            .iter()
            .enumerate()
            .filter(|&(e, _)| edge_seams[e])
            .map(|(_, &[a, b])| edge_key(a, b))
            .collect();

        // Union faces that share a non-seam edge.
        let mut uf: Vec<usize> = (0..face_count).collect();
        for (key, faces) in &edge_faces {
            if faces.len() == 2 && !seam_keys.contains(key) {
                let ra = uf_find(&mut uf, faces[0]);
                let rb = uf_find(&mut uf, faces[1]);
                if ra != rb {
                    uf[ra] = rb;
                }
            }
        }
        let face_island: Vec<u32> = (0..face_count)
            .map(|f| uf_find(&mut uf, f) as u32)
            .collect();

        let mut vert_chart_value = HashMap::new();
        for fi in 0..face_count {
            let chart = face_island[fi];
            let n = parent_mesh.face_vertex_counts[fi] as usize;
            let s = p_off[fi];
            for k in 0..n {
                let v = parent_mesh.face_vertex_indices[s + k];
                vert_chart_value
                    .entry((v, chart))
                    .or_insert(channel.indices[s + k]);
            }
        }

        Islands {
            face_island,
            vert_chart_value,
        }
    }

    /// `v`'s face-varying value in `chart`, if `v` borders that chart.
    fn value_of(&self, v: u32, chart: u32) -> Option<u32> {
        self.vert_chart_value.get(&(v, chart)).copied()
    }
}

fn uf_find(uf: &mut [usize], mut x: usize) -> usize {
    while uf[x] != x {
        uf[x] = uf[uf[x]];
        x = uf[x];
    }
    x
}

/// The smooth-spectrum face-varying modes (`Smooth`,
/// `SmoothWithLinearCorners`, `SmoothWithLinearBoundaries`) for a scheme whose
/// child vertices carry a [`VertexOrigin`]. All three share the same interior
/// rule — the scheme's positional per-child-vertex stencils
/// (`vertex_stencils_from_level`) remapped home-side into face-varying space —
/// and differ only in how seam *vertices* are treated:
///
/// - `SmoothWithLinearBoundaries`: every seam vertex pinned (linear seam).
/// - `Smooth`: a regular seam vertex (exactly two incident seam edges) follows
///   the boundary/crease-curve rule `3/4·v + 1/8·n1 + 1/8·n2` along the seam;
///   junctions (3+ seam edges) and darts (1) fall back to the corner pin.
/// - `SmoothWithLinearCorners`: like `Smooth` but also pins face-varying
///   corners (a value used by a single face).
///
/// Seam *edges* are the crease midpoint in every smooth mode. With no seams the
/// remap is the identity and the result is the positional refinement bit for
/// bit (the parity oracle). `pos_stencils` is indexed by child vertex id.
pub(crate) fn smooth_modes(
    parent_mesh: &Mesh,
    parent_edge_vertices: &[[u32; 2]],
    child_mesh: &Mesh,
    child_lineage: &LineageMaps,
    pos_stencils: &StencilTable,
    channel: &FaceVaryingChannel,
    mode: FaceVaryingInterpolation,
) -> Vec<Sparse> {
    let p_off = face_offsets(parent_mesh);
    let (edge_seams, _vert_boundary) =
        detect_seams(parent_mesh, parent_edge_vertices, channel, &p_off);

    // Per-vertex seam neighbors: far endpoints of the incident seam edges.
    let mut seam_neighbors: Vec<Vec<u32>> = vec![Vec::new(); parent_mesh.vertex_count as usize];
    for (e, &seam) in edge_seams.iter().enumerate() {
        if seam {
            let [a, b] = parent_edge_vertices[e];
            seam_neighbors[a as usize].push(b);
            seam_neighbors[b as usize].push(a);
        }
    }

    let islands = Islands::build(
        parent_mesh,
        parent_edge_vertices,
        channel,
        &edge_seams,
        &p_off,
    );

    // Faces per face-varying value, for corner detection (value used once).
    let mut value_faces = vec![0u32; channel.value_count as usize];
    for &fv in &channel.indices {
        value_faces[fv as usize] += 1;
    }

    // Home-side remap of a positional stencil row.
    let remap = |cv: usize, home: usize, chart: u32| -> Sparse {
        let start = pos_stencils.offsets[cv] as usize;
        let end = pos_stencils.offsets[cv + 1] as usize;
        let mut sten = Sparse::new();
        for r in start..end {
            let u = pos_stencils.indices[r];
            let fu = islands
                .value_of(u, chart)
                .or_else(|| fvar_in_face(parent_mesh, &p_off, channel, home, u))
                .unwrap_or(u);
            merge(&mut sten, &[(fu, 1.0)], pos_stencils.weights[r]);
        }
        sten
    };

    let c_off = face_offsets(child_mesh);
    let mut out = Vec::with_capacity(child_mesh.face_vertex_indices.len());
    for cf in 0..child_mesh.face_vertex_counts.len() {
        let home = child_lineage.face_parent[cf] as usize;
        let chart = islands.face_island[home];
        let n = child_mesh.face_vertex_counts[cf] as usize;
        let s = c_off[cf];
        for k in 0..n {
            let cv = child_mesh.face_vertex_indices[s + k] as usize;
            let stencil = match child_lineage.vertex_origin[cv] {
                VertexOrigin::Edge(e) if edge_seams[e as usize] => {
                    // Seam edge: home-side linear midpoint.
                    let [a, b] = parent_edge_vertices[e as usize];
                    let fa = fvar_in_face(parent_mesh, &p_off, channel, home, a);
                    let fb = fvar_in_face(parent_mesh, &p_off, channel, home, b);
                    match (fa, fb) {
                        (Some(fa), Some(fb)) => vec![(fa, 0.5), (fb, 0.5)],
                        _ => vec![(fa.or(fb).unwrap_or(a), 1.0)],
                    }
                }
                VertexOrigin::Vertex(v) if !seam_neighbors[v as usize].is_empty() => {
                    seam_vertex_stencil(
                        parent_mesh,
                        &p_off,
                        channel,
                        &islands,
                        &seam_neighbors[v as usize],
                        &value_faces,
                        home,
                        chart,
                        v,
                        mode,
                    )
                }
                // A centroid (face point) is face-local: it must be remapped
                // through its OWN face's chart, not the child face's home —
                // they differ when √3 flips a seam edge, leaving a foreign
                // centroid inside this child face.
                VertexOrigin::Face(f) => remap(cv, f as usize, islands.face_island[f as usize]),
                _ => remap(cv, home, chart),
            };
            out.push(stencil);
        }
    }
    out
}

/// Seam-vertex face-varying stencil for the smooth modes (see [`smooth_modes`]).
#[allow(clippy::too_many_arguments)]
fn seam_vertex_stencil(
    parent_mesh: &Mesh,
    p_off: &[usize],
    channel: &FaceVaryingChannel,
    islands: &Islands,
    neighbors: &[u32],
    value_faces: &[u32],
    home: usize,
    chart: u32,
    v: u32,
    mode: FaceVaryingInterpolation,
) -> Sparse {
    let fv_home = fvar_in_face(parent_mesh, p_off, channel, home, v).unwrap_or(v);
    let pin = vec![(fv_home, 1.0)];

    let crease_curve = || -> Option<Sparse> {
        // Regular seam vertex: 3/4·v + 1/8·n1 + 1/8·n2 with home-side neighbors.
        if neighbors.len() != 2 {
            return None;
        }
        let f1 = islands.value_of(neighbors[0], chart)?;
        let f2 = islands.value_of(neighbors[1], chart)?;
        let mut s = Sparse::new();
        merge(&mut s, &[(fv_home, 0.75)], 1.0);
        merge(&mut s, &[(f1, 0.125)], 1.0);
        merge(&mut s, &[(f2, 0.125)], 1.0);
        Some(s)
    };

    match mode {
        FaceVaryingInterpolation::SmoothWithLinearBoundaries | FaceVaryingInterpolation::Linear => {
            pin
        }
        FaceVaryingInterpolation::SmoothWithLinearCorners if value_faces[fv_home as usize] <= 1 => {
            pin
        }
        FaceVaryingInterpolation::Smooth | FaceVaryingInterpolation::SmoothWithLinearCorners => {
            crease_curve().unwrap_or(pin)
        }
    }
}

/// Smooth face-varying stencils for Doo-Sabin. Every child vertex is a
/// face-vertex point `fv(fi, c)` whose positional stencil is **face-local**
/// (a weighted combination of face `fi`'s own corners), so it is remapped
/// through `fi`'s face-varying values directly. This is both the smooth rule
/// and seam-preserving: a face-vertex point never mixes values across a seam
/// because it only ever reads its owning face. With no seams the remap is the
/// identity, so the result equals the positional refinement (parity oracle).
///
/// Because the rule is already face-local, Doo-Sabin's
/// `Smooth`/`SmoothWithLinearBoundaries`/`SmoothWithLinearCorners` modes all
/// coincide (there is no cross-face smoothing to linearize at a seam).
pub(crate) fn smooth_doo_sabin(
    parent_mesh: &Mesh,
    child_mesh: &Mesh,
    pos_stencils: &StencilTable,
    channel: &FaceVaryingChannel,
) -> Vec<Sparse> {
    let p_off = face_offsets(parent_mesh);
    // Each face-vertex point index is a parent corner index; map it to the
    // face that owns that corner.
    let mut owner = vec![0usize; *p_off.last().unwrap_or(&0)];
    for fi in 0..parent_mesh.face_vertex_counts.len() {
        for corner in p_off[fi]..p_off[fi + 1] {
            owner[corner] = fi;
        }
    }

    child_mesh
        .face_vertex_indices
        .iter()
        .map(|&cv| {
            let cv = cv as usize;
            let fi = owner[cv];
            let start = pos_stencils.offsets[cv] as usize;
            let end = pos_stencils.offsets[cv + 1] as usize;
            let mut sten = Sparse::new();
            for r in start..end {
                let u = pos_stencils.indices[r];
                let fu = fvar_in_face(parent_mesh, &p_off, channel, fi, u).unwrap_or(u);
                merge(&mut sten, &[(fu, 1.0)], pos_stencils.weights[r]);
            }
            sten
        })
        .collect()
}
