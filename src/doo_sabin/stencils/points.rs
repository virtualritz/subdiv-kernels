//! Doo-Sabin per-point stencil rules.
//!
//! Doo-Sabin creates one *face-vertex* point per `(face, corner)` pair.
//! Each output stencil is either the Doo-Sabin face-vertex rule (smooth
//! combination of the parent face's corners) or an identity with
//! sharp/corner blending. This module owns the math; topology
//! construction for the child mesh lives in [`super::refine`].

use crate::Mesh;
use crate::SchemeOptions;
use crate::sharpness::{
    auto_corner_sharpness, merge, sharp_vertex_stencil_osd, sharpness_to_blend,
};

use super::super::topology::Topology;
use super::sparse::Sparse;

/// Compute the face-vertex stencils in `[fv_idx(fi, corner)]` order:
/// first all corners of face 0, then face 1, etc.
pub(super) fn face_vertex_stencils(
    topology: &Topology,
    topo_mesh: &Mesh,
    face_selected: &[bool],
    effective_edge_creases: &[f32],
    options: &SchemeOptions,
) -> Vec<Sparse> {
    let face_count = topology.faces.len();
    let vertex_count = topo_mesh.vertex_count as usize;

    // Pre-compute sharp vertex data.
    let sharp_vertex_data =
        compute_sharp_vertex_data(vertex_count, topology, options, effective_edge_creases);

    let total_verts: usize = (0..face_count).map(|fi| topology.faces.row_len(fi)).sum();
    let mut all_stencils: Vec<Sparse> = Vec::with_capacity(total_verts);

    for fi in 0..face_count {
        let fv = topology.faces.row(fi);
        let n = fv.len();
        let n_f32 = n as f32;

        for i in 0..n {
            let vi = fv[i] as usize;

            let smooth_stencil: Sparse = if face_selected[fi] {
                // Doo-Sabin weights.
                (0..n)
                    .map(|j| {
                        let weight = if i == j {
                            (n_f32 + 5.0) / (4.0 * n_f32)
                        } else {
                            let angle =
                                std::f32::consts::TAU * ((i as f32 - j as f32).abs()) / n_f32;
                            (3.0 + 2.0 * angle.cos()) / (4.0 * n_f32)
                        };
                        (fv[j], weight)
                    })
                    .collect()
            } else {
                // Unselected face: identity (no shrinking).
                vec![(fv[i], 1.0)]
            };

            // Apply sharp vertex blending if this vertex has n≥2 sharp edges.
            let stencil = if let Some(sharp) = &sharp_vertex_data[vi] {
                blend_toward_sharp(&smooth_stencil, &sharp.stencil, sharp.blend_factor)
            } else {
                // Corner blending (explicit or auto).
                let corner =
                    effective_corner(vi, topo_mesh, options, effective_edge_creases, topology);
                if corner > 0.0 {
                    let identity: Sparse = vec![(fv[i], 1.0)];
                    let blend = sharpness_to_blend(corner);
                    let mut result = Sparse::new();
                    merge(&mut result, &smooth_stencil, 1.0 - blend);
                    merge(&mut result, &identity, blend);
                    result
                } else {
                    smooth_stencil
                }
            };

            all_stencils.push(stencil);
        }
    }

    all_stencils
}

/// Per-vertex sharp data: pre-computed stencil and blend factor for
/// vertices with n≥2 fully sharp incident edges.
struct SharpVertexData {
    stencil: Sparse,
    blend_factor: f32,
}

/// Pre-compute sharp vertex stencils for all vertices with n≥2 sharp edges.
fn compute_sharp_vertex_data(
    vertex_count: usize,
    topology: &Topology,
    options: &SchemeOptions,
    effective_edge_creases: &[f32],
) -> Vec<Option<SharpVertexData>> {
    (0..vertex_count)
        .map(|vi| {
            let sharp_neighbors: Vec<(u32, f32)> = topology
                .vertex_edges
                .row(vi)
                .iter()
                .filter_map(|&ei| {
                    let crease = effective_edge_creases[ei as usize];
                    if crease > 0.0 {
                        let [a, b] = topology.edge_vertices[ei as usize];
                        let other = if a as usize == vi { b } else { a };
                        Some((other, crease))
                    } else {
                        None
                    }
                })
                .collect();

            let fully_sharp_count = sharp_neighbors.iter().filter(|(_, c)| *c >= 1.0).count();
            if fully_sharp_count < 2 {
                return None;
            }

            // Identity base; the smooth Doo-Sabin rule is blended in by the
            // caller via `blend_factor`, so the DeRose rule positions the
            // crease/corner relative to identity here.
            let smooth: Sparse = vec![(vi as u32, 1.0)];

            let neighbors: Vec<u32> = sharp_neighbors.iter().map(|&(n, _)| n).collect();
            let creases: Vec<f32> = sharp_neighbors.iter().map(|&(_, c)| c).collect();
            let sharp_stencil =
                sharp_vertex_stencil_osd(vi as u32, &smooth, &smooth, 0.0, &neighbors, &creases);

            // Corner blending.
            let corner = effective_corner_raw(vi, topology, options, effective_edge_creases);
            let final_stencil = if corner > 0.0 {
                let identity: Sparse = vec![(vi as u32, 1.0)];
                let blend = sharpness_to_blend(corner);
                let mut result = Sparse::new();
                merge(&mut result, &sharp_stencil, 1.0 - blend);
                merge(&mut result, &identity, blend);
                result
            } else {
                sharp_stencil
            };

            let blend_factor = sharp_neighbors
                .iter()
                .map(|(_, c)| *c)
                .fold(f32::INFINITY, f32::min)
                .min(1.0);

            Some(SharpVertexData {
                stencil: final_stencil,
                blend_factor,
            })
        })
        .collect()
}

/// Compute effective corner sharpness for a vertex (blends explicit
/// + auto-corner from incident creases).
fn effective_corner(
    vi: usize,
    topo: &Mesh,
    options: &SchemeOptions,
    effective_edge_creases: &[f32],
    topology: &Topology,
) -> f32 {
    effective_corner_raw(vi, topology, options, effective_edge_creases).max(topo.vertex_corners[vi])
}

/// Compute effective corner sharpness from topology only (no topo mesh).
fn effective_corner_raw(
    vi: usize,
    topology: &Topology,
    options: &SchemeOptions,
    effective_edge_creases: &[f32],
) -> f32 {
    if !options.auto_corner {
        return 0.0;
    }
    let incident_creases: Vec<f32> = topology
        .vertex_edges
        .row(vi)
        .iter()
        .map(|&ei| effective_edge_creases[ei as usize])
        .collect();
    auto_corner_sharpness(&incident_creases)
}

/// Blend a smooth stencil toward a sharp stencil by `blend_factor`.
fn blend_toward_sharp(smooth: &Sparse, sharp: &Sparse, blend_factor: f32) -> Sparse {
    let mut result = Sparse::new();
    merge(&mut result, smooth, 1.0 - blend_factor);
    merge(&mut result, sharp, blend_factor);
    result
}
