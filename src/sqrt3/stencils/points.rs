//! Sqrt3 per-vertex stencil rule.
//!
//! Sqrt3 refinement doesn't introduce edge points — it inserts a face
//! centroid and then smooths the existing vertices. This file holds
//! the smoothing rule for a single vertex; centroid stencils are
//! trivially computed inline by [`super::refine::vertex_stencils_from_level`].

use crate::SchemeOptions;
use crate::sharpness::{
    auto_corner_sharpness, merge, sharp_vertex_stencil_osd, sharpness_to_blend,
};

use super::super::topology::Topology;
use super::sparse::Sparse;

/// Compute the vertex-point stencil for a given vertex.
#[allow(clippy::too_many_arguments)]
pub(super) fn vertex_point_stencil(
    vi: usize,
    parent_topo: &Topology,
    vertex_corners: &[f32],
    options: &SchemeOptions,
    face_selected: &[bool],
    effective_edge_creases: &[f32],
    centroid_stencils: &[Sparse],
) -> Sparse {
    let original: Sparse = vec![(vi as u32, 1.0)];

    // Effective corner.
    let explicit_corner = vertex_corners[vi];
    let incident_creases: Vec<f32> = parent_topo
        .vertex_edges
        .row(vi)
        .iter()
        .map(|&ei| effective_edge_creases[ei as usize])
        .collect();

    let implicit_corner = if options.auto_corner {
        auto_corner_sharpness(&incident_creases)
    } else {
        0.0
    };

    let effective_corner = explicit_corner.max(implicit_corner);
    if effective_corner >= 1.0 {
        return original;
    }

    // Selected faces incident to this vertex.
    let selected_faces: Vec<usize> = parent_topo
        .vertex_faces
        .row(vi)
        .iter()
        .map(|&fi| fi as usize)
        .filter(|&fi| face_selected[fi])
        .collect();

    if selected_faces.is_empty() {
        return original;
    }

    // Boundary vertex.
    if parent_topo.vertex_is_boundary[vi] {
        let neighbors: Vec<u32> = parent_topo
            .vertex_edges
            .row(vi)
            .iter()
            .map(|&ei| ei as usize)
            .filter(|&ei| parent_topo.edge_is_boundary[ei])
            .map(|ei| {
                let [a, b] = parent_topo.edge_vertices[ei];
                if a as usize == vi { b } else { a }
            })
            .collect();

        if neighbors.len() >= 2 {
            let mut smooth: Sparse = vec![(vi as u32, 0.75)];
            merge(&mut smooth, &[(neighbors[0], 0.125)], 1.0);
            merge(&mut smooth, &[(neighbors[1], 0.125)], 1.0);

            if effective_corner > 0.0 {
                let blend = sharpness_to_blend(effective_corner);
                let mut result = Sparse::new();
                merge(&mut result, &smooth, 1.0 - blend);
                merge(&mut result, &original, blend);
                return result;
            }

            return smooth;
        }

        return original;
    }

    // Smooth Sqrt3 rule: alpha = (4 - 2*cos(2π/n)) / 9
    // position = original * (1 - alpha) + avg_centroid * alpha
    // where avg_centroid = average of centroid stencils for incident SELECTED faces.
    let n = selected_faces.len() as f32;
    let alpha = (4.0 - 2.0 * (std::f32::consts::TAU / n).cos()) / 9.0;

    let mut smooth: Sparse = vec![(vi as u32, 1.0 - alpha)];
    selected_faces.iter().for_each(|&fi| {
        merge(&mut smooth, &centroid_stencils[fi], alpha / n);
    });

    // Sharp edge handling.
    let sharp_neighbors: Vec<(u32, f32)> = parent_topo
        .vertex_edges
        .row(vi)
        .iter()
        .filter_map(|&ei| {
            let ei = ei as usize;
            let crease = effective_edge_creases[ei];
            if crease > 0.0 {
                let [a, b] = parent_topo.edge_vertices[ei];
                let other = if a as usize == vi { b } else { a };
                Some((other, crease))
            } else {
                None
            }
        })
        .collect();

    let neighbors: Vec<u32> = sharp_neighbors.iter().map(|&(n, _)| n).collect();
    let creases: Vec<f32> = sharp_neighbors.iter().map(|&(_, c)| c).collect();
    sharp_vertex_stencil_osd(
        vi as u32,
        &smooth,
        &original,
        effective_corner,
        &neighbors,
        &creases,
    )
}
