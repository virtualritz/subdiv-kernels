//! Catmull-Clark per-point stencil rules.
//!
//! Given a parent-level [`Topology`] + selection + effective creases +
//! effective vertex corners, produces sparse stencils for the face,
//! edge, and vertex points that make up the child level. These
//! stencils are pure math — no topology refinement happens here.
//! [`super::refine`] consumes them and packs the results into a
//! [`StencilTable`].

use crate::SchemeOptions;
use crate::csr::CsrVec;
use crate::options::{BoundaryInterpolation, TriangleSubdivisionRule};
use crate::sharpness::sharp_vertex_stencil_osd;

use super::super::sharpness::{auto_corner_sharpness_from_creases, sharpness_to_blend};
use super::super::topology::Topology;
use super::sparse::{Sparse, identity_entry, merge};

/// Face point stencils: centroid of face corners.
pub(super) fn face_point_stencils(faces: &CsrVec) -> Vec<Sparse> {
    (0..faces.len())
        .map(|fi| {
            let corners = faces.row(fi);
            let w = 1.0 / corners.len() as f32;
            corners.iter().map(|&v| (v, w)).collect()
        })
        .collect()
}

const CATMARK_SMOOTH_TRI_EDGE_WEIGHT: f32 = 0.470;

fn smooth_tri_edge_weights(face_size_0: usize, face_size_1: usize) -> (f32, f32) {
    let f0_tri = face_size_0 == 3;
    let f1_tri = face_size_1 == 3;

    if f0_tri || f1_tri {
        let w0 = if f0_tri {
            CATMARK_SMOOTH_TRI_EDGE_WEIGHT
        } else {
            0.25
        };
        let w1 = if f1_tri {
            CATMARK_SMOOTH_TRI_EDGE_WEIGHT
        } else {
            0.25
        };
        let face_weight = 0.5 * (w0 + w1);
        let vertex_weight = 0.5 * (1.0 - 2.0 * face_weight);
        (vertex_weight, face_weight)
    } else {
        (0.25, 0.25)
    }
}

/// Edge point stencils. Face point stencils are inlined so the result
/// references only the previous level's vertices.
#[allow(clippy::too_many_arguments)]
pub(super) fn edge_point_stencils(
    topology: &Topology,
    face_stencils: &[Sparse],
    options: &SchemeOptions,
    face_selected: &[bool],
    edge_has_selected_face: &[bool],
    effective_edge_creases: &[f32],
) -> Vec<Sparse> {
    topology
        .edge_vertices
        .iter()
        .enumerate()
        .map(|(ei, edge)| {
            let v0 = edge[0];
            let v1 = edge[1];

            // Midpoint stencil (used for boundary/crease/unselected).
            let midpoint: Sparse = Vec::from([(v0, 0.5), (v1, 0.5)]);

            if !edge_has_selected_face[ei] {
                return midpoint;
            }

            let edge_faces = topology.edge_faces.row(ei);
            if edge_faces.len() < 2 {
                return midpoint;
            }

            let f0 = edge_faces[0] as usize;
            let f1 = edge_faces[1] as usize;
            let f0_sel = face_selected[f0];
            let f1_sel = face_selected[f1];

            let smooth = match (f0_sel, f1_sel) {
                (true, true) => {
                    let use_smooth_tri = matches!(
                        options.triangle_subdivision_rule,
                        TriangleSubdivisionRule::SmoothTriangles
                    );
                    let (vw, fw) = if use_smooth_tri {
                        smooth_tri_edge_weights(
                            topology.faces.row_len(f0),
                            topology.faces.row_len(f1),
                        )
                    } else {
                        (0.25, 0.25)
                    };

                    let mut s = Sparse::new();
                    s.push((v0, vw));
                    s.push((v1, vw));
                    merge(&mut s, &face_stencils[f0], fw);
                    merge(&mut s, &face_stencils[f1], fw);
                    s
                }
                (true, false) => {
                    let w = 1.0 / 3.0;
                    let mut s = Sparse::new();
                    s.push((v0, w));
                    s.push((v1, w));
                    merge(&mut s, &face_stencils[f0], w);
                    s
                }
                (false, true) => {
                    let w = 1.0 / 3.0;
                    let mut s = Sparse::new();
                    s.push((v0, w));
                    s.push((v1, w));
                    merge(&mut s, &face_stencils[f1], w);
                    s
                }
                (false, false) => midpoint.clone(),
            };

            // Blend between smooth and midpoint by crease sharpness.
            let blend = sharpness_to_blend(effective_edge_creases[ei]);
            if blend <= 0.0 {
                smooth
            } else if blend >= 1.0 {
                midpoint
            } else {
                let mut result = Sparse::new();
                merge(&mut result, &smooth, 1.0 - blend);
                merge(&mut result, &midpoint, blend);
                result
            }
        })
        .collect()
}

/// Vertex point stencil for a single vertex.
#[allow(clippy::too_many_arguments)]
pub(super) fn vertex_point_stencil(
    vi: usize,
    topology: &Topology,
    vertex_corners: &[f32],
    options: &SchemeOptions,
    face_selected: &[bool],
    edge_has_selected_face: &[bool],
    effective_edge_creases: &[f32],
    face_stencils: &[Sparse],
) -> Sparse {
    let original = identity_entry(vi as u32);
    let explicit_corner = vertex_corners[vi];

    let incident_creases: Vec<f32> = topology
        .vertex_edges
        .row(vi)
        .iter()
        .map(|&ei| topology.edge_creases[ei as usize])
        .collect();

    let implicit_corner = if options.auto_corner {
        auto_corner_sharpness_from_creases(&incident_creases)
    } else {
        0.0
    };

    let effective_corner = explicit_corner.max(implicit_corner);
    if effective_corner >= 1.0 {
        return original;
    }

    let selected_faces: Vec<usize> = topology
        .vertex_faces
        .row(vi)
        .iter()
        .map(|&fi| fi as usize)
        .filter(|&fi| face_selected[fi])
        .collect();

    if selected_faces.is_empty() {
        return original;
    }

    let is_boundary = topology
        .vertex_edges
        .row(vi)
        .iter()
        .any(|&ei| topology.edge_faces.row_len(ei as usize) == 1);

    if is_boundary {
        match options.boundary_interpolation {
            BoundaryInterpolation::Natural => {}
            BoundaryInterpolation::EdgesOnly => {
                return boundary_vertex_stencil(vi, effective_corner, topology);
            }
            BoundaryInterpolation::EdgesAndCorners => {
                if topology.vertex_faces.row_len(vi) <= 1 {
                    return original;
                }
                return boundary_vertex_stencil(vi, effective_corner, topology);
            }
        }
    }

    let n = selected_faces.len() as f32;

    // face_avg stencil = (1/n) * sum of face point stencils for selected faces
    let mut face_avg = Sparse::new();
    selected_faces
        .iter()
        .for_each(|&fi| merge(&mut face_avg, &face_stencils[fi], 1.0 / n));

    // edge_midpoint_avg stencil
    let mut edge_mid_sum = Sparse::new();
    let mut edge_mid_count = 0usize;

    topology.vertex_edges.row(vi).iter().for_each(|&ei| {
        let ei = ei as usize;
        if !edge_has_selected_face[ei] {
            return;
        }
        let [a, b] = topology.edge_vertices[ei];
        let other = if a as usize == vi { b } else { a };
        // midpoint = 0.5 * original + 0.5 * other
        merge(&mut edge_mid_sum, &[(vi as u32, 0.5)], 1.0);
        merge(&mut edge_mid_sum, &[(other, 0.5)], 1.0);
        edge_mid_count += 1;
    });

    if edge_mid_count == 0 {
        return original;
    }

    // edge_midpoint_avg = edge_mid_sum / edge_mid_count
    let inv_m = 1.0 / edge_mid_count as f32;

    // smooth = (face_avg + 2*edge_midpoint_avg + (n-3)*original) / n
    let mut smooth = Sparse::new();
    merge(&mut smooth, &face_avg, 1.0 / n);
    merge(&mut smooth, &edge_mid_sum, 2.0 * inv_m / n);
    merge(&mut smooth, &original, (n - 3.0) / n);

    // Handle sharp edges / creases.
    let sharp_edges: Vec<(u32, f32)> = topology
        .vertex_edges
        .row(vi)
        .iter()
        .filter_map(|&ei| {
            let ei = ei as usize;
            let crease = effective_edge_creases[ei];
            if crease > 0.0 {
                let [a, b] = topology.edge_vertices[ei];
                let other = if a as usize == vi { b } else { a };
                Some((other, crease))
            } else {
                None
            }
        })
        .collect();

    if sharp_edges.len() >= 2 {
        let neighbors: Vec<u32> = sharp_edges.iter().map(|&(n, _)| n).collect();
        let creases: Vec<f32> = sharp_edges.iter().map(|&(_, c)| c).collect();
        return sharp_vertex_stencil_osd(
            vi as u32,
            &smooth,
            &original,
            effective_corner,
            &neighbors,
            &creases,
        );
    }

    if effective_corner > 0.0 {
        let blend = sharpness_to_blend(effective_corner);
        let mut result = Sparse::new();
        merge(&mut result, &smooth, 1.0 - blend);
        merge(&mut result, &original, blend);
        return result;
    }

    smooth
}

fn boundary_vertex_stencil(vi: usize, effective_corner: f32, topology: &Topology) -> Sparse {
    let original = identity_entry(vi as u32);
    let neighbors: Vec<usize> = topology
        .vertex_edges
        .row(vi)
        .iter()
        .filter_map(|&ei| {
            let ei = ei as usize;
            if topology.edge_faces.row_len(ei) == 1 {
                let [a, b] = topology.edge_vertices[ei];
                let other = if a as usize == vi { b } else { a } as usize;
                Some(other)
            } else {
                None
            }
        })
        .collect();

    if neighbors.len() < 2 {
        return original;
    }

    // boundary vertex rule: 0.75 * original + 0.125 * n0 + 0.125 * n1
    let mut smooth = Sparse::new();
    smooth.push((vi as u32, 0.75));
    smooth.push((neighbors[0] as u32, 0.125));
    smooth.push((neighbors[1] as u32, 0.125));

    if effective_corner > 0.0 {
        let blend = sharpness_to_blend(effective_corner);
        let mut result = Sparse::new();
        merge(&mut result, &smooth, 1.0 - blend);
        merge(&mut result, &original, blend);
        result
    } else {
        smooth
    }
}

