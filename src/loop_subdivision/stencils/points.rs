//! Loop per-point stencil rules.
//!
//! Given a parent-level [`Topology`] + selection + effective creases +
//! effective vertex corners, produces sparse stencils for the edge and
//! vertex points that make up the child level. Pure math; topology
//! refinement happens in [`super::refine`].

use crate::sharpness::sharp_vertex_stencil_osd;

use super::super::topology::Topology;
use super::sparse::{Sparse, merge};

pub(super) fn edge_point_stencil(
    ei: usize,
    topology: &Topology,
    effective_edge_creases: &[f32],
) -> Sparse {
    let [v0, v1] = topology.edge_vertices[ei];
    let midpoint: Sparse = vec![(v0, 0.5), (v1, 0.5)];

    if topology.edge_is_boundary[ei] {
        return midpoint;
    }

    let [f0, f1] = topology.edge_faces[ei];
    if f1 == u32::MAX {
        return midpoint;
    }

    let opposite_0 = face_opposite_vertex(topology.faces.row(f0 as usize), v0, v1);
    let opposite_1 = face_opposite_vertex(topology.faces.row(f1 as usize), v0, v1);

    let smooth = match (opposite_0, opposite_1) {
        (Some(o0), Some(o1)) => {
            // Loop edge rule: 3/8*(v0+v1) + 1/8*(o0+o1)
            vec![(v0, 0.375), (v1, 0.375), (o0, 0.125), (o1, 0.125)]
        }
        _ => midpoint.clone(),
    };

    let crease = effective_edge_creases[ei].clamp(0.0, 1.0);
    if crease <= 0.0 {
        smooth
    } else if crease >= 1.0 {
        midpoint
    } else {
        let mut result = Sparse::new();
        merge(&mut result, &smooth, 1.0 - crease);
        merge(&mut result, &midpoint, crease);
        result
    }
}

pub(super) fn vertex_point_stencil(
    vi: usize,
    topology: &Topology,
    face_selected: &[bool],
    effective_edge_creases: &[f32],
    effective_vertex_corners: &[f32],
) -> Sparse {
    let vi32 = vi as u32;
    let original: Sparse = vec![(vi32, 1.0)];
    let corner = effective_vertex_corners[vi];

    if corner >= 1.0 {
        return original;
    }

    let incident_edges = topology.vertex_edges.row(vi);
    let incident_faces = topology.vertex_faces.row(vi);

    let selected_count = incident_faces
        .iter()
        .filter(|&&fi| face_selected[fi as usize])
        .count();

    if selected_count == 0 {
        return original;
    }

    // Boundary vertex
    if topology.vertex_is_boundary[vi] {
        let mut bn0 = None;
        let mut bn1 = None;

        incident_edges.iter().for_each(|&ei| {
            if bn1.is_some() {
                return;
            }
            let ei = ei as usize;
            if topology.edge_is_boundary[ei] {
                let [a, b] = topology.edge_vertices[ei];
                let other = if a == vi32 { b } else { a };
                if bn0.is_none() {
                    bn0 = Some(other);
                } else if bn1.is_none() {
                    bn1 = Some(other);
                }
            }
        });

        if let (Some(n0), Some(n1)) = (bn0, bn1) {
            // Boundary rule: 0.75*v + 0.125*n0 + 0.125*n1
            let smooth: Sparse = vec![(vi32, 0.75), (n0, 0.125), (n1, 0.125)];

            if corner > 0.0 {
                let blend = corner.clamp(0.0, 1.0);
                let mut result = Sparse::new();
                merge(&mut result, &smooth, 1.0 - blend);
                merge(&mut result, &original, blend);
                return result;
            }

            return smooth;
        }

        return original;
    }

    if incident_edges.is_empty() {
        return original;
    }

    // Smooth Loop vertex rule: (1 - n*beta)*v + beta*sum(neighbors)
    let n = incident_edges.len() as f32;
    let beta = loop_beta(n);

    let mut smooth = Sparse::new();
    smooth.push((vi32, 1.0 - n * beta));
    incident_edges.iter().for_each(|&ei| {
        let [a, b] = topology.edge_vertices[ei as usize];
        let other = if a == vi32 { b } else { a };
        merge(&mut smooth, &[(other, beta)], 1.0);
    });

    // Sharp edge handling
    let sharp_neighbors: Vec<(u32, f32)> = incident_edges
        .iter()
        .filter_map(|&ei| {
            let crease = effective_edge_creases[ei as usize];
            if crease > 0.0 {
                let [a, b] = topology.edge_vertices[ei as usize];
                let other = if a == vi32 { b } else { a };
                Some((other, crease))
            } else {
                None
            }
        })
        .collect();

    let neighbors: Vec<u32> = sharp_neighbors.iter().map(|&(n, _)| n).collect();
    let creases: Vec<f32> = sharp_neighbors.iter().map(|&(_, c)| c).collect();
    sharp_vertex_stencil_osd(vi32, &smooth, &original, corner, &neighbors, &creases)
}

#[inline]
fn loop_beta(n: f32) -> f32 {
    if n == 3.0 {
        3.0 / 16.0
    } else {
        3.0 / (8.0 * n)
    }
}

#[inline]
fn face_opposite_vertex(face_vertices: &[u32], v0: u32, v1: u32) -> Option<u32> {
    face_vertices.iter().find(|&&v| v != v0 && v != v1).copied()
}
